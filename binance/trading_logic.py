from __future__ import annotations

import asyncio
import json
import logging
import math
import signal
import ssl
import time
from datetime import datetime

import certifi
import websockets

import user_auth
from polymarket_stream import WS_URL, ET, _build_slug, _fetch_market

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────

BINANCE_WS = "wss://data-stream.binance.vision"
SYMBOL = "btcusdt"
THRESHOLD_USD = 0.08
ORDER_SIZE = 5.0
MIN_NOTIONAL = 1.0
BUY_TTL_S = 5
BUY_POLL_INTERVAL_S = 0.5
SELL_DELAY_S = 3
RECONNECT_DELAY_S = 3
MAX_RECONNECT_DELAY_S = 60


# ── Shared in-memory state ───────────────────────────────────────────


class BookState:
    """Holds latest Polymarket best bid/ask for YES and NO outcomes."""

    __slots__ = (
        "yes_token_id", "no_token_id",
        "best_ask_yes", "best_ask_no",
        "best_bid_yes", "best_bid_no",
        "updated_ts",
    )

    def __init__(self) -> None:
        self.yes_token_id: str | None = None
        self.no_token_id: str | None = None
        self.best_ask_yes: float | None = None
        self.best_ask_no: float | None = None
        self.best_bid_yes: float | None = None
        self.best_bid_no: float | None = None
        self.updated_ts: float = 0.0


# ── Polymarket WS task ───────────────────────────────────────────────


async def _polymarket_loop(shutdown: asyncio.Event, state: BookState) -> None:
    delay = RECONNECT_DELAY_S
    current_slug: str | None = None

    while not shutdown.is_set():
        slug, next_hour = _build_slug()

        if slug != current_slug:
            log.info("resolving polymarket market: %s", slug)
            market = await _fetch_market(slug)
            if market is None:
                log.warning("market not found, retrying in %ds", delay)
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, MAX_RECONNECT_DELAY_S)
                continue

            current_slug = slug
            delay = RECONNECT_DELAY_S
            tokens: dict[str, str] = market["tokens"]

            state.yes_token_id = None
            state.no_token_id = None
            state.best_ask_yes = None
            state.best_ask_no = None
            state.best_bid_yes = None
            state.best_bid_no = None

            for tid, outcome in tokens.items():
                normalized = str(outcome).strip().lower()
                if normalized == "yes" or normalized == "up":
                    state.yes_token_id = tid
                elif normalized == "no" or normalized == "down":
                    state.no_token_id = tid

            log.info("market: %s", market["question"])
            log.info("  YES token=%s", state.yes_token_id and state.yes_token_id[:16])
            log.info("  NO  token=%s", state.no_token_id and state.no_token_id[:16])

        token_ids = list(market["tokens"].keys())
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        try:
            async with websockets.connect(
                WS_URL, ssl=ssl_ctx, ping_interval=20, ping_timeout=60,
            ) as ws:
                await ws.send(json.dumps({"type": "market", "assets_ids": token_ids}))
                log.info("polymarket subscribed (%s)", current_slug)

                async for raw in ws:
                    if shutdown.is_set():
                        return

                    if datetime.now(ET) >= next_hour:
                        log.info("hour rolled over — switching market")
                        current_slug = None
                        break

                    msg = json.loads(raw)
                    items = msg if isinstance(msg, list) else [msg]
                    for data in items:
                        if data.get("event_type") != "book":
                            continue
                        asset_id = data.get("asset_id", "")
                        asks = data.get("asks") or data.get("sells") or []
                        bids = data.get("bids") or data.get("buys") or []

                        best_ask = min((float(o["price"]) for o in asks), default=None)
                        best_bid = max((float(o["price"]) for o in bids), default=None)

                        if asset_id == state.yes_token_id:
                            if best_ask is not None:
                                state.best_ask_yes = best_ask
                            if best_bid is not None:
                                state.best_bid_yes = best_bid
                        elif asset_id == state.no_token_id:
                            if best_ask is not None:
                                state.best_ask_no = best_ask
                            if best_bid is not None:
                                state.best_bid_no = best_bid

                        state.updated_ts = time.monotonic()

        except (websockets.ConnectionClosed, websockets.InvalidURI) as e:
            log.warning("polymarket ws closed: %s", e)
        except Exception:
            log.exception("polymarket ws error")

        if not shutdown.is_set():
            log.info("polymarket reconnecting in %ds", delay)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, MAX_RECONNECT_DELAY_S)


# ── Binance WS task + jump detection ────────────────────────────────


async def _binance_loop(
    shutdown: asyncio.Event,
    state: BookState,
    client,
) -> None:
    url = f"{BINANCE_WS}/ws/{SYMBOL}@bookTicker"
    delay = RECONNECT_DELAY_S
    prev_mid: float | None = None

    while not shutdown.is_set():
        try:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            log.info("connecting to binance %s", SYMBOL.upper())
            async with websockets.connect(
                url, ssl=ssl_ctx, ping_interval=20, ping_timeout=60,
            ) as ws:
                log.info("binance connected — streaming %s bookTicker", SYMBOL.upper())
                delay = RECONNECT_DELAY_S

                async for raw in ws:
                    if shutdown.is_set():
                        return

                    data = json.loads(raw)
                    bid = float(data["b"])
                    ask = float(data["a"])
                    mid = (bid + ask) * 0.5

                    prev = prev_mid
                    prev_mid = mid
                    if prev is None:
                        continue

                    delta = mid - prev

                    if delta >= THRESHOLD_USD:
                        token_id = state.yes_token_id
                        best_ask = state.best_ask_yes
                        side_label = "YES"
                    elif delta <= -THRESHOLD_USD:
                        token_id = state.no_token_id
                        best_ask = state.best_ask_no
                        side_label = "NO"
                    else:
                        continue

                    if token_id is None or best_ask is None:
                        log.warning(
                            "jump detected (delta=%+.2f) but %s ask not available",
                            delta, side_label,
                        )
                        continue

                    if not (0.02 <= best_ask <= 0.90):
                        log.info(
                            "[SKIP] delta=%+.2f %s ask=%.4f outside 0.02–0.90 range",
                            delta, side_label, best_ask,
                        )
                        continue

                    log.info(
                        "[JUMP] delta=%+.2f mid=%.2f → BUY %s @ %.4f x%.0f",
                        delta, mid, side_label, best_ask, ORDER_SIZE,
                    )

                    asyncio.create_task(
                        _place_order(client, state, token_id, best_ask, side_label, delta, mid)
                    )

        except (websockets.ConnectionClosed, websockets.InvalidURI) as e:
            log.warning("binance ws closed: %s", e)
        except Exception:
            log.exception("binance ws error")

        if not shutdown.is_set():
            log.info("binance reconnecting in %ds", delay)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, MAX_RECONNECT_DELAY_S)


async def _place_order(
    client,
    state: BookState,
    token_id: str,
    price: float,
    side_label: str,
    delta: float,
    mid: float,
) -> None:
    # ── Effective size: at least ORDER_SIZE, bumped up if notional < MIN_NOTIONAL ──
    size = ORDER_SIZE if price * ORDER_SIZE >= MIN_NOTIONAL else math.ceil(MIN_NOTIONAL / price)

    # ── BUY ──
    try:
        resp = await asyncio.to_thread(
            user_auth.place_limit_order,
            client, token_id, "BUY", price, size,
        )
        log.info("[ORDER] BUY %s posted (size=%.0f): %s", side_label, size, resp)
    except Exception:
        log.exception("[ORDER] BUY %s failed (delta=%+.2f mid=%.2f)", side_label, delta, mid)
        return

    order_id = resp.get("orderID") or resp.get("id")
    if not order_id:
        log.error("[ORDER] BUY %s — no order ID in response: %s", side_label, resp)
        return

    # ── Poll for fill (up to BUY_TTL_S) ──
    filled = 0.0
    deadline = time.monotonic() + BUY_TTL_S
    while time.monotonic() < deadline:
        await asyncio.sleep(BUY_POLL_INTERVAL_S)
        try:
            order = await asyncio.to_thread(client.get_order, order_id)
            filled = float(order.get("size_matched", 0))
            status = order.get("status", "")
            if filled >= size or status not in ("LIVE", "OPEN", ""):
                break
        except Exception:
            log.warning("[POLL] failed to fetch order %s", order_id)

    # ── Cancel unfilled remainder ──
    if filled < size:
        try:
            await asyncio.to_thread(client.cancel, order_id)
            log.info("[CANCEL] BUY %s remainder cancelled (filled=%.2f/%.0f)", side_label, filled, size)
        except Exception:
            log.warning("[CANCEL] BUY %s cancel failed (may already be fully filled)", side_label)

    if filled <= 0:
        log.info("[ORDER] BUY %s nothing filled after %ds — aborting", side_label, BUY_TTL_S)
        return

    log.info("[ORDER] BUY %s filled %.2f shares", side_label, filled)

    # ── SELL after holding delay ──
    await asyncio.sleep(SELL_DELAY_S)

    best_bid = state.best_bid_yes if side_label == "YES" else state.best_bid_no
    if best_bid is None:
        log.warning("[SELL] %s best bid not available — skipping sell", side_label)
        return

    log.info("[SELL] %s → SELL @ %.4f x%.2f", side_label, best_bid, filled)
    try:
        resp = await asyncio.to_thread(
            user_auth.place_limit_order,
            client, token_id, "SELL", best_bid, filled,
        )
        log.info("[ORDER] SELL %s posted: %s", side_label, resp)
    except Exception:
        log.exception("[ORDER] SELL %s failed (price=%.4f size=%.2f)", side_label, best_bid, filled)


# ── Entrypoint ───────────────────────────────────────────────────────


async def _main_async() -> None:
    shutdown = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    state = BookState()
    client = user_auth.build_client()

    await asyncio.gather(
        _polymarket_loop(shutdown, state),
        _binance_loop(shutdown, state, client),
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
