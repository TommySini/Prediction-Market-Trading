from __future__ import annotations

import asyncio
import csv
import json
import logging
import signal
import ssl
import time
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
import certifi
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────

BINANCE_WS = "wss://data-stream.binance.vision"
SYMBOL = "btcusdt"

GAMMA_API = "https://gamma-api.polymarket.com"
POLY_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
ET = ZoneInfo("America/New_York")

RECONNECT_DELAY_S = 3
MAX_RECONNECT_DELAY_S = 60

CSV_HEADER = [
    "ts_wall_iso",
    "ts_mono_ns",
    "source",
    "binance_bid",
    "binance_ask",
    "binance_mid",
    "poly_yes_bid",
    "poly_yes_ask",
    "poly_no_bid",
    "poly_no_ask",
    "age_binance_ms",
    "age_poly_ms",
    "market_slug",
]


# ── Shared snapshots ─────────────────────────────────────────────────


class Snapshot:
    __slots__ = (
        "binance_bid", "binance_ask", "binance_mid", "binance_ts",
        "poly_yes_bid", "poly_yes_ask",
        "poly_no_bid", "poly_no_ask", "poly_ts",
        "yes_token_id", "no_token_id",
        "market_slug",
    )

    def __init__(self) -> None:
        self.binance_bid: Optional[float] = None
        self.binance_ask: Optional[float] = None
        self.binance_mid: Optional[float] = None
        self.binance_ts: Optional[int] = None

        self.poly_yes_bid: Optional[float] = None
        self.poly_yes_ask: Optional[float] = None
        self.poly_no_bid: Optional[float] = None
        self.poly_no_ask: Optional[float] = None
        self.poly_ts: Optional[int] = None

        self.yes_token_id: Optional[str] = None
        self.no_token_id: Optional[str] = None
        self.market_slug: str = ""


# ── Market resolution (from polymarket_stream) ───────────────────────


def _build_slug() -> tuple[str, datetime]:
    now = datetime.now(ET)
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    next_hour = hour_start + timedelta(hours=1)
    month = hour_start.strftime("%B").lower()
    day = hour_start.day
    hour_12 = hour_start.strftime("%I").lstrip("0")
    ampm = hour_start.strftime("%p").lower()
    slug = f"bitcoin-up-or-down-{month}-{day}-{hour_12}{ampm}-et"
    return slug, next_hour


async def _fetch_market(slug: str) -> Optional[dict]:
    url = f"{GAMMA_API}/events/slug/{slug}"
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=ssl_ctx) as resp:
                if resp.status != 200:
                    log.warning("gamma API %d for slug=%s", resp.status, slug)
                    return None
                event = await resp.json()
    except Exception:
        log.exception("failed to fetch event slug=%s", slug)
        return None

    raw_markets = event.get("markets", [])
    if not raw_markets:
        log.warning("no markets in event slug=%s", slug)
        return None

    tokens: dict[str, str] = {}
    question = slug
    for m in raw_markets:
        if isinstance(m, str):
            m = json.loads(m)
        question = m.get("question", question)
        raw_outcomes = m.get("outcomes", "[]")
        raw_clob_ids = m.get("clobTokenIds", "[]")
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        clob_ids = json.loads(raw_clob_ids) if isinstance(raw_clob_ids, str) else raw_clob_ids
        for tid, outcome in zip(clob_ids, outcomes):
            tokens[tid] = outcome
        for tok in m.get("tokens", []):
            tid = tok.get("token_id") or tok.get("tokenId")
            outcome = tok.get("outcome", "?")
            if tid and tid not in tokens:
                tokens[tid] = outcome

    if not tokens:
        log.warning("no tokens found for slug=%s", slug)
        return None
    return {"slug": slug, "tokens": tokens, "question": question}


# ── Polymarket WS task ───────────────────────────────────────────────


async def _polymarket_loop(
    shutdown: asyncio.Event,
    snap: Snapshot,
    queue: asyncio.Queue,
) -> None:
    delay = RECONNECT_DELAY_S
    current_slug: Optional[str] = None
    market: Optional[dict] = None

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
            snap.market_slug = slug
            snap.yes_token_id = None
            snap.no_token_id = None
            snap.poly_yes_bid = None
            snap.poly_yes_ask = None
            snap.poly_no_bid = None
            snap.poly_no_ask = None

            for tid, outcome in market["tokens"].items():
                norm = str(outcome).strip().lower()
                if norm in ("yes", "up"):
                    snap.yes_token_id = tid
                elif norm in ("no", "down"):
                    snap.no_token_id = tid

            log.info("market: %s", market["question"])
            log.info("  YES token=%s", snap.yes_token_id and snap.yes_token_id[:16])
            log.info("  NO  token=%s", snap.no_token_id and snap.no_token_id[:16])

        token_ids = list(market["tokens"].keys())
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        try:
            async with websockets.connect(
                POLY_WS, ssl=ssl_ctx, ping_interval=20, ping_timeout=60,
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

                        if asset_id == snap.yes_token_id:
                            if best_ask is not None:
                                snap.poly_yes_ask = best_ask
                            if best_bid is not None:
                                snap.poly_yes_bid = best_bid
                        elif asset_id == snap.no_token_id:
                            if best_ask is not None:
                                snap.poly_no_ask = best_ask
                            if best_bid is not None:
                                snap.poly_no_bid = best_bid

                        snap.poly_ts = time.monotonic_ns()
                        queue.put_nowait("polymarket")

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


# ── Binance WS task ──────────────────────────────────────────────────


async def _binance_loop(
    shutdown: asyncio.Event,
    snap: Snapshot,
    queue: asyncio.Queue,
) -> None:
    url = f"{BINANCE_WS}/ws/{SYMBOL}@bookTicker"
    delay = RECONNECT_DELAY_S

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
                    snap.binance_bid = float(data["b"])
                    snap.binance_ask = float(data["a"])
                    snap.binance_mid = (snap.binance_bid + snap.binance_ask) * 0.5
                    snap.binance_ts = time.monotonic_ns()
                    queue.put_nowait("binance")

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


# ── CSV writer task ──────────────────────────────────────────────────


def _age_ms(now_ns: int, ts_ns: Optional[int]) -> str:
    if ts_ns is None:
        return ""
    return f"{(now_ns - ts_ns) / 1_000_000:.1f}"


async def _writer_loop(
    shutdown: asyncio.Event,
    snap: Snapshot,
    queue: asyncio.Queue,
) -> None:
    fname = datetime.now().strftime("stream_%Y%m%d_%H%M%S.csv")
    log.info("writing CSV to %s", fname)

    with open(fname, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        fh.flush()

        while not shutdown.is_set():
            try:
                source = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            now_ns = time.monotonic_ns()
            writer.writerow([
                datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                now_ns,
                source,
                snap.binance_bid if snap.binance_bid is not None else "",
                snap.binance_ask if snap.binance_ask is not None else "",
                snap.binance_mid if snap.binance_mid is not None else "",
                snap.poly_yes_bid if snap.poly_yes_bid is not None else "",
                snap.poly_yes_ask if snap.poly_yes_ask is not None else "",
                snap.poly_no_bid if snap.poly_no_bid is not None else "",
                snap.poly_no_ask if snap.poly_no_ask is not None else "",
                _age_ms(now_ns, snap.binance_ts),
                _age_ms(now_ns, snap.poly_ts),
                snap.market_slug,
            ])
            fh.flush()


# ── Entrypoint ───────────────────────────────────────────────────────


async def _main_async() -> None:
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    snap = Snapshot()
    queue: asyncio.Queue = asyncio.Queue()

    await asyncio.gather(
        _polymarket_loop(shutdown, snap, queue),
        _binance_loop(shutdown, snap, queue),
        _writer_loop(shutdown, snap, queue),
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
