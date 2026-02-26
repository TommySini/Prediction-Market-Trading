import asyncio
import json
import logging
import signal
import ssl
from datetime import datetime, timedelta
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

GAMMA_API = "https://gamma-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
ET = ZoneInfo("America/New_York")
RECONNECT_DELAY_S = 3
MAX_RECONNECT_DELAY_S = 60


def _build_slug() -> tuple[str, datetime]:
    """Return (slug, next_hour_boundary) for the current BTC hourly market."""
    now = datetime.now(ET)
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    next_hour = hour_start + timedelta(hours=1)

    month = hour_start.strftime("%B").lower()
    day = hour_start.day
    hour_12 = hour_start.strftime("%I").lstrip("0")
    ampm = hour_start.strftime("%p").lower()

    slug = f"bitcoin-up-or-down-{month}-{day}-{hour_12}{ampm}-et"
    return slug, next_hour


async def _fetch_market(slug: str) -> dict | None:
    """Fetch event from the Gamma API and extract token IDs for the WS subscription."""
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
    condition_id = None
    question = slug

    for m in raw_markets:
        if isinstance(m, str):
            m = json.loads(m)
        cid = m.get("condition_id") or m.get("conditionId")
        if cid:
            condition_id = cid
        question = m.get("question", question)

        # Gamma API returns outcomes / clobTokenIds as JSON-encoded strings
        raw_outcomes = m.get("outcomes", "[]")
        raw_clob_ids = m.get("clobTokenIds", "[]")
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        clob_ids = json.loads(raw_clob_ids) if isinstance(raw_clob_ids, str) else raw_clob_ids

        for tid, outcome in zip(clob_ids, outcomes):
            tokens[tid] = outcome

        # Fallback: nested tokens array (CLOB API shape)
        for tok in m.get("tokens", []):
            tid = tok.get("token_id") or tok.get("tokenId")
            outcome = tok.get("outcome", "?")
            if tid and tid not in tokens:
                tokens[tid] = outcome

    if not tokens:
        log.warning("no tokens found for slug=%s", slug)
        return None

    return {
        "slug": slug,
        "condition_id": condition_id,
        "tokens": tokens,
        "question": question,
    }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_book(data: dict, tokens: dict[str, str]) -> str:
    outcome = tokens.get(data.get("asset_id", ""), "?")
    bids = data.get("bids", data.get("buys", []))
    asks = data.get("asks", data.get("sells", []))
    best_bid = bids[0]["price"] if bids else "-"
    best_ask = asks[0]["price"] if asks else "-"
    depth_b = sum(float(l["size"]) for l in bids)
    depth_a = sum(float(l["size"]) for l in asks)
    return (
        f"[BOOK]   {outcome:>4}  bid={best_bid}  ask={best_ask}"
        f"  depth_bid={depth_b:.0f}  depth_ask={depth_a:.0f}"
    )


def _fmt_price(data: dict, tokens: dict[str, str]) -> str:
    parts = []
    for ch in data.get("price_changes", []):
        outcome = tokens.get(ch.get("asset_id", ""), "?")
        parts.append(
            f"{outcome}: best_bid={ch.get('best_bid', '-')}"
            f" best_ask={ch.get('best_ask', '-')}"
            f" Δpx={ch.get('price', '-')}"
            f" sz={ch.get('size', '-')}"
            f" side={ch.get('side', '-')}"
        )
    return "[PRICE]  " + " | ".join(parts)


def _fmt_trade(data: dict, tokens: dict[str, str]) -> str:
    outcome = tokens.get(data.get("asset_id", ""), "?")
    return (
        f"[TRADE]  {outcome:>4}  price={data.get('price', '-')}"
        f"  size={data.get('size', '-')}  side={data.get('side', '-')}"
    )


_FMT = {
    "book": _fmt_book,
    "price_change": _fmt_price,
    "last_trade_price": _fmt_trade,
}


def _handle_message(raw: str, tokens: dict[str, str]) -> None:
    msg = json.loads(raw)
    items = msg if isinstance(msg, list) else [msg]
    for data in items:
        fmt = _FMT.get(data.get("event_type", ""))
        if fmt:
            log.info(fmt(data, tokens))


# ---------------------------------------------------------------------------
# Main stream loop
# ---------------------------------------------------------------------------

async def stream(shutdown_event: asyncio.Event) -> None:
    delay = RECONNECT_DELAY_S
    current_slug: str | None = None
    market: dict | None = None
    next_hour = datetime.now(ET)

    while not shutdown_event.is_set():
        slug, next_hour = _build_slug()

        if slug != current_slug:
            log.info("resolving market: %s", slug)
            market = await _fetch_market(slug)
            if market is None:
                log.warning("market not found, retrying in %ds …", delay)
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, MAX_RECONNECT_DELAY_S)
                continue
            current_slug = slug
            delay = RECONNECT_DELAY_S
            log.info("market: %s", market["question"])
            for tid, outcome in market["tokens"].items():
                log.info("  %-5s  token=%s…", outcome, tid[:16])

        token_ids = list(market["tokens"].keys())
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        try:
            log.info("connecting to polymarket ws …")
            async with websockets.connect(
                WS_URL, ssl=ssl_ctx, ping_interval=20, ping_timeout=60
            ) as ws:
                await ws.send(json.dumps({
                    "type": "market",
                    "assets_ids": token_ids,
                }))
                log.info("subscribed — streaming %s", current_slug)
                delay = RECONNECT_DELAY_S

                async for raw in ws:
                    if shutdown_event.is_set():
                        break
                    if datetime.now(ET) >= next_hour:
                        log.info("hour rolled over — switching market")
                        current_slug = None
                        break
                    _handle_message(raw, market["tokens"])

        except (websockets.ConnectionClosed, websockets.InvalidURI) as e:
            log.warning("ws closed: %s", e)
        except Exception:
            log.exception("unexpected error")

        if not shutdown_event.is_set():
            log.info("reconnecting in %ds …", delay)
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, MAX_RECONNECT_DELAY_S)


def main() -> None:
    shutdown = asyncio.Event()

    def _signal_handler() -> None:
        log.info("shutting down …")
        shutdown.set()

    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(stream(shutdown))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
