import asyncio
import json
import logging
import signal
import ssl
import sys

import certifi
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_WS_URL = "wss://data-stream.binance.vision"
SYMBOL = "btcusdt"
STREAMS = [
    f"{SYMBOL}@bookTicker",
]
RECONNECT_DELAY_S = 3
MAX_RECONNECT_DELAY_S = 60


def _handle_message(raw: str) -> None:
    msg = json.loads(raw)
    data = msg.get("data", msg)
    bid = float(data["b"])
    ask = float(data["a"])
    mid = (bid + ask) / 2
    log.info("[MID] %.2f", mid)


async def stream(shutdown_event: asyncio.Event) -> None:
    url = f"{BASE_WS_URL}/stream?streams={'/'.join(STREAMS)}"
    delay = RECONNECT_DELAY_S

    while not shutdown_event.is_set():
        try:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            log.info("connecting to %s", url)
            async with websockets.connect(url, ssl=ssl_ctx, ping_interval=20, ping_timeout=60) as ws:
                log.info("connected — streaming %s spot data", SYMBOL.upper())
                delay = RECONNECT_DELAY_S
                async for raw in ws:
                    if shutdown_event.is_set():
                        break
                    _handle_message(raw)
        except (websockets.ConnectionClosed, websockets.InvalidURI) as e:
            log.warning("connection closed: %s", e)
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
