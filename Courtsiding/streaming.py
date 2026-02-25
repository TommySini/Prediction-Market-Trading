# streaming.py
from __future__ import annotations

import asyncio
import json
import time
import sys
import select
import termios
import tty
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict

from clients import KalshiWebSocketClient, KalshiHttpClient
import requests


# -----------------------------
# OrderBook (in-memory state)
# -----------------------------
@dataclass
class OrderBook:
    market_ticker: str
    yes_levels: Dict[int, int] = field(default_factory=dict)
    no_levels: Dict[int, int] = field(default_factory=dict)

    best_yes: Optional[int] = None
    best_no: Optional[int] = None

    sid: Optional[int] = None
    last_seq: Optional[int] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def apply_snapshot(self, *, yes: List[List[int]], no: List[List[int]], sid: int, seq: int) -> None:
        async with self.lock:
            self.sid = sid
            self.last_seq = seq

            self.yes_levels.clear()
            self.no_levels.clear()

            by = None
            bn = None

            for price, qty in yes:
                qty = int(qty)
                if qty > 0:
                    p = int(price)
                    self.yes_levels[p] = qty
                    if by is None or p > by:
                        by = p

            for price, qty in no:
                qty = int(qty)
                if qty > 0:
                    p = int(price)
                    self.no_levels[p] = qty
                    if bn is None or p > bn:
                        bn = p

            self.best_yes = by
            self.best_no = bn

    async def apply_delta(self, *, side: str, price: int, delta: int, sid: int, seq: int) -> bool:
        """
        Returns False if we detect a gap and must resync.
        """
        async with self.lock:
            # If Kalshi switches sid (new subscription), reset sequencing expectations
            if self.sid is None or self.sid != sid:
                self.sid = sid
                self.last_seq = None

            # If we have sequencing and this isn't the next message -> gap
            if self.last_seq is not None and seq != self.last_seq + 1:
                return False

            self.last_seq = seq

            book = self.yes_levels if side == "yes" else self.no_levels
            best = self.best_yes if side == "yes" else self.best_no

            new_qty = book.get(price, 0) + delta

            if new_qty <= 0:
                if price in book:
                    del book[price]
                if best == price:
                    best = max(book.keys()) if book else None
            else:
                book[price] = new_qty
                if best is None or price > best:
                    best = price

            if side == "yes":
                self.best_yes = best
            else:
                self.best_no = best

            return True

    def best_yes_bid(self) -> Optional[int]:
        return self.best_yes

    def best_no_bid(self) -> Optional[int]:
        return self.best_no

    def best_yes_ask_implied(self) -> Optional[int]:
        # YES ask = 100 - best NO bid
        if self.best_no is None:
            return None
        return 100 - self.best_no

    def best_no_ask_implied(self) -> Optional[int]:
        # NO ask = 100 - best YES bid
        if self.best_yes is None:
            return None
        return 100 - self.best_yes

    def best_yes_bid_direct(self) -> Optional[int]:
        """Returns the best YES bid (for selling YES)."""
        return self.best_yes

    def best_no_bid_direct(self) -> Optional[int]:
        """Returns the best NO bid (for selling NO)."""
        return self.best_no


# -----------------------------
# Websocket Streamer
# -----------------------------
class MarketDataStreamer(KalshiWebSocketClient):
    def __init__(
        self,
        key_id: str,
        private_key,
        market_ticker: str,
        position_tracker: Optional[PositionTracker] = None,
        http_client: Optional[KalshiHttpClient] = None,
    ):
        super().__init__(key_id=key_id, private_key=private_key)
        self.market_ticker = market_ticker
        self.book = OrderBook(market_ticker=market_ticker)
        self.position_tracker = position_tracker
        self.http_client = http_client

        self._cmd_id = 1
        self._orderbook_sid: Optional[int] = None  # IMPORTANT: subscription id for orderbook_delta
        self._user_fills_sid: Optional[int] = None  # subscription id for user_fills

    def _next_cmd_id(self) -> int:
        cid = self._cmd_id
        self._cmd_id += 1
        return cid

    async def on_open(self):
        await self._subscribe_orderbook()

    async def _subscribe_orderbook(self):
        """Subscribe to orderbook_delta for this market."""
        if self.ws is None:
            return
        
        await asyncio.sleep(0.2)  # Small delay after connection
        
        cmd = {
            "id": self._next_cmd_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_ticker": self.market_ticker,
            },
        }
        sys.stdout.write(f"\n[WS] Subscribing to orderbook_delta: {json.dumps(cmd)}\n")
        sys.stdout.flush()
        await self.ws.send(json.dumps(cmd))

    async def _unsubscribe_orderbook(self):
        if self._orderbook_sid is None:
            return
        cmd = {
            "id": self._next_cmd_id(),
            "cmd": "unsubscribe",
            "params": {"sids": [self._orderbook_sid]},
        }
        await self.ws.send(json.dumps(cmd))
        self._orderbook_sid = None

    async def _resync(self):
        """
        Full clean resync:
        - unsubscribe old orderbook sid
        - subscribe again
        """
        sys.stdout.write("\n[RESYNC] Unsubscribing + resubscribing orderbook...\n")
        sys.stdout.flush()
        await self._unsubscribe_orderbook()
        await self._subscribe_orderbook()
    
    async def on_fill(self, fill_msg: Dict[str, Any]):
        """Override base class on_fill to handle fills with position tracker."""
        if self.position_tracker is None:
            return
        
        market_ticker = fill_msg.get("market_ticker")
        # Only process fills for our market
        if market_ticker != self.market_ticker:
            return
        
        action = fill_msg.get("action")  # "buy" or "sell"
        side = fill_msg.get("side")  # "yes" or "no"
        order_id = fill_msg.get("order_id")
        trade_id = fill_msg.get("trade_id")  # unique per fill
        ts = fill_msg.get("ts")  # unix seconds for when fill happened
        count = fill_msg.get("count", 0)  # quantity filled
        
        if not action or not side:
            sys.stdout.write(f"\n[FILL ERROR] Missing action or side: action={action}, side={side}\n")
            sys.stdout.flush()
            return
        
        if action == "buy":
            # Buy fill:
            # Each fill event becomes its own independent liquidation lot.
            lot_id = await self.position_tracker.register_filled_lot(
                side=side,
                quantity=count,
                order_id=order_id,
                trade_id=trade_id,
                ts=ts,
            )
            sys.stdout.write(f"\n[FILL] {side.upper()} BUY - {count} shares filled - Order ID: {order_id} - Trade ID: {trade_id} - Lot: {lot_id}\n")
            sys.stdout.flush()
        # DISABLED: sell logic commented out
        # elif action == "sell":
        #     # Sell fill - route to the correct lot by sell order id
        #     lot_id = await self.position_tracker.find_lot_id_by_sell_order(order_id)
        #     if lot_id is not None:
        #         # Verify this fill is from the current sell order, not a stale/cancelled one
        #         lot = await self.position_tracker.get_lot(lot_id)
        #         if lot is None or lot.sell_order_id != order_id:
        #             sys.stdout.write(f"\n[FILL IGNORED] Stale sell fill from cancelled order - Order ID: {order_id} - Lot: {lot_id} - Current order: {lot.sell_order_id if lot else 'none'}\n")
        #             sys.stdout.flush()
        #             return
        #
        #         remaining = await self.position_tracker.reduce_lot_on_sell(lot_id, count)
        #         if remaining <= 0:
        #             sys.stdout.write(f"\n[FILL] {side.upper()} SELL - {count} shares filled - Order ID: {order_id} - Lot liquidated: {lot_id}\n")
        #             sys.stdout.flush()
        #             await self.position_tracker.remove_lot(lot_id)
        #         else:
        #             # Trigger fast re-check by resetting last_bid_check_time
        #             await self.position_tracker.update_bid_check_time(lot_id, 0.0)
        #             sys.stdout.write(f"\n[FILL] {side.upper()} SELL PARTIAL - {count} filled - Remaining: {remaining} - Order ID: {order_id} - Lot: {lot_id}\n")
        #             sys.stdout.flush()
        #     else:
        #         sys.stdout.write(f"\n[FILL] {side.upper()} SELL - {count} shares filled - Order ID: {order_id} - No lot found\n")
        #         sys.stdout.flush()

    async def process_message(self, message: str):
        """Override to handle orderbook messages, then call super for fills."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            sys.stdout.write(f"\n[WS JSON ERROR] Failed to parse message: {message[:200]}, error: {e}\n")
            sys.stdout.flush()
            return

        msg_type = data.get("type")
        
        # Handle orderbook messages
        if msg_type == "orderbook_snapshot":
            sid = data.get("sid")
            seq = data.get("seq")
            msg = data.get("msg", {})
            if msg.get("market_ticker") == self.market_ticker:
                await self.book.apply_snapshot(
                    yes=msg.get("yes", []),
                    no=msg.get("no", []),
                    sid=sid,
                    seq=seq,
                )
            return
        
        if msg_type == "orderbook_delta":
            sid = data.get("sid")
            seq = data.get("seq")
            msg = data.get("msg", {})
            if msg.get("market_ticker") == self.market_ticker:
                ok = await self.book.apply_delta(
                    side=msg["side"],
                    price=int(msg["price"]),
                    delta=int(msg["delta"]),
                    sid=sid,
                    seq=seq,
                )
                if not ok:
                    await self._resync()
            return
        
        # Handle subscription confirmations
        if msg_type == "subscribed":
            msg = data.get("msg", {})
            channel = msg.get("channel")
            if channel == "orderbook_delta":
                self._orderbook_sid = msg.get("sid")
                sys.stdout.write(f"\n[WS] Subscribed to orderbook_delta, sid={self._orderbook_sid}\n")
                sys.stdout.flush()
            return
        
        if msg_type == "ok":
            subs = data.get("subscriptions", [])
            for s in subs:
                channel = s.get("channel")
                sid = s.get("sid")
                if channel == "orderbook_delta":
                    self._orderbook_sid = sid
                    sys.stdout.write(f"\n[WS] Subscribed to orderbook_delta, sid={self._orderbook_sid}\n")
                    sys.stdout.flush()
            return
        
        # For all other messages (including fills), call parent process_message
        await super().process_message(message)

        # Subscription confirmation includes sids per channel
        if msg_type == "ok":
            subs = data.get("subscriptions", [])
            
            # #region agent log
            with open('/Users/tommasosini/Desktop/latency-trading/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"G","location":"streaming.py:295","message":"Received ok message with subscriptions","data":{"subscriptions":subs,"full_data":data},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
            
            for s in subs:
                channel = s.get("channel")
                sid = s.get("sid")
                if channel == "orderbook_delta":
                    self._orderbook_sid = sid
                    sys.stdout.write(f"\n[WS] Subscribed to orderbook_delta, sid={self._orderbook_sid}\n")
                    sys.stdout.flush()
                elif channel == "user_fills":
                    self._user_fills_sid = sid
                    sys.stdout.write(f"\n[WS] ✓ Successfully subscribed to user_fills, sid={self._user_fills_sid}\n")
                    sys.stdout.flush()
                    
                    # #region agent log
                    with open('/Users/tommasosini/Desktop/latency-trading/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"G","location":"streaming.py:305","message":"user_fills subscription confirmed","data":{"sid":sid},"timestamp":int(time.time()*1000)}) + '\n')
                    # #endregion
            return

        # Hypothesis C: Maybe "subscribed" message type contains subscription info
        if msg_type == "subscribed":
            # #region agent log
            with open('/Users/tommasosini/Desktop/latency-trading/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"streaming.py:315","message":"Received subscribed message","data":{"full_data":data},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
            # Check if this contains subscription info
            msg = data.get("msg", {})
            channel = msg.get("channel")
            if channel:
                sys.stdout.write(f"\n[WS] Subscribed message for channel: {channel}, sid: {msg.get('sid')}\n")
                sys.stdout.flush()
                if channel == "user_fills":
                    self._user_fills_sid = msg.get("sid")
                    sys.stdout.write(f"\n[WS] ✓ Successfully subscribed to user_fills via 'subscribed' message, sid={self._user_fills_sid}\n")
                    sys.stdout.flush()
                    
                    # #region agent log
                    with open('/Users/tommasosini/Desktop/latency-trading/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"streaming.py:325","message":"user_fills subscription confirmed via subscribed message","data":{"sid":self._user_fills_sid},"timestamp":int(time.time()*1000)}) + '\n')
                    # #endregion
            return

        if msg_type == "error":
            error_msg = data.get("msg", {})
            error_code = error_msg.get("code")
            error_text = error_msg.get("msg")
            request_id = data.get("id")
            sys.stdout.write(f"\n[WS ERROR] Request ID: {request_id}, Code: {error_code}, Message: {error_text}, Full: {json.dumps(data)}\n")
            sys.stdout.flush()
            
            # #region agent log
            with open('/Users/tommasosini/Desktop/latency-trading/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"G","location":"streaming.py:335","message":"WebSocket error received","data":{"request_id":request_id,"error_code":error_code,"error_text":error_text,"full_data":data},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
            return

        if msg_type == "orderbook_snapshot":
            sid = data.get("sid")
            seq = data.get("seq")
            msg = data.get("msg", {})

            if msg.get("market_ticker") != self.market_ticker:
                return

            await self.book.apply_snapshot(
                yes=msg.get("yes", []),
                no=msg.get("no", []),
                sid=sid,
                seq=seq,
            )
            return

        if msg_type == "orderbook_delta":
            sid = data.get("sid")
            seq = data.get("seq")
            msg = data.get("msg", {})

            if msg.get("market_ticker") != self.market_ticker:
                return

            ok = await self.book.apply_delta(
                side=msg["side"],
                price=int(msg["price"]),
                delta=int(msg["delta"]),
                sid=sid,
                seq=seq,
            )

            if not ok:
                # IMPORTANT: gap -> clean resync
                await self._resync()
            return



# -----------------------------
# 1ms NEWLINE Tape Printer
# -----------------------------
def fmt_px(x: Optional[int]) -> str:
    return "---" if x is None else f"{x:02d}"


async def print_tape_every_1ms(book: OrderBook, http_client: KalshiHttpClient):
    """
    Print two updating lines every 1ms:
    - Line 1: Price data with YES/NO bid+ask
    - Line 2: Portfolio value
    Uses ANSI escape codes to update both lines in place.
    """
    interval_ns = 1_000_000  # 1ms
    next_tick = time.perf_counter_ns()
    last_price_line_length = 0
    last_portfolio_line_length = 0
    
    # Portfolio value cache - update every 500ms to avoid rate limiting
    portfolio_update_interval_ns = 500_000_000  # 500ms
    next_portfolio_update = time.perf_counter_ns()
    portfolio_value = "Loading..."
    portfolio_line = "Portfolio: Loading..."

    while True:
        now = time.perf_counter_ns()
        if now < next_tick:
            await asyncio.sleep((next_tick - now) / 1_000_000_000)
            continue

        ts_ns = time.time_ns()
        sec = ts_ns // 1_000_000_000
        micros = (ts_ns % 1_000_000_000) // 1_000
        tstr = time.strftime("%H:%M:%S", time.localtime(sec)) + f".{micros:06d}"

        async with book.lock:
            yb = book.best_yes_bid()
            ya = book.best_yes_ask_implied()
            nb = book.best_no_bid()
            na = book.best_no_ask_implied()

        price_line = f"{tstr} | YES {fmt_px(yb)}/{fmt_px(ya)} || NO {fmt_px(nb)}/{fmt_px(na)}"
        
        # Update portfolio value periodically (every 500ms)
        if now >= next_portfolio_update:
            try:
                balance_data = http_client.get_balance()
                # Extract balance - API returns value in cents, so divide by 100
                balance = balance_data.get("balance", balance_data.get("portfolio_balance", "N/A"))
                if isinstance(balance, (int, float)):
                    balance_dollars = balance / 100.0
                    portfolio_value = f"${balance_dollars:,.2f}"
                else:
                    portfolio_value = str(balance)
                portfolio_line = f"Portfolio: {portfolio_value}"
            except Exception as e:
                portfolio_line = f"Portfolio: Error ({str(e)[:30]})"
            next_portfolio_update = now + portfolio_update_interval_ns
        
        # Pad lines to clear any leftover characters
        if len(price_line) < last_price_line_length:
            price_line = price_line.ljust(last_price_line_length)
        last_price_line_length = len(price_line)
        
        if len(portfolio_line) < last_portfolio_line_length:
            portfolio_line = portfolio_line.ljust(last_portfolio_line_length)
        last_portfolio_line_length = len(portfolio_line)
        
        # Use ANSI escape codes to update both lines:
        # \033[2K = clear entire line
        # \033[A = move cursor up one line
        # \r = return to start of line
        # Print both lines, then move up 1 to return to price line for next iteration
        output = f"\033[2K\r{price_line}\n\033[2K\r{portfolio_line}\033[A"
        sys.stdout.write(output)
        sys.stdout.flush()

        next_tick += interval_ns

        # drift correction
        if now - next_tick > 10 * interval_ns:
            next_tick = now + interval_ns


# -----------------------------
# Startup validation (REST snapshot)
# -----------------------------
def seed_book_with_rest_snapshot(market_ticker: str) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Pull REST orderbook once to confirm the market has activity
    and to seed state before WS updates arrive.
    REST orderbook is public.
    """
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}/orderbook"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()
    ob = data.get("orderbook", {}) or {}
    yes = ob.get("yes") or []
    no = ob.get("no") or []
    # Ensure we return lists, not None
    return (yes if isinstance(yes, list) else []), (no if isinstance(no, list) else [])


# -----------------------------
# Position Sizing Configuration
# -----------------------------
POSITION_PCT = 0.25  # % of portfolio to trade


def compute_shares_from_budget(target_usd: float, price_cents: int) -> Optional[int]:
    """
    Compute integer share count from USD budget and price in cents.
    
    Args:
        target_usd: Target USD budget for the trade
        price_cents: Limit price in cents (1-99)
    
    Returns:
        Integer share count, or None if invalid/too small
    """
    if price_cents <= 0:
        return None
    
    # Convert cents to dollars
    price_dollars = price_cents / 100.0
    
    if price_dollars <= 0:
        return None
    
    # Compute shares using floor division
    shares = int(target_usd // price_dollars)
    
    # Safety: must be at least 1 share
    if shares < 1:
        return None
    
    return shares


def get_portfolio_usd(balance_data: Dict[str, Any]) -> float:
    """
    Extract portfolio USD value from balance response.
    Balance is returned in cents, so divide by 100.
    """
    balance = balance_data.get("balance") or balance_data.get("portfolio_balance")
    if balance is None:
        return 0.0
    
    if isinstance(balance, (int, float)):
        return balance / 100.0
    elif isinstance(balance, str):
        try:
            return float(balance)
        except ValueError:
            return 0.0
    return 0.0


# -----------------------------
# Position Tracker
# -----------------------------
@dataclass
class PendingBuyOrder:
    side: str  # "yes" or "no"
    quantity: int
    order_id: str
    placed_time: float


@dataclass
class PositionLot:
    lot_id: str
    side: str  # "yes" or "no"
    quantity: int
    entry_time: float
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    liquidation_start_time: Optional[float] = None
    current_sell_price: Optional[int] = None
    last_bid_check_time: float = 0.0
    escalated: bool = False


class PositionTracker:
    def __init__(self):
        # Per-fill "lots" so each buy fill chunk has its own liquidation lifecycle
        self.lots: Dict[str, PositionLot] = {}  # key: lot_id
        # Route sell fills (order_id) back to the originating lot
        self.sell_order_to_lot: Dict[str, str] = {}  # key: sell_order_id -> lot_id
        self.pending_orders: Dict[str, PendingBuyOrder] = {}  # key: order_id, value: PendingBuyOrder
        self.lock = asyncio.Lock()
        self._lot_seq: int = 0

    async def register_pending_buy_order(self, side: str, quantity: int, order_id: str):
        """Register a pending buy order - will check if filled and register position when filled."""
        async with self.lock:
            self.pending_orders[order_id] = PendingBuyOrder(
                side=side,
                quantity=quantity,
                order_id=order_id,
                placed_time=time.time()
            )

    async def register_filled_lot(
        self,
        side: str,
        quantity: int,
        order_id: str,
        trade_id: Optional[str],
        ts: Optional[int],
    ) -> str:
        """Register a new per-fill lot when a buy fill is received.
        
        If a trade_id is provided by the exchange, it is used as the lot_id so
        each fill chunk is keyed exactly by its unique trade identifier.
        """
        async with self.lock:
            # Prefer exchange timestamp if present (seconds), fallback to local time
            entry_time = float(ts) if ts is not None else time.time()
            # Prefer exchange-provided trade_id as the lot identifier
            if trade_id:
                lot_id = trade_id
            else:
                self._lot_seq += 1
                lot_id = f"{side}:{order_id}:{int(entry_time * 1000)}:{self._lot_seq}"
            self.lots[lot_id] = PositionLot(
                lot_id=lot_id,
                side=side,
                quantity=quantity,
                entry_time=entry_time,
                buy_order_id=order_id,
                last_bid_check_time=0.0,
                escalated=False
            )
            # Remove from pending orders
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            return lot_id

    async def get_lot(self, lot_id: str) -> Optional[PositionLot]:
        async with self.lock:
            return self.lots.get(lot_id)

    async def list_open_lots(self, side: Optional[str] = None) -> List[PositionLot]:
        async with self.lock:
            if side is None:
                return list(self.lots.values())
            return [lot for lot in self.lots.values() if lot.side == side]

    async def has_open_lots(self, side: str) -> bool:
        async with self.lock:
            return any(lot.side == side for lot in self.lots.values())

    async def get_pending_orders(self) -> List[PendingBuyOrder]:
        """Get all pending buy orders."""
        async with self.lock:
            return list(self.pending_orders.values())

    async def remove_pending_order(self, order_id: str):
        """Remove a pending order."""
        async with self.lock:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

    async def remove_lot(self, lot_id: str):
        async with self.lock:
            lot = self.lots.pop(lot_id, None)
            if lot and lot.sell_order_id:
                # Keep mapping cleanup best-effort; late fills will then log "unmapped"
                self.sell_order_to_lot.pop(lot.sell_order_id, None)

    async def find_lot_id_by_sell_order(self, sell_order_id: Optional[str]) -> Optional[str]:
        if not sell_order_id:
            return None
        async with self.lock:
            return self.sell_order_to_lot.get(sell_order_id)

    async def reduce_lot_on_sell(self, lot_id: str, filled_qty: int) -> int:
        """Decrement remaining lot size after a sell fill. Returns remaining quantity."""
        async with self.lock:
            lot = self.lots.get(lot_id)
            if lot is None:
                return 0
            lot.quantity = max(0, lot.quantity - filled_qty)
            return lot.quantity

    async def update_sell_order(self, lot_id: str, sell_order_id: str, sell_price: int):
        async with self.lock:
            lot = self.lots.get(lot_id)
            if lot is None:
                return
            # Remove old mapping to prevent stale/cancelled order fills from being processed
            if lot.sell_order_id and lot.sell_order_id in self.sell_order_to_lot:
                del self.sell_order_to_lot[lot.sell_order_id]
            lot.sell_order_id = sell_order_id
            lot.current_sell_price = sell_price
            lot.last_bid_check_time = time.time()
            if sell_order_id:
                self.sell_order_to_lot[sell_order_id] = lot_id
    
    async def start_liquidation(self, lot_id: str, start_time: float):
        """Start liquidation for a lot."""
        async with self.lock:
            lot = self.lots.get(lot_id)
            if lot is None:
                return
            lot.liquidation_start_time = start_time
            lot.last_bid_check_time = start_time
    
    async def update_bid_check_time(self, lot_id: str, check_time: float):
        """Update the last bid check time."""
        async with self.lock:
            lot = self.lots.get(lot_id)
            if lot is None:
                return
            lot.last_bid_check_time = check_time
    
    async def mark_escalated(self, lot_id: str):
        """Mark lot as escalated."""
        async with self.lock:
            lot = self.lots.get(lot_id)
            if lot is None:
                return
            lot.escalated = True


async def keyboard_input_handler(
    book: OrderBook, 
    http_client: KalshiHttpClient, 
    market_ticker: str,
    position_tracker: PositionTracker,
    position_pct: float = POSITION_PCT
):
    """
    Handle keyboard input asynchronously:
    - 'q': Buy YES at ask price (sized as percentage of portfolio)
    - 'w': Buy NO at ask price (sized as percentage of portfolio)
    """
    # Set stdin to non-blocking mode
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    
    try:
        loop = asyncio.get_event_loop()
        while True:
            # Check if input is available (non-blocking)
            if select.select([sys.stdin], [], [], 0.01)[0]:
                char = sys.stdin.read(1)
                
                if char == 'q':
                    # DISABLED: position check commented out
                    # # Check if YES position already exists - prevent new order if not liquidated
                    # if await position_tracker.has_open_lots("yes"):
                    #     sys.stdout.write(f"\n[ORDER ERROR] YES position already exists - cannot place new buy until liquidated\n")
                    #     sys.stdout.flush()
                    #     continue

                    # Fetch portfolio balance
                    try:
                        balance_data = http_client.get_balance()
                        portfolio_usd = get_portfolio_usd(balance_data)
                    except Exception as e:
                        sys.stdout.write(f"\n[ORDER ERROR] Failed to fetch balance: {e}\n")
                        sys.stdout.flush()
                        continue
                    
                    # Compute target USD
                    target_usd = portfolio_usd * position_pct
                    
                    # Get limit price (ask for BUY)
                    async with book.lock:
                        ya = book.best_yes_ask_implied()
                    
                    if ya is None:
                        sys.stdout.write(f"\n[ORDER ERROR] YES ask price not available\n")
                        sys.stdout.flush()
                        continue
                    
                    # Compute shares from budget
                    shares = compute_shares_from_budget(target_usd, ya)
                    if shares is None:
                        sys.stdout.write(f"\n[ORDER ERROR] Budget too small: portfolio=${portfolio_usd:.2f}, target=${target_usd:.2f} ({position_pct*100:.0f}%), price={ya} cents\n")
                        sys.stdout.flush()
                        continue
                    
                    # Compute estimated cost
                    price_dollars = ya / 100.0
                    est_cost = shares * price_dollars
                    
                    # Log sizing details
                    sys.stdout.write(f"\n[SIZING] portfolio=${portfolio_usd:.2f}, target=${target_usd:.2f} ({position_pct*100:.0f}%), price=${price_dollars:.2f}, shares={shares}, est_cost=${est_cost:.2f}\n")
                    sys.stdout.flush()
                    
                    # Place order
                    try:
                        result = http_client.create_order(
                            ticker=market_ticker,
                            action="buy",
                            side="yes",
                            count=shares,
                            yes_price=ya,
                            order_type="limit"
                        )
                        order_id = result.get("order", {}).get("order_id", "unknown")
                        await position_tracker.register_pending_buy_order("yes", shares, order_id)
                        sys.stdout.write(f"\n[ORDER] YES BUY @ {ya} - {shares} shares - Order ID: {order_id}\n")
                        sys.stdout.flush()
                    except Exception as e:
                        sys.stdout.write(f"\n[ORDER ERROR] YES BUY failed: {e}\n")
                        sys.stdout.flush()
                
                elif char == 'w':
                    # DISABLED: position check commented out
                    # # Check if NO position already exists - prevent new order if not liquidated
                    # if await position_tracker.has_open_lots("no"):
                    #     sys.stdout.write(f"\n[ORDER ERROR] NO position already exists - cannot place new buy until liquidated\n")
                    #     sys.stdout.flush()
                    #     continue

                    # Fetch portfolio balance
                    try:
                        balance_data = http_client.get_balance()
                        portfolio_usd = get_portfolio_usd(balance_data)
                    except Exception as e:
                        sys.stdout.write(f"\n[ORDER ERROR] Failed to fetch balance: {e}\n")
                        sys.stdout.flush()
                        continue
                    
                    # Compute target USD
                    target_usd = portfolio_usd * position_pct
                    
                    # Get limit price (ask for BUY)
                    async with book.lock:
                        na = book.best_no_ask_implied()
                    
                    if na is None:
                        sys.stdout.write(f"\n[ORDER ERROR] NO ask price not available\n")
                        sys.stdout.flush()
                        continue
                    
                    # Compute shares from budget
                    shares = compute_shares_from_budget(target_usd, na)
                    if shares is None:
                        sys.stdout.write(f"\n[ORDER ERROR] Budget too small: portfolio=${portfolio_usd:.2f}, target=${target_usd:.2f} ({position_pct*100:.0f}%), price={na} cents\n")
                        sys.stdout.flush()
                        continue
                    
                    # Compute estimated cost
                    price_dollars = na / 100.0
                    est_cost = shares * price_dollars
                    
                    # Log sizing details
                    sys.stdout.write(f"\n[SIZING] portfolio=${portfolio_usd:.2f}, target=${target_usd:.2f} ({position_pct*100:.0f}%), price=${price_dollars:.2f}, shares={shares}, est_cost=${est_cost:.2f}\n")
                    sys.stdout.flush()
                    
                    # Place order
                    try:
                        result = http_client.create_order(
                            ticker=market_ticker,
                            action="buy",
                            side="no",
                            count=shares,
                            no_price=na,
                            order_type="limit"
                        )
                        order_id = result.get("order", {}).get("order_id", "unknown")
                        await position_tracker.register_pending_buy_order("no", shares, order_id)
                        sys.stdout.write(f"\n[ORDER] NO BUY @ {na} - {shares} shares - Order ID: {order_id}\n")
                        sys.stdout.flush()
                    except Exception as e:
                        sys.stdout.write(f"\n[ORDER ERROR] NO BUY failed: {e}\n")
                        sys.stdout.flush()
                
                elif char == '\x03':  # Ctrl+C
                    break
            
            await asyncio.sleep(0.01)
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)






async def liquidation_manager(
    book: OrderBook,
    http_client: KalshiHttpClient,
    market_ticker: str,
    position_tracker: PositionTracker
):
    """
    Manages liquidation of positions following exact specification:
    - Starts 6 seconds after entry
    - Places sell order at best bid (bid₀)
    - Every 1s: checks if position flat, then checks if bid changed and cancels/replaces
    - After 10s: escalates with bid - 0.02
    """
    while True:
        await asyncio.sleep(0.1)  # Check every 100ms
        
        current_time = time.time()
        
        # Check each open lot (each buy fill chunk has its own lifecycle)
        lots = await position_tracker.list_open_lots()
        for lot in lots:
            lot_id = lot.lot_id
            # Re-fetch to use most recent quantity / state
            lot = await position_tracker.get_lot(lot_id)
            if lot is None:
                continue
            
            side = lot.side
            time_since_entry = current_time - lot.entry_time
            
            # Start liquidation 6 seconds after entry (entry_time is set when buy fill is received)
            if time_since_entry >= 6.0 and lot.liquidation_start_time is None:
                sys.stdout.write(f"\n[LIQUIDATION] Starting liquidation for {side.upper()} lot {lot_id} after {time_since_entry:.2f}s\n")
                sys.stdout.flush()
                await position_tracker.start_liquidation(lot_id, current_time)
                # Re-fetch lot to get updated values
                lot = await position_tracker.get_lot(lot_id)
                if lot is None:
                    continue
                
                # Get best bid (bid₀) - the lower of the two numbers in the spread
                async with book.lock:
                    if side == "yes":
                        bid = book.best_yes_bid_direct()
                    else:
                        bid = book.best_no_bid_direct()
                
                # If bid missing / market halted → panic stop
                if bid is None:
                    sys.stdout.write(f"\n[LIQUIDATION ERROR] {side.upper()} bid missing - panic stop\n")
                    sys.stdout.flush()
                    if lot.sell_order_id:
                        try:
                            http_client.cancel_order(lot.sell_order_id)
                        except:
                            pass
                    await position_tracker.remove_lot(lot_id)
                    continue
                
                # Submit LIMIT SELL: price = bid₀, quantity = filled_qty
                try:
                    result = http_client.create_order(
                        ticker=market_ticker,
                        action="sell",
                        side=side,
                        count=lot.quantity,
                        yes_price=bid if side == "yes" else None,
                        no_price=bid if side == "no" else None,
                        order_type="limit"
                    )
                    sell_order_id = result.get("order", {}).get("order_id", "unknown")
                    await position_tracker.update_sell_order(lot_id, sell_order_id, bid)
                    sys.stdout.write(f"\n[LIQUIDATION] {side.upper()} SELL @ {bid} - {lot.quantity} shares - Lot {lot_id} - Order ID: {sell_order_id}\n")
                    sys.stdout.flush()
                except Exception as e:
                    sys.stdout.write(f"\n[LIQUIDATION ERROR] {side.upper()} SELL failed: {e}\n")
                    sys.stdout.flush()
                    continue
            
            # Bid Adjustment Walkdown Loop - Every 1s
            if lot.liquidation_start_time is not None:
                liquidation_age = current_time - lot.liquidation_start_time
                
                # Check every 1 second
                if current_time - lot.last_bid_check_time >= 1.0:
                    # Re-fetch lot to get updated values
                    lot = await position_tracker.get_lot(lot_id)
                    if lot is None:
                        # Lot was cleared (likely by sell fill)
                        continue
                    
                    if lot.quantity <= 0:
                        await position_tracker.remove_lot(lot_id)
                        continue
                    
                    # Mark check time up front to avoid rapid retry loops on errors
                    await position_tracker.update_bid_check_time(lot_id, current_time)
                    
                    # Read bid_new
                    async with book.lock:
                        if side == "yes":
                            bid_new = book.best_yes_bid_direct()
                        else:
                            bid_new = book.best_no_bid_direct()
                    
                    # If bid missing / halted → panic stop
                    if bid_new is None:
                        sys.stdout.write(f"\n[LIQUIDATION ERROR] {side.upper()} bid missing - panic stop\n")
                        sys.stdout.flush()
                        if lot.sell_order_id:
                            try:
                                http_client.cancel_order(lot.sell_order_id)
                            except:
                                pass
                        await position_tracker.remove_lot(lot_id)
                        continue
                    
                    # Determine target price (respect escalation offset when set)
                    target_price = bid_new
                    if lot.escalated:
                        target_price = max(1, bid_new - 2)
                    
                    # Always cancel and replace at the current best bid (or aggressive bid) with remaining size
                    if lot.sell_order_id:
                        try:
                            http_client.cancel_order(lot.sell_order_id)
                        except:
                            pass
                    
                    try:
                        result = http_client.create_order(
                            ticker=market_ticker,
                            action="sell",
                            side=side,
                            count=lot.quantity,
                            yes_price=target_price if side == "yes" else None,
                            no_price=target_price if side == "no" else None,
                            order_type="limit"
                        )
                        sell_order_id = result.get("order", {}).get("order_id", "unknown")
                        await position_tracker.update_sell_order(lot_id, sell_order_id, target_price)
                        sys.stdout.write(f"\n[LIQUIDATION] {side.upper()} REPRICE @ {target_price} - Remaining {lot.quantity} - Lot {lot_id} - Order ID: {sell_order_id}\n")
                        sys.stdout.flush()
                    except Exception as e:
                        sys.stdout.write(f"\n[LIQUIDATION ERROR] {side.upper()} REPRICE failed: {e}\n")
                        sys.stdout.flush()
                
                # Exit Timeout Escalation - If still not fully liquidated 10.0s after liquidation start
                if liquidation_age >= 10.0 and not lot.escalated and lot.sell_order_id:
                    await position_tracker.mark_escalated(lot_id)
                    # Re-fetch lot to get updated values
                    lot = await position_tracker.get_lot(lot_id)
                    if lot is None:
                        continue
                    
                    # Cancel resting sell
                    try:
                        http_client.cancel_order(lot.sell_order_id)
                    except:
                        pass
                    
                    # Get current bid
                    async with book.lock:
                        if side == "yes":
                            bid_aggressive = book.best_yes_bid_direct()
                        else:
                            bid_aggressive = book.best_no_bid_direct()
                    
                    if bid_aggressive is None:
                        sys.stdout.write(f"\n[LIQUIDATION ERROR] {side.upper()} bid missing during escalation - panic stop\n")
                        sys.stdout.flush()
                        await position_tracker.remove_lot(lot_id)
                        continue
                    
                    # Submit more aggressive LIMIT SELL at: price = best_bid − 0.02
                    aggressive_price = max(1, bid_aggressive - 2)  # Subtract 2 cents (0.02), min 1
                    
                    try:
                        result = http_client.create_order(
                            ticker=market_ticker,
                            action="sell",
                            side=side,
                            count=lot.quantity,
                            yes_price=aggressive_price if side == "yes" else None,
                            no_price=aggressive_price if side == "no" else None,
                            order_type="limit"
                        )
                        sell_order_id = result.get("order", {}).get("order_id", "unknown")
                        await position_tracker.update_sell_order(lot_id, sell_order_id, aggressive_price)
                        sys.stdout.write(f"\n[LIQUIDATION ESCALATE] {side.upper()} SELL @ {aggressive_price} (bid-2) - Lot {lot_id} - Order ID: {sell_order_id}\n")
                        sys.stdout.flush()
                    except Exception as e:
                        sys.stdout.write(f"\n[LIQUIDATION ERROR] {side.upper()} ESCALATE failed: {e}\n")
                        sys.stdout.flush()


async def run_stream(key_id: str, private_key, market_ticker: str):
    http_client = KalshiHttpClient(key_id=key_id, private_key=private_key)
    position_tracker = PositionTracker()
    streamer = MarketDataStreamer(
        key_id=key_id,
        private_key=private_key,
        market_ticker=market_ticker,
        position_tracker=position_tracker,
        http_client=http_client,
    )

    # Seed from REST once (fast + confirms real liquidity)
    try:
        yes, no = seed_book_with_rest_snapshot(market_ticker)
        await streamer.book.apply_snapshot(yes=yes, no=no, sid=0, seq=0)
        sys.stdout.write(f"[REST SEED] yes_levels={len(yes)} no_levels={len(no)}\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"[REST SEED WARN] couldn't seed orderbook: {e}\n")
        sys.stdout.flush()

    await asyncio.gather(
        streamer.connect(),
        print_tape_every_1ms(streamer.book, http_client),
        keyboard_input_handler(streamer.book, http_client, market_ticker, position_tracker),
        # liquidation_manager(streamer.book, http_client, market_ticker, position_tracker),  # DISABLED: sell logic commented out
    )


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    from cryptography.hazmat.primitives import serialization
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()

    key_id = os.getenv("KALSHI_KEY_ID")
    if not key_id:
        raise ValueError("Missing KALSHI_KEY_ID in .env")

    private_key_path = os.getenv("PRIVATE_KEY_PATH") or str(Path.home() / "Downloads" / "bot.txt")
    private_key_file = Path(private_key_path)
    if not private_key_file.exists():
        raise FileNotFoundError(f"Private key file not found: {private_key_file}")

    with open(private_key_file, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    market_ticker = os.getenv("KALSHI_MARKET_TICKER") or "KXBTC15M-26JAN170515-15"

    sys.stdout.write(f"[STARTING] Streaming {market_ticker}...\n")
    sys.stdout.flush()

    # Run unbuffered if you want the fastest display:
    # python -u streaming.py
    asyncio.run(run_stream(key_id, private_key, market_ticker))
