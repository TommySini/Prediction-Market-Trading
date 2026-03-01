#!/usr/bin/env python3
"""
rl_live_trader.py  —  Live Polymarket latency trader driven by the trained RegNet model.

Architecture
============
HOT PATH  (async event loop – inference only, zero I/O blocking):
  1. Stream Binance book-ticker + Polymarket order-book via data_stream.
  2. On every Binance update:  compute fair YES probability, check for dislocation.
  3. Build the 12-dim state vector (same features as RL_1_hour.py).
  4. Run RegNet forward pass → scalar prediction.
  5. If pred >= threshold → submit BUY limit at current Polymarket ask.
  6. Track inventory; reset at market-hour boundary.

BACKGROUND WORKERS  (separate threads/processes — never block hot path):
  • DataLogger thread:  buffer ticks + orders → flush to SQLite every 500 ms.
  • Retrainer process:  once per hour, reads historical DB + new data, fine-tunes
    the RegNet, writes new state_dict to a temp file, and signals a hot-swap.
  • WeightLoader:  main loop polls for new weight file at hour boundary → atomic
    load into the live model.

Run (dry):
  python rl_live_trader.py --dry-run

Run (live):
  POLYMARKET_PRIVATE_KEY=... python rl_live_trader.py --live

Run with custom model:
  python rl_live_trader.py --live --model-path my_model.pt --min-pred -0.3
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import signal
import sqlite3
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np

# ---- torch ----
import torch
import torch.nn as nn
import torch.optim as optim

# ---- project imports ----
try:
    import data_stream as ds
except ModuleNotFoundError:
    from freshlatency import data_stream as ds

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.order_builder.builder import ROUNDING_CONFIG

# ====================================================================
# Logging
# ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rl_live")

# ====================================================================
# Constants
# ====================================================================
EPS = 1e-9
ET = ZoneInfo("America/New_York")
SEC_PER_YEAR = 365.0 * 24.0 * 3600.0

_SLUG_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_SLUG_RE = re.compile(
    r"bitcoin-up-or-down-(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)-(\d{1,2})-(\d{1,2})(am|pm)-et$"
)

# State vector dimension (must match RL_1_hour.py)
STATE_DIM = 12

# ====================================================================
# Math helpers  (identical to 1_hour_latency_trader / RL_1_hour)
# ====================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def softsign(x: float) -> float:
    return x / (1.0 + abs(x))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    p = clamp(p, 1e-9, 1 - 1e-9)
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)


def norm_size(sz: float, decimals: int) -> float:
    scale = 10 ** int(max(0, decimals))
    return math.floor(max(0.0, float(sz)) * scale + EPS) / scale


def norm_buy_size_for_amount_constraints(
    min_size: float, px: float, amount_step_decimals: int = 4,
) -> float:
    scale = 10 ** int(max(0, amount_step_decimals))
    size_units = int(math.ceil(max(0.0, float(min_size)) * scale - EPS))
    price_cents = max(1, int(round(max(0.0, float(px)) * 100.0)))
    step_units = scale // math.gcd(price_cents, scale)
    size_units = ((size_units + step_units - 1) // step_units) * step_units
    return float(size_units) / float(scale)


# ====================================================================
# Black-Scholes fair-value model  (from 1_hour_latency_trader)
# ====================================================================

def bs_digital_prob(S: float, K: float, sigma: float, T_years: float) -> float:
    if not (S > 0 and K > 0 and sigma > 1e-12 and T_years > 1e-12):
        return 0.5
    d2 = (math.log(S / K) - 0.5 * sigma * sigma * T_years) / (sigma * math.sqrt(T_years))
    return clamp(norm_cdf(d2), 1e-6, 1 - 1e-6)


def implied_sigma_from_prob(p: float, S: float, K: float, T_years: float) -> Optional[float]:
    if not (S > 0 and K > 0 and T_years > 1e-12):
        return None
    p = clamp(p, 1e-6, 1 - 1e-6)
    d2 = norm_ppf(p)
    T = T_years
    a_coeff = 0.5 * T
    b_coeff = d2 * math.sqrt(T)
    c_coeff = -math.log(S / K)
    disc = b_coeff * b_coeff - 4 * a_coeff * c_coeff
    if disc < 0:
        disc = 0.0
    sqrt_disc = math.sqrt(disc)
    denom = 2 * a_coeff
    r1 = (-b_coeff + sqrt_disc) / denom
    r2 = (-b_coeff - sqrt_disc) / denom
    sigs = [r for r in (r1, r2) if r > 1e-9 and math.isfinite(r)]
    return min(sigs) if sigs else None


def bs_digital_prob_slope_numeric(
    S: float, K: float, sigma: float, T_years: float, eps_frac: float,
) -> Optional[float]:
    if not (S > 0 and K > 0 and sigma > 1e-12 and T_years > 1e-12):
        return None
    eps_frac = max(1e-8, float(eps_frac))
    S_up = float(S) * (1.0 + eps_frac)
    if not math.isfinite(S_up) or S_up <= 0.0:
        return None
    p0 = bs_digital_prob(float(S), float(K), float(sigma), float(T_years))
    p_up = bs_digital_prob(float(S_up), float(K), float(sigma), float(T_years))
    denom = float(S) * eps_frac
    if denom <= 0.0:
        return None
    return (p_up - p0) / denom


def time_left_years(market_end_ts: float) -> float:
    return max(0.0, (float(market_end_ts) - time.time()) / SEC_PER_YEAR)


def parse_slug_hour_window(slug: str) -> Optional[tuple[datetime, datetime]]:
    m = _SLUG_RE.match(slug.strip().lower())
    if not m:
        return None
    mon_name, day_s, h_s, ampm = m.group(1), m.group(2), m.group(3), m.group(4)
    month = _SLUG_MONTHS[mon_name]
    day = int(day_s)
    h12 = int(h_s)
    if h12 == 12:
        hour = 0 if ampm == "am" else 12
    else:
        hour = h12 if ampm == "am" else (h12 + 12)
    now_et = datetime.now(ET)
    best: Optional[tuple[float, datetime, datetime]] = None
    for y in (now_et.year - 1, now_et.year, now_et.year + 1):
        try:
            start = datetime(y, month, day, hour, 0, 0, tzinfo=ET)
            end = start + timedelta(hours=1)
            dist = abs((start - now_et).total_seconds())
            if best is None or dist < best[0]:
                best = (dist, start, end)
        except ValueError:
            continue
    return (best[1], best[2]) if best else None


# ====================================================================
# RegNet  (identical architecture to RL_1_hour.py)
# ====================================================================

class RegNet(nn.Module):
    def __init__(self, in_dim: int = STATE_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ====================================================================
# Live calibration state  (mimics StrategyState calibration fields)
# ====================================================================

@dataclass
class CalibrationState:
    """BS-IV calibration state, updated on every Polymarket book update."""
    sigma_raw: Optional[float] = None
    sigma_eff_ema: Optional[float] = None
    implied_z_scale_ema: Optional[float] = None
    slope_ref_S: Optional[float] = None
    slope_ref_yes_mid: Optional[float] = None
    slope_ref_sigma: Optional[float] = None
    slope_ref_wall_ts: Optional[float] = None

    # Market params
    strike_k: float = 0.0
    strike_k_source: str = ""
    market_end_ts: float = 0.0


def calibrate_on_poly_update(
    *,
    cal: CalibrationState,
    S: float,
    y_mid: float,
    args: argparse.Namespace,
) -> None:
    """Update BS-IV calibration on every Polymarket book update."""
    if S is None or not math.isfinite(float(S)) or float(S) <= 0:
        return
    if y_mid is None or not math.isfinite(float(y_mid)):
        return
    K = float(cal.strike_k)
    if K <= 0:
        return
    T = time_left_years(cal.market_end_ts)
    if T * SEC_PER_YEAR < float(args.min_T_sec):
        return

    y_mid_clamped = clamp(float(y_mid), 0.001, 0.999)

    # Implied sigma
    sig = implied_sigma_from_prob(y_mid_clamped, float(S), K, T)
    if sig is None or sig <= 0.0 or not math.isfinite(sig):
        return
    cal.sigma_raw = sig

    # EMA smoothed sigma
    alpha = float(args.sigma_ema_alpha)
    if cal.sigma_eff_ema is None:
        cal.sigma_eff_ema = sig
    else:
        cal.sigma_eff_ema = (1.0 - alpha) * float(cal.sigma_eff_ema) + alpha * sig

    # Update slope reference
    cal.slope_ref_S = float(S)
    cal.slope_ref_yes_mid = float(y_mid_clamped)
    cal.slope_ref_sigma = float(cal.sigma_eff_ema)
    cal.slope_ref_wall_ts = time.time()

    # Update implied z scale EMA
    # z_scale = sigma * sqrt(T)
    z_scale_raw = float(cal.sigma_eff_ema) * math.sqrt(max(1e-12, T))
    z_alpha = float(args.implied_z_ema_alpha)
    clamp_ratio = float(args.implied_z_clamp_ratio)
    if cal.implied_z_scale_ema is None:
        cal.implied_z_scale_ema = z_scale_raw
    else:
        z_scale_raw = clamp(
            z_scale_raw,
            float(cal.implied_z_scale_ema) / clamp_ratio,
            float(cal.implied_z_scale_ema) * clamp_ratio,
        )
        cal.implied_z_scale_ema = (1.0 - z_alpha) * float(cal.implied_z_scale_ema) + z_alpha * z_scale_raw


def fair_prob_yes_live(
    *,
    S: float,
    cal: CalibrationState,
    args: argparse.Namespace,
) -> Optional[float]:
    """Compute fair YES probability using anchor-blend model."""
    if (
        cal.slope_ref_S is None
        or cal.slope_ref_yes_mid is None
        or cal.slope_ref_sigma is None
        or cal.implied_z_scale_ema is None
    ):
        return None
    if S is None or not math.isfinite(float(S)) or float(S) <= 0:
        return None
    if float(cal.slope_ref_S) <= 0:
        return None
    K = float(cal.strike_k)
    if K <= 0:
        return None
    T = time_left_years(cal.market_end_ts)
    if T * SEC_PER_YEAR < float(args.min_T_sec):
        return None

    ref_S = float(cal.slope_ref_S)
    ref_yes_mid = float(cal.slope_ref_yes_mid)
    ref_sigma = float(cal.slope_ref_sigma)
    z_scale_ema = max(1e-6, float(cal.implied_z_scale_ema))

    if cal.slope_ref_wall_ts is not None and float(cal.slope_ref_wall_ts) > 0:
        T_ref = max(1e-12, (float(cal.market_end_ts) - float(cal.slope_ref_wall_ts)) / SEC_PER_YEAR)
    else:
        T_ref = T

    # 1. Local slope
    slope_per_usd = bs_digital_prob_slope_numeric(ref_S, K, ref_sigma, T_ref, float(args.slope_eps_frac))
    if slope_per_usd is None:
        return clamp(bs_digital_prob(float(S), K, ref_sigma, T), float(args.price_min), float(args.price_max))

    pred_mid_slope = clamp(
        ref_yes_mid + float(slope_per_usd) * (float(S) - ref_S),
        0.001, 0.999,
    )

    # 2. Implied-z
    move_log = math.log(float(S) / ref_S) if ref_S > 0 else 0.0
    move_bps = abs(move_log) * 10000.0
    z0 = norm_ppf(clamp(ref_yes_mid, 0.001, 0.999))
    pred_mid_z = clamp(norm_cdf(z0 + (move_log / z_scale_ema)), 0.001, 0.999)

    # 3. Blend
    w = sigmoid((move_bps - float(args.blend_mid_bps)) / max(1e-6, float(args.blend_width_bps)))
    pred_blend = ((1.0 - w) * pred_mid_slope) + (w * pred_mid_z)
    return clamp(float(pred_blend), float(args.price_min), float(args.price_max))


# ====================================================================
# Edges
# ====================================================================

def compute_edges_pp(
    yes_ask: Optional[float], no_ask: Optional[float], fair_yes: float,
) -> tuple[Optional[float], Optional[float]]:
    if yes_ask is None or no_ask is None:
        return None, None
    edge_yes = (float(fair_yes) - float(yes_ask)) * 100.0
    edge_no = ((1.0 - float(fair_yes)) - float(no_ask)) * 100.0
    return float(edge_yes), float(edge_no)


# ====================================================================
# State vector builder (same 12 features as RL_1_hour.py)
# ====================================================================

def build_state_vec(
    *,
    mid: float,
    fair_yes: float,
    edge_pp: float,
    spread_c: float,
    dP_cents: float,
    dF_cents: float,
    dS: float,
    poly_reaction: float,
    inv_yes: float,
    inv_no: float,
    max_net: float,
    max_yes: float,
    max_no: float,
    side: str,
    bin_dS_scale: float,
) -> np.ndarray:
    inv_net = float(inv_yes - inv_no)
    inv_gross = float(inv_yes + inv_no)
    cap_net = max(1.0, max_net)
    cap_yes = max(1.0, max_yes)
    cap_no = max(1.0, max_no)
    inv_side = inv_yes if side == "YES" else inv_no
    inv_other = inv_no if side == "YES" else inv_yes
    inv_scaled = float(inv_side) / (cap_yes if side == "YES" else cap_no)
    inv_other_scaled = float(inv_other) / (cap_no if side == "YES" else cap_yes)
    inv_net_scaled = float(inv_net) / cap_net
    inv_gross_scaled = float(inv_gross) / max(1.0, cap_yes + cap_no)

    return np.asarray(
        [
            float(mid),
            float(fair_yes),
            float(edge_pp) / 10.0,
            float(spread_c) / 10.0,
            softsign(float(dP_cents) / 10.0),
            softsign(float(dF_cents) / 10.0),
            softsign(float(dS) / max(EPS, bin_dS_scale)),
            softsign(float(poly_reaction)),
            float(inv_scaled),
            float(inv_other_scaled),
            float(inv_net_scaled),
            float(inv_gross_scaled),
        ],
        dtype=np.float32,
    )


# ====================================================================
# Trader (order submission)  – copied from 1_hour_latency_trader
# ====================================================================

def tune_clob_http_client(keepalive_sec: float, max_conns: int) -> None:
    try:
        import httpx
        from py_clob_client.http_helpers import helpers as clob_http_helpers
        old_client = getattr(clob_http_helpers, "_http_client", None)
        new_client = httpx.Client(
            http2=True,
            limits=httpx.Limits(
                max_connections=max(1, int(max_conns)),
                max_keepalive_connections=max(1, int(max_conns)),
                keepalive_expiry=max(30.0, float(keepalive_sec)),
            ),
            timeout=httpx.Timeout(connect=1.0, read=5.0, write=5.0, pool=0.5),
        )
        clob_http_helpers._http_client = new_client
        if old_client is not None and old_client is not new_client:
            try:
                old_client.close()
            except Exception:
                pass
    except Exception:
        return


class Trader:
    def __init__(self, live: bool) -> None:
        self.live = bool(live)
        self.client: Optional[ClobClient] = None
        self.io_executor: Optional[ThreadPoolExecutor] = None
        self._token_tick_cache: dict[str, float] = {}
        self._token_size_decimals_cache: dict[str, int] = {}

    async def _run_client_io(self, fn: Any, *a: Any) -> Any:
        loop = asyncio.get_running_loop()
        if self.io_executor is None:
            return await asyncio.to_thread(fn, *a)
        return await loop.run_in_executor(self.io_executor, functools.partial(fn, *a))

    def init(self) -> None:
        if not self.live:
            return
        tune_clob_http_client(1800.0, 8)
        key = os.environ.get("POLYMARKET_PRIVATE_KEY", "").strip()
        if not key:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY missing for --live")
        funder = os.environ.get("POLYMARKET_PROXY", "").strip() or None
        st_raw = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
        st_val = int(st_raw.lstrip("$")) if st_raw else (1 if funder else 0)
        chain = int(float(os.environ.get("POLYMARKET_CHAIN_ID", "137")))
        host = os.environ.get("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com").strip()
        c = ClobClient(host, key=key, chain_id=chain, signature_type=st_val, funder=funder)
        c.set_api_creds(c.create_or_derive_api_creds())
        self.client = c
        self.io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clob_io")

    async def prewarm_token_metadata(self, token_ids: list[str]) -> list[str]:
        if self.client is None:
            return []
        tids = [str(t).strip() for t in token_ids if str(t).strip()]
        if not tids:
            return []

        def _f() -> list[str]:
            warmed: list[str] = []
            for tid in tids:
                try:
                    tick_raw = self.client.get_tick_size(tid)
                    self.client.get_neg_risk(tid)
                    self.client.get_fee_rate_bps(tid)
                    tick = float(tick_raw)
                    self._token_tick_cache[tid] = tick
                    tick_txt = str(tick)
                    rc = ROUNDING_CONFIG.get(tick_txt) or ROUNDING_CONFIG.get(str(tick_raw))
                    if rc is not None:
                        self._token_size_decimals_cache[tid] = int(rc.size)
                    warmed.append(tid)
                except Exception:
                    continue
            return warmed
        return await asyncio.to_thread(_f)

    def token_constraints(
        self, token_id: str, *, default_tick: float, default_size_decimals: int,
        default_price_min: float, default_price_max: float,
    ) -> tuple[float, int, float, float]:
        tick = float(default_tick)
        size_decimals = int(default_size_decimals)
        if self.client is not None and token_id:
            try:
                tick = self._token_tick_cache.get(token_id, tick)
                if token_id not in self._token_tick_cache:
                    tick_raw = self.client.get_tick_size(token_id)
                    tick = float(tick_raw)
                    self._token_tick_cache[token_id] = tick
                tick = max(1e-6, float(tick))
                size_decimals = self._token_size_decimals_cache.get(token_id, size_decimals)
                if token_id not in self._token_size_decimals_cache:
                    tick_txt = str(tick)
                    rc = ROUNDING_CONFIG.get(tick_txt)
                    if rc is not None:
                        self._token_size_decimals_cache[token_id] = int(rc.size)
            except Exception:
                pass
        price_min = max(float(default_price_min), tick)
        price_max = min(float(default_price_max), max(price_min, 1.0 - tick))
        return tick, size_decimals, price_min, price_max

    async def place_buy(
        self, token_id: str, price: float, size: float, otype: Any,
    ) -> tuple[Optional[str], str]:
        if not self.live:
            return None, "dry_run"
        if self.client is None:
            return None, "no_client"

        def _f() -> tuple[Optional[str], str]:
            try:
                signed = self.client.create_order(
                    OrderArgs(price=price, size=size, side=BUY, token_id=token_id)
                )
                r = self.client.post_order(signed, otype)
                if isinstance(r, dict):
                    oid = r.get("orderID") or r.get("orderId") or r.get("id")
                    if oid:
                        return str(oid), ""
                for k in ("orderID", "orderId", "id"):
                    if hasattr(r, k):
                        v = getattr(r, k)
                        if v:
                            return str(v), ""
                return None, f"bad_place_response:{r}"
            except Exception as e:
                return None, str(e)
        return await self._run_client_io(_f)

    def close(self) -> None:
        if self.io_executor is not None:
            self.io_executor.shutdown(wait=False)
            self.io_executor = None


# ====================================================================
# Async buffered SQLite logger (off critical path)
# ====================================================================

_DB_STOP = object()


class AsyncDBLogger:
    """Threaded SQLite logger — never blocks the hot path."""

    def __init__(self, db_path: Path, max_queue: int = 50_000) -> None:
        self.db_path = Path(db_path)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.enabled = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="db_logger")
        self._thread.start()

    def _run(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_wall_iso TEXT,
                ts_ms INTEGER,
                slug TEXT,
                side TEXT,
                token_id TEXT,
                price REAL,
                size REAL,
                order_id TEXT,
                pred REAL,
                edge_pp REAL,
                fair_yes REAL,
                bin_mid REAL,
                inv_yes REAL,
                inv_no REAL,
                mode TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER,
                slug TEXT,
                bin_mid REAL,
                yes_ask REAL,
                no_ask REAL,
                yes_bid REAL,
                no_bid REAL,
                fair_yes REAL,
                source TEXT
            )
        """)
        conn.commit()

        buf: list[tuple[str, tuple]] = []
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                if buf:
                    self._flush(conn, buf)
                    buf.clear()
                continue
            if item is _DB_STOP:
                if buf:
                    self._flush(conn, buf)
                conn.close()
                return
            buf.append(item)  # type: ignore
            # Drain non-blocking
            while len(buf) < 500:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if item is _DB_STOP:
                    self._flush(conn, buf)
                    conn.close()
                    return
                buf.append(item)  # type: ignore
            if len(buf) >= 200:
                self._flush(conn, buf)
                buf.clear()

    @staticmethod
    def _flush(conn: sqlite3.Connection, buf: list[tuple[str, tuple]]) -> None:
        try:
            for sql, params in buf:
                conn.execute(sql, params)
            conn.commit()
        except Exception as e:
            log.warning("DB flush error: %s", e)

    def log_trade(self, row: tuple) -> None:
        sql = (
            "INSERT INTO live_trades "
            "(ts_wall_iso,ts_ms,slug,side,token_id,price,size,order_id,pred,edge_pp,fair_yes,bin_mid,inv_yes,inv_no,mode) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        try:
            self._q.put_nowait((sql, row))
        except queue.Full:
            pass

    def log_snapshot(self, row: tuple) -> None:
        sql = (
            "INSERT INTO live_snapshots "
            "(ts_ms,slug,bin_mid,yes_ask,no_ask,yes_bid,no_bid,fair_yes,source) "
            "VALUES (?,?,?,?,?,?,?,?,?)"
        )
        try:
            self._q.put_nowait((sql, row))
        except queue.Full:
            pass

    def close(self) -> None:
        try:
            self._q.put(_DB_STOP, timeout=2.0)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)


class NullDBLogger:
    enabled = False

    def log_trade(self, row: tuple) -> None:
        return

    def log_snapshot(self, row: tuple) -> None:
        return

    def close(self) -> None:
        return


# ====================================================================
# Background retrainer (runs in a separate process)
# ====================================================================

def _retrain_worker(
    historical_db_path: str,
    live_db_path: str,
    current_weights_path: str,
    output_weights_path: str,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    done_event_path: str,
) -> None:
    """
    Fine-tune the RegNet on historical + live data.  Write new weights to
    *output_weights_path* and touch *done_event_path* when finished.

    This function runs in a child process and NEVER touches the hot path.
    """
    import sys
    try:
        # Import RL_1_hour helpers
        sys.path.insert(0, str(Path(historical_db_path).parent))
        from RL_1_hour import (
            load_rows as rl_load_rows,
            materialize_samples,
            RegNet as RLRegNet,
            batch_iter,
        )

        # Load historical rows
        hist_rows = rl_load_rows(Path(historical_db_path))

        # Build args namespace for sample building  (defaults match RL_1_hour)
        ns = argparse.Namespace(
            fair_move_min_cents=0.10,
            max_poly_move_cents=0.0,
            poly_both_moved_cents=1.0,
            max_usables=None,
            entry_cross_cost_cents=0.0,
            label_clip_usd=0.0,
            max_net=250.0,
            max_yes=500.0,
            max_no=500.0,
            bin_dS_scale=50.0,
            entry_notional=5.0,
            max_qty_per_entry=50,
            max_inventory_per_side=1000.0,
        )

        X, y = materialize_samples(hist_rows, args=ns)
        if X.shape[0] == 0:
            log.warning("[retrain] no samples built, skipping")
            return

        state_dim = int(X.shape[1])
        model = RLRegNet(state_dim, hidden=hidden)

        # Load current weights
        if Path(current_weights_path).exists():
            sd = torch.load(current_weights_path, map_location="cpu")
            model.load_state_dict(sd, strict=False)

        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.HuberLoss(delta=1.0)
        rng = np.random.default_rng(42)

        for ep in range(epochs):
            model.train()
            for xb, yb in batch_iter(X, y, batch_size, rng):
                xt = torch.from_numpy(xb)
                yt = torch.from_numpy(yb)
                pred = model(xt)
                loss = loss_fn(pred, yt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        # Save new weights atomically
        tmp = output_weights_path + ".tmp"
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, output_weights_path)

        # Signal completion
        Path(done_event_path).write_text("done")
        log.info("[retrain] completed, weights -> %s", output_weights_path)

    except Exception:
        log.exception("[retrain] worker failed")


# ====================================================================
# Main strategy loop  (HOT PATH — inference only)
# ====================================================================

async def strategy_loop(
    shutdown: asyncio.Event,
    snap: ds.Snapshot,
    q: asyncio.Queue,
    trader: Trader,
    model: RegNet,
    args: argparse.Namespace,
    db_logger: AsyncDBLogger | NullDBLogger,
    model_lock: threading.Lock,
) -> None:
    """
    Core trading loop.  Runs in the async event loop.
    Model inference is the ONLY compute on the critical path.
    """
    device = torch.device("cpu")
    model.eval()

    cal = CalibrationState()

    # ---- Inventory tracking ----
    inv_yes = 0.0
    inv_no = 0.0
    inv_yes_cost = 0.0
    inv_no_cost = 0.0

    # ---- Previous-tick state (for deltas) ----
    prev_bin_mid: Optional[float] = None
    prev_yes_mid: Optional[float] = None
    prev_no_mid: Optional[float] = None
    prev_fair_yes: Optional[float] = None
    prev_slug: str = ""

    # ---- Market hour tracking ----
    current_slug: str = ""

    # ---- Token warmup done? ----
    warmed: bool = False

    events_total = 0
    events_binance = 0
    events_poly = 0
    last_trade_log_s = 0.0

    log.info("[strategy] loop started (dry_run=%s)", args.dry_run)

    while not shutdown.is_set():
        try:
            source = await asyncio.wait_for(q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        events_total += 1
        if source == "binance":
            events_binance += 1
        else:
            events_poly += 1

        # ---- Read snapshot ----
        slug = str(snap.market_slug or "")
        bin_mid = snap.binance_mid
        yes_bid = snap.poly_yes_bid
        yes_ask = snap.poly_yes_ask
        no_bid = snap.poly_no_bid
        no_ask = snap.poly_no_ask
        yes_token = snap.yes_token_id
        no_token = snap.no_token_id

        # ---- Market rollover: reset state ----
        if slug and slug != current_slug:
            if current_slug:
                log.info(
                    "[strategy] market rollover: %s -> %s | settled inv YES=%.1f NO=%.1f",
                    current_slug, slug, inv_yes, inv_no,
                )
            current_slug = slug
            inv_yes = 0.0
            inv_no = 0.0
            inv_yes_cost = 0.0
            inv_no_cost = 0.0
            prev_bin_mid = None
            prev_yes_mid = None
            prev_no_mid = None
            prev_fair_yes = None
            prev_slug = slug
            warmed = False

            # Derive market params from slug
            hw = parse_slug_hour_window(slug)
            if hw:
                _, end_et = hw
                cal.market_end_ts = end_et.timestamp()
            else:
                cal.market_end_ts = time.time() + 3600.0

            # Reset calibration
            cal.sigma_raw = None
            cal.sigma_eff_ema = None
            cal.implied_z_scale_ema = None
            cal.slope_ref_S = None
            cal.slope_ref_yes_mid = None
            cal.slope_ref_sigma = None
            cal.slope_ref_wall_ts = None

        if not slug:
            continue

        # ---- Need both sides of the book + Binance ----
        if bin_mid is None or yes_bid is None or yes_ask is None or no_bid is None or no_ask is None:
            continue

        # ---- Token prewarm (once per market) ----
        if not warmed and trader.live:
            tids = [t for t in [yes_token, no_token] if t]
            if tids:
                try:
                    await trader.prewarm_token_metadata(tids)
                except Exception:
                    pass
            warmed = True

        # ---- Compute mids ----
        y_mid = (float(yes_bid) + float(yes_ask)) * 0.5
        n_mid = (float(no_bid) + float(no_ask)) * 0.5

        # ---- Strike from kline if available ----
        if cal.strike_k <= 0:
            if snap.kline_open is not None:
                cal.strike_k = float(snap.kline_open)
                cal.strike_k_source = "kline"
            elif bin_mid is not None:
                cal.strike_k = float(bin_mid)
                cal.strike_k_source = "first_binance_mid"

        # ---- Calibrate on Polymarket update ----
        if source == "polymarket" and bin_mid is not None:
            calibrate_on_poly_update(cal=cal, S=float(bin_mid), y_mid=y_mid, args=args)
            # Log snapshot (off hot path)
            db_logger.log_snapshot((
                int(time.time_ns() // 1_000_000), slug,
                float(bin_mid), float(yes_ask), float(no_ask),
                float(yes_bid), float(no_bid),
                None, "polymarket",
            ))

        # ---- Only trade on Binance updates (dislocation trigger) ----
        if source != "binance":
            prev_yes_mid = y_mid
            prev_no_mid = n_mid
            prev_slug = slug
            continue

        # ---- Compute fair value ----
        fair_yes = fair_prob_yes_live(S=float(bin_mid), cal=cal, args=args)
        if fair_yes is None:
            prev_bin_mid = float(bin_mid)
            prev_yes_mid = y_mid
            prev_no_mid = n_mid
            prev_slug = slug
            continue

        # Log snapshot (off hot path)
        db_logger.log_snapshot((
            int(time.time_ns() // 1_000_000), slug,
            float(bin_mid), float(yes_ask), float(no_ask),
            float(yes_bid), float(no_bid),
            float(fair_yes), "binance",
        ))

        # ---- Compute edges ----
        edge_yes_pp, edge_no_pp = compute_edges_pp(yes_ask, no_ask, fair_yes)
        if edge_yes_pp is None or edge_no_pp is None:
            prev_bin_mid = float(bin_mid)
            prev_yes_mid = y_mid
            prev_no_mid = n_mid
            prev_fair_yes = float(fair_yes)
            prev_slug = slug
            continue

        # ---- Determine best side ----
        best_side = "YES" if edge_yes_pp >= edge_no_pp else "NO"

        # ---- Dislocation filter: fair moved, poly hasn't updated much ----
        dF_cents = 0.0 if prev_fair_yes is None else (float(fair_yes) - float(prev_fair_yes)) * 100.0
        dP_yes = 0.0 if prev_yes_mid is None else (y_mid - float(prev_yes_mid)) * 100.0
        dP_no = 0.0 if prev_no_mid is None else (n_mid - float(prev_no_mid)) * 100.0
        dS = 0.0 if prev_bin_mid is None else (float(bin_mid) - float(prev_bin_mid))

        fair_moved = abs(dF_cents) >= float(args.fair_move_min_cents)
        poly_not_moved = not (
            abs(dP_yes) > float(args.poly_both_moved_cents)
            and abs(dP_no) > float(args.poly_both_moved_cents)
        )

        if not (fair_moved and poly_not_moved):
            prev_bin_mid = float(bin_mid)
            prev_yes_mid = y_mid
            prev_no_mid = n_mid
            prev_fair_yes = float(fair_yes)
            prev_slug = slug
            continue

        # ---- Build state vector ----
        edge_pp = float(edge_yes_pp) if best_side == "YES" else float(edge_no_pp)
        spread_c = ((float(yes_ask) - float(yes_bid)) if best_side == "YES"
                     else (float(no_ask) - float(no_bid))) * 100.0
        dP_cents = dP_yes if best_side == "YES" else dP_no
        poly_reaction = abs(float(dP_cents)) / (abs(float(dF_cents)) + 1e-6)

        mid_for_state = y_mid if best_side == "YES" else n_mid

        state = build_state_vec(
            mid=mid_for_state, fair_yes=fair_yes,
            edge_pp=edge_pp, spread_c=spread_c,
            dP_cents=dP_cents, dF_cents=dF_cents, dS=dS,
            poly_reaction=poly_reaction,
            inv_yes=inv_yes, inv_no=inv_no,
            max_net=float(args.max_net), max_yes=float(args.max_yes),
            max_no=float(args.max_no), side=best_side,
            bin_dS_scale=float(args.bin_dS_scale),
        )

        # ---- Model inference ----
        with model_lock:
            with torch.no_grad():
                x = torch.from_numpy(state.reshape(1, -1)).to(device)
                pred = float(model(x).detach().cpu().item())

        # ---- Trade decision ----
        if pred < float(args.min_pred):
            prev_bin_mid = float(bin_mid)
            prev_yes_mid = y_mid
            prev_no_mid = n_mid
            prev_fair_yes = float(fair_yes)
            prev_slug = slug
            continue

        trade_side = best_side
        if args.reverse_strategy:
            trade_side = "NO" if best_side == "YES" else "YES"

        # ---- Sizing: $5 notional at the ask ----
        ask = float(yes_ask) if trade_side == "YES" else float(no_ask)
        if ask <= 0:
            prev_bin_mid = float(bin_mid)
            prev_yes_mid = y_mid
            prev_no_mid = n_mid
            prev_fair_yes = float(fair_yes)
            prev_slug = slug
            continue

        raw_qty = float(args.entry_notional) / ask
        qty = min(raw_qty, float(args.max_qty_per_entry))

        # Inventory cap
        cur_inv = inv_yes if trade_side == "YES" else inv_no
        room = max(0.0, float(args.max_inv_per_side) - cur_inv)
        qty = min(qty, room)
        if qty <= 0:
            prev_bin_mid = float(bin_mid)
            prev_yes_mid = y_mid
            prev_no_mid = n_mid
            prev_fair_yes = float(fair_yes)
            prev_slug = slug
            continue

        # ---- Determine token_id ----
        token_id = yes_token if trade_side == "YES" else no_token
        if not token_id:
            prev_bin_mid = float(bin_mid)
            prev_yes_mid = y_mid
            prev_no_mid = n_mid
            prev_fair_yes = float(fair_yes)
            prev_slug = slug
            continue

        # ---- Normalize price & size ----
        tick, size_decimals, pmin, pmax = trader.token_constraints(
            token_id,
            default_tick=float(args.price_tick),
            default_size_decimals=int(args.size_decimals),
            default_price_min=float(args.price_min),
            default_price_max=float(args.price_max),
        )
        entry_px = clamp(ask, pmin, pmax)
        size = norm_buy_size_for_amount_constraints(qty, entry_px, amount_step_decimals=4)
        size = norm_size(size, size_decimals)
        min_shares = max(0.0, float(args.min_order_shares))
        if size < min_shares:
            size = math.ceil(min_shares * (10 ** size_decimals)) / (10 ** size_decimals)
        if size <= 0:
            prev_bin_mid = float(bin_mid)
            prev_yes_mid = y_mid
            prev_no_mid = n_mid
            prev_fair_yes = float(fair_yes)
            prev_slug = slug
            continue

        # ---- Submit order ----
        t_order_ns = time.monotonic_ns()
        oid, err = await trader.place_buy(token_id, entry_px, size, OrderType.GTC)
        latency_ms = (time.monotonic_ns() - t_order_ns) / 1_000_000

        if args.dry_run:
            oid = f"DRY_{int(time.time_ns())}"
            err = ""

        if err and "dry_run" not in err:
            log.warning("[trade] order error: %s", err)
        else:
            # Update inventory (assume full fill at ask for accumulation)
            if trade_side == "YES":
                inv_yes += size
                inv_yes_cost += entry_px * size
            else:
                inv_no += size
                inv_no_cost += entry_px * size

            now = time.time()
            if now - last_trade_log_s > 0.5:  # Throttle log prints
                log.info(
                    "[trade] %s %s %.1f@%.4f pred=%.3f edge=%.2fpp inv=Y%.0f/N%.0f lat=%.1fms oid=%s",
                    "DRY" if args.dry_run else "LIVE",
                    trade_side, size, entry_px, pred, edge_pp,
                    inv_yes, inv_no, latency_ms, str(oid or "")[:16],
                )
                last_trade_log_s = now

            # Log trade (off hot path)
            db_logger.log_trade((
                datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                int(time.time_ns() // 1_000_000),
                slug, trade_side, token_id, entry_px, size,
                str(oid or ""), pred, edge_pp, float(fair_yes),
                float(bin_mid), inv_yes, inv_no,
                "dry" if args.dry_run else "live",
            ))

        # ---- Update previous-tick ----
        prev_bin_mid = float(bin_mid)
        prev_yes_mid = y_mid
        prev_no_mid = n_mid
        prev_fair_yes = float(fair_yes)
        prev_slug = slug


# ====================================================================
# Hourly weight hot-swap scheduler
# ====================================================================

async def retrain_scheduler(
    shutdown: asyncio.Event,
    model: RegNet,
    model_lock: threading.Lock,
    args: argparse.Namespace,
) -> None:
    """
    Runs in the event loop.  At each hour boundary:
      1. Spawn a child process to fine-tune the model on historical data.
      2. Poll for completion (non-blocking).
      3. Hot-swap weights atomically via model_lock.
    """
    if not args.enable_retrain:
        log.info("[retrain] disabled (--no-retrain)")
        return

    historical_db = str(Path(args.historical_db).resolve())
    live_db = str(Path(args.live_db).resolve())
    weights_path = str(Path(args.model_path).resolve())
    new_weights_path = weights_path + ".new"
    done_flag = weights_path + ".retrain_done"

    log.info("[retrain] scheduler started (interval=hourly)")

    while not shutdown.is_set():
        # Wait until next hour boundary
        now = datetime.now(ET)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        wait_sec = max(1.0, (next_hour - now).total_seconds())
        log.info("[retrain] next retrain at %s (in %.0fs)", next_hour.strftime("%H:%M"), wait_sec)

        try:
            await asyncio.wait_for(shutdown.wait(), timeout=wait_sec)
            return  # shutdown
        except asyncio.TimeoutError:
            pass

        # Clean up any previous done flag
        Path(done_flag).unlink(missing_ok=True)

        # Save current weights for the worker to load
        with model_lock:
            torch.save(model.state_dict(), weights_path)

        log.info("[retrain] spawning retrainer process")
        proc = multiprocessing.Process(
            target=_retrain_worker,
            kwargs=dict(
                historical_db_path=historical_db,
                live_db_path=live_db,
                current_weights_path=weights_path,
                output_weights_path=new_weights_path,
                hidden=int(args.hidden),
                epochs=int(args.retrain_epochs),
                lr=float(args.retrain_lr),
                batch_size=int(args.retrain_batch_size),
                done_event_path=done_flag,
            ),
            daemon=True,
        )
        proc.start()

        # Poll for completion without blocking
        while not shutdown.is_set():
            proc.join(timeout=2.0)
            if not proc.is_alive():
                break
            if Path(done_flag).exists():
                break

        # Hot-swap weights if retrainer produced them
        if Path(new_weights_path).exists():
            try:
                new_sd = torch.load(new_weights_path, map_location="cpu")
                with model_lock:
                    model.load_state_dict(new_sd, strict=False)
                    model.eval()
                log.info("[retrain] HOT-SWAP: loaded new weights into live model")
                Path(new_weights_path).unlink(missing_ok=True)
            except Exception:
                log.exception("[retrain] failed to load new weights")
        else:
            log.warning("[retrain] no new weights produced")

        Path(done_flag).unlink(missing_ok=True)

        # Ensure process is fully dead
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)


# ====================================================================
# Heartbeat / status printer
# ====================================================================

async def heartbeat_loop(
    shutdown: asyncio.Event,
    snap: ds.Snapshot,
    args: argparse.Namespace,
) -> None:
    interval = 30.0
    while not shutdown.is_set():
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        slug = snap.market_slug or "(none)"
        bm = f"{snap.binance_mid:.2f}" if snap.binance_mid else "?"
        ya = f"{snap.poly_yes_ask:.4f}" if snap.poly_yes_ask else "?"
        na = f"{snap.poly_no_ask:.4f}" if snap.poly_no_ask else "?"
        log.info("[heartbeat] slug=%s  BTC=%s  YES_ask=%s  NO_ask=%s", slug, bm, ya, na)


# ====================================================================
# CLI
# ====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL-driven live Polymarket latency trader")

    # ---- Mode ----
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true", help="Submit real orders")
    g.add_argument("--dry-run", action="store_true", help="Paper trading (no real orders)")

    # ---- Model ----
    _script_dir = Path(__file__).resolve().parent
    _default_model = str(_script_dir / "reg_model_state.pt")
    p.add_argument("--model-path", type=str, default=_default_model,
                   help="Path to RegNet state_dict (.pt)")
    p.add_argument("--hidden", type=int, default=128, help="Hidden layer size (must match training)")

    # ---- Decision ----
    p.add_argument("--min-pred", type=float, default=-0.3,
                   help="Trade if model prediction >= this threshold")
    p.add_argument("--reverse-strategy", action="store_true",
                   help="Trade the opposite side of what the model recommends")

    # ---- Sizing ----
    p.add_argument("--entry-notional", type=float, default=5.0, help="USD per trade")
    p.add_argument("--max-qty-per-entry", type=float, default=50.0, help="Max contracts per entry")
    p.add_argument("--max-inv-per-side", type=float, default=1000.0,
                   help="Max total contracts per side (YES or NO)")

    # ---- Inventory caps (for state features) ----
    p.add_argument("--max-yes", type=float, default=500.0)
    p.add_argument("--max-no", type=float, default=500.0)
    p.add_argument("--max-net", type=float, default=250.0)
    p.add_argument("--bin-dS-scale", type=float, default=50.0)

    # ---- Dislocation filter ----
    p.add_argument("--fair-move-min-cents", type=float, default=0.10)
    p.add_argument("--poly-both-moved-cents", type=float, default=1.0)

    # ---- BS-IV fair-value model ----
    p.add_argument("--sigma-ema-alpha", type=float, default=0.15)
    p.add_argument("--implied-z-ema-alpha", type=float, default=0.10)
    p.add_argument("--implied-z-clamp-ratio", type=float, default=3.0)
    p.add_argument("--slope-eps-frac", type=float, default=1e-5)
    p.add_argument("--blend-mid-bps", type=float, default=30.0)
    p.add_argument("--blend-width-bps", type=float, default=15.0)
    p.add_argument("--min-T-sec", type=float, default=60.0,
                   help="Don't trade if < this many seconds left in market")
    p.add_argument("--price-min", type=float, default=0.01)
    p.add_argument("--price-max", type=float, default=0.99)
    p.add_argument("--price-tick", type=float, default=0.01)
    p.add_argument("--size-decimals", type=int, default=0)
    p.add_argument("--min-order-shares", type=float, default=1.0)

    # ---- Retraining ----
    p.add_argument("--no-retrain", action="store_true", help="Disable hourly retraining")
    _default_hist_db = str(_script_dir / "trader_log.snapshot.db")
    p.add_argument("--historical-db", type=str, default=_default_hist_db,
                   help="Path to historical opportunity DB for retraining")
    _default_live_db = str(_script_dir / "rl_live_trades.db")
    p.add_argument("--live-db", type=str, default=_default_live_db,
                   help="Path to live trade/snapshot DB")
    p.add_argument("--retrain-epochs", type=int, default=10, help="Epochs per retrain cycle")
    p.add_argument("--retrain-lr", type=float, default=5e-4, help="Learning rate for fine-tuning")
    p.add_argument("--retrain-batch-size", type=int, default=512)

    return p.parse_args()


# ====================================================================
# Main
# ====================================================================

def _autoload_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        if key and key not in os.environ:
            vv = v.strip()
            if len(vv) >= 2 and vv[0] == vv[-1] and vv[0] in ("'", '"'):
                vv = vv[1:-1]
            os.environ[key] = vv


async def async_main(args: argparse.Namespace) -> None:
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            signal.signal(sig, lambda _s, _f: shutdown.set())

    # ---- Load model ----
    model_path = Path(args.model_path)
    model = RegNet(in_dim=STATE_DIM, hidden=int(args.hidden))
    if model_path.exists():
        sd = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(sd, strict=False)
        log.info("Loaded model from %s", model_path)
    else:
        log.warning("No model file at %s — starting with random weights!", model_path)
    model.eval()
    model_lock = threading.Lock()

    # ---- Set up Trader ----
    trader = Trader(live=args.live)
    if args.live:
        trader.init()
        log.info("Trader initialized (LIVE mode)")
    else:
        log.info("Trader initialized (DRY-RUN mode)")

    # ---- DB logger ----
    db_logger: AsyncDBLogger | NullDBLogger
    db_logger = AsyncDBLogger(Path(args.live_db))
    log.info("DB logger -> %s", args.live_db)

    # ---- Data streams ----
    snap = ds.Snapshot()
    q: asyncio.Queue = asyncio.Queue(maxsize=1)

    # ---- Retrain flag ----
    args.enable_retrain = not args.no_retrain

    mode = "LIVE" if args.live else "DRY-RUN"
    rev = " (REVERSED)" if args.reverse_strategy else ""
    log.info("=" * 60)
    log.info("  RL Live Trader — %s%s", mode, rev)
    log.info("  Model:     %s", args.model_path)
    log.info("  Threshold: pred >= %.3f", args.min_pred)
    log.info("  Sizing:    $%.2f/trade, max %d/entry, max %.0f/side",
             args.entry_notional, int(args.max_qty_per_entry), args.max_inv_per_side)
    log.info("  Retrain:   %s (every hour)", "ON" if args.enable_retrain else "OFF")
    log.info("=" * 60)

    try:
        await asyncio.gather(
            ds._polymarket_loop(shutdown, snap, q),
            ds._binance_loop(shutdown, snap, q),
            ds._binance_kline_loop(shutdown, snap),
            strategy_loop(shutdown, snap, q, trader, model, args, db_logger, model_lock),
            heartbeat_loop(shutdown, snap, args),
            retrain_scheduler(shutdown, model, model_lock, args),
        )
    finally:
        db_logger.close()
        trader.close()
        # Save final weights
        with model_lock:
            torch.save(model.state_dict(), str(model_path))
        log.info("Shutdown complete. Model saved to %s", model_path)


def main() -> None:
    _autoload_env()
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
