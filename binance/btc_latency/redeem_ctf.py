#!/usr/bin/env python3
"""
redeem_ctf.py  --  Redeem resolved Polymarket CTF tokens into USDC.e.

Supports BOTH regular CTF markets AND NegRisk markets (e.g. "bitcoin up or down").

Env vars (loaded from .env if present):
  POLYMARKET_PRIVATE_KEY = Private key for the wallet holding the tokens
  POLYGON_RPC_URL        = (optional) Polygon RPC URL. Falls back to public RPCs if not set.
  POLYMARKET_PROXY       = (optional) Your Polymarket proxy wallet address.
  WALLET_ADDR            = (optional) Alias for proxy wallet.
  CONDITION_IDS          = (optional) Comma-separated bytes32 conditionIds to check.

Usage:
  # Search for a specific market by slug and redeem:
  python redeem_ctf.py --slug bitcoin-up-or-down-february-28 --dry-run --once

  # Auto-discover all positions and redeem:
  python redeem_ctf.py --dry-run --once

  # Loop mode:
  python redeem_ctf.py --interval-min 15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from web3 import Web3
from web3.exceptions import ContractLogicError

# ====================================================================
# Logging
# ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("redeem_ctf")

# ====================================================================
# Constants
# ====================================================================
USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
DEFAULT_CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Polymarket NegRisk Adapter (ERC-1155 that wraps CTF positions for NegRisk markets)
NEG_RISK_ADAPTER = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")
# Alternative: NegRisk CTF Exchange (also check here)
NEG_RISK_CTF_EXCHANGE = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")

ZERO_BYTES32 = "0x" + "00" * 32
ZERO_BYTES32_RAW = b"\x00" * 32
COLLATERAL_DECIMALS = 6

# Public Polygon RPC endpoints (tried in order)
FALLBACK_RPC_URLS = [
    "https://polygon-rpc.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://polygon-bor-rpc.publicnode.com",
]

# Polymarket Gamma API
GAMMA_API = "https://gamma-api.polymarket.com"

# ====================================================================
# ABIs
# ====================================================================
CTF_ABI = [
    {
        "name": "payoutDenominator",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "payoutNumerators",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "index", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "getCollectionId",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSet", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bytes32"}],
    },
    {
        "name": "getPositionId",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "collectionId", "type": "bytes32"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "redeemPositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "outputs": [],
    },
]

# Gnosis Safe execTransaction ABI (for proxy wallet calls)
SAFE_EXEC_ABI = [
    {
        "name": "execTransaction",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "signatures", "type": "bytes"},
        ],
        "outputs": [{"name": "success", "type": "bool"}],
    },
    {
        "name": "nonce",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "getTransactionHash",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "_nonce", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bytes32"}],
    },
]

_BYTES32_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")

# ====================================================================
# Graceful shutdown
# ====================================================================
_SHUTDOWN = False


def _signal_handler(signum, frame):
    global _SHUTDOWN
    _SHUTDOWN = True
    log.info("Shutdown signal received, will exit after current cycle.")


# ====================================================================
# Helpers
# ====================================================================

def load_env() -> None:
    """Load .env file if present (same directory or parent)."""
    for candidate in [Path(".env"), Path(__file__).resolve().parent / ".env",
                      Path(__file__).resolve().parent.parent / ".env"]:
        if candidate.exists():
            log.info("Loading env from %s", candidate)
            for raw in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
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
            break


def must_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def parse_condition_ids(raw: str) -> Optional[List[str]]:
    """Parse condition IDs from env var. Returns None if empty (auto-discover mode)."""
    ids = [s.strip() for s in raw.split(",") if s.strip()]
    if not ids:
        return None
    for c in ids:
        if not _BYTES32_RE.match(c):
            raise RuntimeError(f"Not a valid bytes32 hex: {c}")
    return ids


# ====================================================================
# Gamma API helpers
# ====================================================================

def _parse_market_tokens(m: dict) -> Dict[str, str]:
    """Extract {token_id: outcome} from a Gamma API market object."""
    tokens: Dict[str, str] = {}
    # clobTokenIds + outcomes (JSON-encoded strings)
    raw_outcomes = m.get("outcomes", "[]")
    raw_clob_ids = m.get("clobTokenIds", "[]")
    outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
    clob_ids = json.loads(raw_clob_ids) if isinstance(raw_clob_ids, str) else raw_clob_ids
    for tid, outcome in zip(clob_ids, outcomes):
        tokens[str(tid)] = str(outcome)
    # Fallback: nested tokens array
    for tok in m.get("tokens", []):
        tid = tok.get("token_id") or tok.get("tokenId")
        outcome = tok.get("outcome", "?")
        if tid and str(tid) not in tokens:
            tokens[str(tid)] = str(outcome)
    return tokens


def fetch_market_by_slug(slug: str) -> Optional[dict]:
    """
    Resolve a Polymarket event slug via the Gamma API.
    Returns {condition_id, tokens: {token_id: outcome}, question, slug, neg_risk}
    or None if not found.
    """
    import requests as _req

    # Try exact slug first via /events/slug/<slug>
    url = f"{GAMMA_API}/events/slug/{slug}"
    event = None
    try:
        resp = _req.get(url, timeout=15)
        if resp.status_code == 200:
            event = resp.json()
    except Exception:
        pass

    if not event:
        # Try searching markets by slug substring
        log.info("  Exact slug not found, searching markets...")
        try:
            resp = _req.get(f"{GAMMA_API}/markets", params={"slug": slug}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    event = {"markets": data, "slug": slug}
        except Exception:
            pass

    if not event:
        # Try searching by slug substring in events (active + closed)
        for closed_flag in [None, "true"]:
            if event:
                break
            params: dict = {"limit": 100}
            if closed_flag:
                params["closed"] = closed_flag
            try:
                resp = _req.get(f"{GAMMA_API}/events", params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        for ev in data:
                            ev_slug = ev.get("slug", "")
                            if slug.lower() in ev_slug.lower() or ev_slug.lower() in slug.lower():
                                event = ev
                                log.info("  Matched event slug: %s", ev_slug)
                                break
            except Exception:
                pass

    if not event:
        # Try slug variations: add common time suffixes for hourly BTC markets
        slug_lower = slug.lower()
        if "am-et" not in slug_lower and "pm-et" not in slug_lower:
            log.info("  Trying time-suffix variations...")
            for hour in range(1, 13):
                for ampm in ["am", "pm"]:
                    candidate_slug = f"{slug}-{hour}{ampm}-et"
                    try:
                        resp = _req.get(f"{GAMMA_API}/events/slug/{candidate_slug}", timeout=10)
                        if resp.status_code == 200:
                            event = resp.json()
                            log.info("  Found match: %s", candidate_slug)
                            break
                    except Exception:
                        pass
                if event:
                    break

    if not event:
        return None

    raw_markets = event.get("markets", [])
    if not raw_markets:
        return None

    condition_id: Optional[str] = None
    question = slug
    tokens: Dict[str, str] = {}
    neg_risk = False

    for m in raw_markets:
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(m, dict):
            continue
        cid = m.get("condition_id") or m.get("conditionId")
        if cid:
            condition_id = str(cid)
        question = m.get("question", question)
        # Check neg_risk flag
        nr = m.get("neg_risk")
        if nr and str(nr).lower() not in ("false", "0", "none", ""):
            neg_risk = True
        tokens.update(_parse_market_tokens(m))

    if not condition_id:
        return None

    return {
        "slug": event.get("slug", slug),
        "condition_id": condition_id,
        "tokens": tokens,
        "question": question,
        "neg_risk": neg_risk,
    }


def fetch_all_hourly_slugs(base_slug: str) -> List[dict]:
    """
    For hourly BTC markets, try all 24 hourly slugs and return all that exist.
    E.g. base_slug='bitcoin-up-or-down-february-28' → tries 1am..12am, 1pm..12pm.
    Returns list of market dicts (same format as fetch_market_by_slug).
    """
    import requests as _req

    results: List[dict] = []
    for hour in range(1, 13):
        for ampm in ["am", "pm"]:
            candidate_slug = f"{base_slug}-{hour}{ampm}-et"
            try:
                resp = _req.get(f"{GAMMA_API}/events/slug/{candidate_slug}", timeout=10)
                if resp.status_code != 200:
                    continue
                event = resp.json()
            except Exception:
                continue

            raw_markets = event.get("markets", [])
            if not raw_markets:
                continue

            condition_id = None
            question = candidate_slug
            tokens: Dict[str, str] = {}
            neg_risk = False

            for m in raw_markets:
                if isinstance(m, str):
                    try:
                        m = json.loads(m)
                    except Exception:
                        continue
                if not isinstance(m, dict):
                    continue
                cid = m.get("condition_id") or m.get("conditionId")
                if cid:
                    condition_id = str(cid)
                question = m.get("question", question)
                nr = m.get("neg_risk")
                if nr and str(nr).lower() not in ("false", "0", "none", ""):
                    neg_risk = True
                tokens.update(_parse_market_tokens(m))

            if condition_id:
                results.append({
                    "slug": candidate_slug,
                    "condition_id": condition_id,
                    "tokens": tokens,
                    "question": question,
                    "neg_risk": neg_risk,
                })

    return results


def fetch_markets_from_api(limit: int = 200) -> List[dict]:
    """
    Query Polymarket Gamma API for markets and extract condition IDs + token IDs.
    Returns list of {condition_id, slug, question, tokens: {tid: outcome}, neg_risk} dicts.
    """
    import requests as _req

    all_markets: List[dict] = []
    seen_cids: set = set()

    queries = [
        (f"{GAMMA_API}/events", {"limit": limit}),
        (f"{GAMMA_API}/events", {"limit": limit, "closed": "true"}),
        (f"{GAMMA_API}/markets", {"limit": min(limit, 100)}),
        (f"{GAMMA_API}/markets", {"limit": min(limit, 100), "closed": "true"}),
    ]
    for offset in [limit, limit * 2]:
        queries.append(
            (f"{GAMMA_API}/events", {"limit": limit, "closed": "true", "offset": offset})
        )

    for url, params in queries:
        try:
            resp = _req.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception:
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            raw_markets = item.get("markets", [item]) if isinstance(item, dict) else [item]
            slug = item.get("slug", "") if isinstance(item, dict) else ""
            question = item.get("question", slug) if isinstance(item, dict) else ""

            for m in raw_markets:
                if isinstance(m, str):
                    try:
                        m = json.loads(m)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if not isinstance(m, dict):
                    continue
                cid = m.get("condition_id") or m.get("conditionId")
                if not cid:
                    continue
                cid = str(cid)
                if cid not in seen_cids:
                    seen_cids.add(cid)
                    mslug = m.get("slug") or slug
                    mq = m.get("question") or question or mslug
                    tokens = _parse_market_tokens(m)
                    nr = m.get("neg_risk")
                    neg_risk = bool(nr and str(nr).lower() not in ("false", "0", "none", ""))
                    all_markets.append({
                        "condition_id": cid,
                        "slug": mslug,
                        "question": mq,
                        "tokens": tokens,
                        "neg_risk": neg_risk,
                    })

    return all_markets


# ====================================================================
# On-chain balance helpers
# ====================================================================

def _batch_balance_of(
    rpc_url: str,
    contract_addr: str,
    wallet_addr: str,
    token_ids: List[int],
    batch_size: int = 100,
) -> List[int]:
    """
    Batch JSON-RPC eth_call for ERC-1155 balanceOf(wallet, tokenId).
    Works for any ERC-1155 contract (CTF or NegRiskAdapter).
    Returns list of balances in the same order as token_ids.
    """
    import requests as _req

    selector = Web3.keccak(text="balanceOf(address,uint256)")[:4].hex()
    wallet_padded = wallet_addr.lower()[2:].zfill(64)

    results: List[int] = [0] * len(token_ids)

    for chunk_start in range(0, len(token_ids), batch_size):
        chunk = token_ids[chunk_start : chunk_start + batch_size]
        batch = []
        for i, tid in enumerate(chunk):
            tid_padded = hex(tid)[2:].zfill(64)
            call_data = "0x" + selector + wallet_padded + tid_padded
            batch.append({
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": contract_addr, "data": call_data}, "latest"],
                "id": chunk_start + i,
            })

        try:
            resp = _req.post(rpc_url, json=batch, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, dict) and "error" in body:
                log.warning("  Batch RPC error: %s", body["error"])
                continue
            for item in body:
                idx = item.get("id")
                raw = item.get("result", "0x0")
                if idx is not None and raw and raw != "0x":
                    try:
                        results[idx] = int(raw, 16)
                    except ValueError:
                        pass
        except Exception as e:
            log.warning("  Batch RPC failed at offset %d: %s", chunk_start, e)

    return results


def _compute_collection_id(condition_id_hex: str, index_set: int) -> bytes:
    """Local (off-chain) CTF collectionId with parentCollectionId=0."""
    parent = b"\x00" * 32
    cid_bytes = bytes.fromhex(condition_id_hex.replace("0x", ""))
    idx_bytes = index_set.to_bytes(32, "big")
    return Web3.keccak(parent + cid_bytes + idx_bytes)


def _compute_position_id(collection_id: bytes) -> int:
    """Local (off-chain) CTF positionId."""
    token_bytes = bytes.fromhex(USDC_E[2:])
    return int.from_bytes(Web3.keccak(token_bytes + collection_id), "big")


def check_balances_for_market(
    rpc_url: str,
    wallet_addr: str,
    market: dict,
    ctf_addr: str,
) -> Dict[str, Dict[str, int]]:
    """
    Check wallet balances for a market's tokens on multiple contracts.
    Uses the API-provided clobTokenIds directly (works for BOTH regular and NegRisk).
    Also tries computed position IDs with parentCollectionId=0 as fallback.

    Returns: {
        "ctf": {"<outcome>": balance, ...},
        "ctf_computed": {"YES": balance, "NO": balance},
        "neg_risk_adapter": {"<outcome>": balance, ...},
        "neg_risk_exchange": {"<outcome>": balance, ...},
    }
    """
    wallet_cs = Web3.to_checksum_address(wallet_addr)
    tokens = market.get("tokens", {})
    condition_id = market.get("condition_id", "")
    result: Dict[str, Dict[str, int]] = {}

    # --- Method 1: API-provided token IDs on CTF ---
    if tokens:
        tids = list(tokens.keys())
        outcomes = [tokens[t] for t in tids]
        tid_ints = [int(t) for t in tids]

        # Check on CTF contract
        bals = _batch_balance_of(rpc_url, ctf_addr, wallet_cs, tid_ints)
        result["ctf"] = {outcomes[i]: bals[i] for i in range(len(outcomes))}

        # Check on NegRisk Adapter
        bals_nra = _batch_balance_of(rpc_url, NEG_RISK_ADAPTER, wallet_cs, tid_ints)
        result["neg_risk_adapter"] = {outcomes[i]: bals_nra[i] for i in range(len(outcomes))}

        # Check on NegRisk CTF Exchange
        bals_nrx = _batch_balance_of(rpc_url, NEG_RISK_CTF_EXCHANGE, wallet_cs, tid_ints)
        result["neg_risk_exchange"] = {outcomes[i]: bals_nrx[i] for i in range(len(outcomes))}

    # --- Method 2: Computed position IDs (parentCollectionId=0) ---
    if _BYTES32_RE.match(condition_id):
        col_yes = _compute_collection_id(condition_id, 1)
        col_no = _compute_collection_id(condition_id, 2)
        pos_yes = _compute_position_id(col_yes)
        pos_no = _compute_position_id(col_no)
        bals_computed = _batch_balance_of(rpc_url, ctf_addr, wallet_cs, [pos_yes, pos_no])
        result["ctf_computed"] = {"YES": bals_computed[0], "NO": bals_computed[1]}

    return result


def discover_condition_ids_from_wallet(
    w3: Web3,
    ctf,
    wallet_addr: str,
    rpc_url: str,
) -> List[dict]:
    """
    Discover markets with non-zero wallet balance.
    Uses API-provided token IDs (works for both regular and NegRisk markets).

    Returns list of market dicts with non-zero balances.
    """
    log.info("Discovering positions from wallet...")
    wallet_cs = Web3.to_checksum_address(wallet_addr)

    log.info("  Fetching markets from Polymarket API...")
    markets = fetch_markets_from_api(limit=200)
    log.info("  Found %d unique condition IDs from API", len(markets))

    if not markets:
        return []

    # Separate markets with and without token_ids
    markets_with_tokens = [m for m in markets if m.get("tokens")]
    markets_without_tokens = [m for m in markets if not m.get("tokens") and _BYTES32_RE.match(m["condition_id"])]

    discovered: List[dict] = []

    # --- Check markets using API token IDs (handles NegRisk correctly) ---
    if markets_with_tokens:
        log.info("  Checking %d markets using API token IDs...", len(markets_with_tokens))
        # Build flat list of all token IDs
        all_tids: List[int] = []
        tid_map: List[Tuple[int, str, dict]] = []  # (tid_int, outcome, market)

        for m in markets_with_tokens:
            for tid_str, outcome in m["tokens"].items():
                tid_int = int(tid_str)
                all_tids.append(tid_int)
                tid_map.append((tid_int, outcome, m))

        # Batch check on CTF
        log.info("  Batch balanceOf on CTF for %d tokens...", len(all_tids))
        bals_ctf = _batch_balance_of(rpc_url, ctf.address, wallet_cs, all_tids)

        # Batch check on NegRisk Adapter
        log.info("  Batch balanceOf on NegRisk Adapter for %d tokens...", len(all_tids))
        bals_nra = _batch_balance_of(rpc_url, NEG_RISK_ADAPTER, wallet_cs, all_tids)

        # Collect results by condition_id
        seen_cids: set = set()
        for i, (tid_int, outcome, m) in enumerate(tid_map):
            bal_ctf = bals_ctf[i]
            bal_nra = bals_nra[i]
            total = bal_ctf + bal_nra
            if total > 0 and m["condition_id"] not in seen_cids:
                seen_cids.add(m["condition_id"])
                discovered.append(m)
                contract_name = "CTF" if bal_ctf > 0 else "NegRiskAdapter"
                bal_human = total / (10 ** COLLATERAL_DECIMALS)
                log.info("  + %s (%s=%d [%s], %.4f USDC) - %s%s",
                         m["condition_id"][:18] + "...",
                         outcome, total, contract_name, bal_human,
                         m.get("slug", ""),
                         " [NegRisk]" if m.get("neg_risk") else "")
            elif total > 0:
                # Same condition_id, different outcome
                contract_name = "CTF" if bal_ctf > 0 else "NegRiskAdapter"
                log.info("    %s=%d [%s]", outcome, total, contract_name)

    # --- Fallback: computed position IDs for markets without token data ---
    if markets_without_tokens:
        log.info("  Also checking %d markets using computed position IDs (parentCollectionId=0)...",
                 len(markets_without_tokens))
        all_pos: List[int] = []
        for m in markets_without_tokens:
            cid = m["condition_id"]
            col_yes = _compute_collection_id(cid, 1)
            col_no = _compute_collection_id(cid, 2)
            all_pos.append(_compute_position_id(col_yes))
            all_pos.append(_compute_position_id(col_no))

        bals = _batch_balance_of(rpc_url, ctf.address, wallet_cs, all_pos)
        for i, m in enumerate(markets_without_tokens):
            bal_yes = bals[i * 2]
            bal_no = bals[i * 2 + 1]
            if (bal_yes > 0 or bal_no > 0) and m["condition_id"] not in {d["condition_id"] for d in discovered}:
                discovered.append(m)
                log.info("  + %s (YES=%d NO=%d) - %s",
                         m["condition_id"][:18] + "...", bal_yes, bal_no,
                         m.get("slug", ""))

    log.info("  Discovered %d market(s) with non-zero balances", len(discovered))
    return discovered


# ====================================================================
# Proxy wallet (Gnosis Safe) transaction helper
# ====================================================================

def _build_safe_exec_tx(
    w3: Web3,
    safe_addr: str,
    eoa_addr: str,
    private_key: str,
    target: str,
    call_data: bytes,
    chain_id: int,
) -> Optional[dict]:
    """
    Build a Gnosis Safe execTransaction call from the EOA.
    Uses the 'pre-approved' signature format (v=1) since the EOA is the msg.sender and Safe owner.
    """
    safe = w3.eth.contract(address=Web3.to_checksum_address(safe_addr), abi=SAFE_EXEC_ABI)
    zero_addr = "0x" + "00" * 20

    # Pre-approved signature: r=owner_address (padded), s=0, v=1
    owner_bytes = bytes.fromhex(eoa_addr[2:].lower())
    signature = b"\x00" * 12 + owner_bytes + b"\x00" * 32 + b"\x01"  # 65 bytes

    try:
        safe_nonce = safe.functions.nonce().call()
    except Exception:
        log.warning("  Could not read Safe nonce - proxy may not be a Gnosis Safe")
        return None

    try:
        tx = safe.functions.execTransaction(
            Web3.to_checksum_address(target),  # to
            0,                                  # value
            call_data,                          # data
            0,                                  # operation (CALL)
            0,                                  # safeTxGas
            0,                                  # baseGas
            0,                                  # gasPrice
            zero_addr,                          # gasToken
            zero_addr,                          # refundReceiver
            signature,                          # signatures
        ).build_transaction({
            "from": Web3.to_checksum_address(eoa_addr),
            "nonce": w3.eth.get_transaction_count(eoa_addr),
            "chainId": chain_id,
        })
        return tx
    except Exception as e:
        log.warning("  Failed to build Safe execTransaction: %s", e)
        return None


# ====================================================================
# Core redemption logic
# ====================================================================

def redeem_cycle(
    w3: Web3,
    ctf,
    wallet_addr: str,
    eoa_addr: str,
    private_key: str,
    markets: List[dict],
    chain_id: int,
    rpc_url: str,
    dry_run: bool,
) -> int:
    """
    Check each market for resolution and non-zero balance, then redeem.
    Handles both regular CTF and NegRisk markets.
    Returns the number of successful redemptions.
    """
    redeemed = 0
    is_proxy = wallet_addr.lower() != eoa_addr.lower()

    for market in markets:
        condition_id = market["condition_id"]
        neg_risk = market.get("neg_risk", False)
        slug = market.get("slug", "")
        tokens = market.get("tokens", {})

        log.info("--- conditionId: %s ---", condition_id)
        if slug:
            log.info("  Slug: %s", slug)
        if neg_risk:
            log.info("  Type: NegRisk market")

        # 1) Check if resolved
        try:
            denom = ctf.functions.payoutDenominator(condition_id).call()
        except Exception as e:
            log.warning("  Failed to read payoutDenominator: %s", e)
            continue

        if denom == 0:
            log.info("  Not resolved (payoutDenominator = 0), skipping.")
            continue
        log.info("  Resolved (denominator = %d)", denom)

        # Read payout numerators
        try:
            p0 = ctf.functions.payoutNumerators(condition_id, 0).call()
            p1 = ctf.functions.payoutNumerators(condition_id, 1).call()
            log.info("  Payouts: [%d, %d]", p0, p1)
        except ContractLogicError:
            log.info("  (payoutNumerators read reverted)")

        # 2) Check balances using multiple methods
        log.info("  Checking balances...")
        bal_info = check_balances_for_market(rpc_url, wallet_addr, market, ctf.address)

        # Determine where the tokens are and total balance
        # Priority: ctf > ctf_computed > neg_risk_adapter > neg_risk_exchange
        total_balance = 0
        token_location = None
        priority_order = ["ctf", "ctf_computed", "neg_risk_adapter", "neg_risk_exchange"]

        for loc_name in priority_order:
            loc_bals = bal_info.get(loc_name, {})
            loc_total = sum(loc_bals.values())
            if loc_total > 0:
                if token_location is None:
                    token_location = loc_name
                    total_balance = loc_total
                for outcome, bal in loc_bals.items():
                    if bal > 0:
                        log.info("  %s=%d (%.4f USDC) on %s",
                                 outcome, bal, bal / (10 ** COLLATERAL_DECIMALS), loc_name)

        if total_balance == 0:
            log.info("  No tokens found on any contract, skipping.")
            continue

        log.info("  Tokens found on: %s (total raw=%d, %.4f USDC)",
                 token_location, total_balance, total_balance / (10 ** COLLATERAL_DECIMALS))

        if dry_run:
            log.info("  [DRY RUN] Would redeem -- skipping tx.")
            continue

        # 3) Attempt redemption
        index_sets = [1, 2]
        success = False

        if token_location in ("ctf", "ctf_computed"):
            # Regular CTF redemption with parentCollectionId=0
            success = _attempt_ctf_redeem(
                w3, ctf, wallet_addr, eoa_addr, private_key,
                condition_id, index_sets, chain_id, is_proxy,
            )
        elif token_location in ("neg_risk_adapter", "neg_risk_exchange"):
            # NegRisk: try CTF redeem first (in case it works), then proxy route
            log.info("  NegRisk market: attempting redemption through CTF...")
            success = _attempt_ctf_redeem(
                w3, ctf, wallet_addr, eoa_addr, private_key,
                condition_id, index_sets, chain_id, is_proxy,
            )
            if not success:
                log.info("  CTF direct redemption failed. Trying via proxy Safe...")
                target_contract = NEG_RISK_ADAPTER if token_location == "neg_risk_adapter" else ctf.address
                success = _attempt_proxy_redeem(
                    w3, wallet_addr, eoa_addr, private_key,
                    target_contract, condition_id, index_sets, chain_id,
                )

        if success:
            redeemed += 1

    return redeemed


def _attempt_ctf_redeem(
    w3: Web3,
    ctf,
    wallet_addr: str,
    eoa_addr: str,
    private_key: str,
    condition_id: str,
    index_sets: List[int],
    chain_id: int,
    is_proxy: bool,
) -> bool:
    """Attempt to redeem via CTF contract. If proxy, routes through Safe."""
    try:
        if is_proxy:
            # Route through Gnosis Safe proxy
            call_data = ctf.encode_abi(
                abi_element_identifier="redeemPositions",
                args=[USDC_E, ZERO_BYTES32_RAW, bytes.fromhex(condition_id.replace("0x", "")), index_sets],
            )
            tx = _build_safe_exec_tx(
                w3, wallet_addr, eoa_addr, private_key,
                ctf.address, bytes.fromhex(call_data[2:]), chain_id,
            )
            if tx is None:
                log.warning("  Could not build Safe tx")
                return False
            tx_from = eoa_addr
        else:
            # Direct call from EOA
            nonce = w3.eth.get_transaction_count(wallet_addr)
            tx = ctf.functions.redeemPositions(
                USDC_E, ZERO_BYTES32, condition_id, index_sets
            ).build_transaction({
                "from": wallet_addr,
                "nonce": nonce,
                "chainId": chain_id,
            })
            tx_from = wallet_addr

        # Gas estimation
        try:
            gas_est = w3.eth.estimate_gas(tx)
            tx["gas"] = int(gas_est * 1.25)
        except Exception as e:
            log.warning("  Gas estimation failed (%s), using default 500k", e)
            tx["gas"] = 500_000

        # EIP-1559 fee params
        try:
            latest = w3.eth.get_block("latest")
            base_fee = latest.get("baseFeePerGas", None)
            if base_fee is not None:
                priority = w3.to_wei(40, "gwei")
                tx["maxPriorityFeePerGas"] = priority
                tx["maxFeePerGas"] = int(base_fee * 2 + priority)
            else:
                tx["gasPrice"] = w3.eth.gas_price
        except Exception:
            tx["gasPrice"] = w3.eth.gas_price

        log.info("  Sending redeemPositions tx (from %s)...", tx_from)
        signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        log.info("  tx: %s", tx_hash.hex())

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        if receipt.status == 1:
            log.info("  Confirmed in block %d (status=1 OK)", receipt.blockNumber)
            return True
        else:
            log.warning("  Tx reverted in block %d (status=0)", receipt.blockNumber)
            return False

    except Exception as e:
        log.error("  CTF redemption failed: %s", e)
        return False


def _attempt_proxy_redeem(
    w3: Web3,
    safe_addr: str,
    eoa_addr: str,
    private_key: str,
    target_contract: str,
    condition_id: str,
    index_sets: List[int],
    chain_id: int,
) -> bool:
    """Attempt redemption by routing the call through the Gnosis Safe proxy."""
    try:
        # Encode the redeemPositions call for the CTF contract
        ctf_temp = w3.eth.contract(
            address=Web3.to_checksum_address(target_contract),
            abi=CTF_ABI,
        )
        call_data = ctf_temp.encode_abi(
            abi_element_identifier="redeemPositions",
            args=[USDC_E, ZERO_BYTES32_RAW, bytes.fromhex(condition_id.replace("0x", "")), index_sets],
        )

        tx = _build_safe_exec_tx(
            w3, safe_addr, eoa_addr, private_key,
            target_contract, bytes.fromhex(call_data[2:]), chain_id,
        )
        if tx is None:
            return False

        # Gas estimation
        try:
            gas_est = w3.eth.estimate_gas(tx)
            tx["gas"] = int(gas_est * 1.25)
        except Exception as e:
            log.warning("  Gas estimation failed (%s), using 600k", e)
            tx["gas"] = 600_000

        # EIP-1559 fee params
        try:
            latest = w3.eth.get_block("latest")
            base_fee = latest.get("baseFeePerGas", None)
            if base_fee is not None:
                priority = w3.to_wei(40, "gwei")
                tx["maxPriorityFeePerGas"] = priority
                tx["maxFeePerGas"] = int(base_fee * 2 + priority)
            else:
                tx["gasPrice"] = w3.eth.gas_price
        except Exception:
            tx["gasPrice"] = w3.eth.gas_price

        log.info("  Sending proxy execTransaction (EOA %s -> Safe %s -> %s)...",
                 eoa_addr[:10], safe_addr[:10], target_contract[:10])
        signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        log.info("  tx: %s", tx_hash.hex())

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        if receipt.status == 1:
            log.info("  Confirmed in block %d (status=1 OK)", receipt.blockNumber)
            return True
        else:
            log.warning("  Tx reverted in block %d (status=0)", receipt.blockNumber)
            return False

    except Exception as e:
        log.error("  Proxy redemption failed: %s", e)
        return False


# ====================================================================
# CLI
# ====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Redeem resolved Polymarket CTF positions (regular + NegRisk) into USDC.e"
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Check balances but do not submit any transactions")
    p.add_argument("--once", action="store_true",
                   help="Run a single redemption cycle and exit (no loop)")
    p.add_argument("--interval-min", type=float, default=30.0,
                   help="Minutes between redemption cycles (default: 30)")
    p.add_argument("--wallet", type=str, default="",
                   help="Polymarket proxy wallet address. Overrides POLYMARKET_PROXY env var.")
    p.add_argument("--slug", type=str, default="",
                   help="Search for a specific market by slug "
                        "(e.g. bitcoin-up-or-down-february-28-11am-et). "
                        "Skips broad auto-discovery and targets this market directly.")
    return p.parse_args()


# ====================================================================
# Main
# ====================================================================

def main() -> int:
    load_env()
    args = parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ---- Read env ----
    private_key = must_env("POLYMARKET_PRIVATE_KEY")
    ctf_addr_str = os.getenv("CTF_ADDRESS", "").strip() or DEFAULT_CTF_ADDRESS
    ctf_address = Web3.to_checksum_address(ctf_addr_str)

    # Condition IDs: use env var if provided
    condition_ids_raw = os.getenv("CONDITION_IDS", "").strip()
    condition_ids = parse_condition_ids(condition_ids_raw) if condition_ids_raw else None

    # ---- Connect ----
    explicit_rpc = os.getenv("POLYGON_RPC_URL", "").strip() or os.getenv("RPC_URL", "").strip()
    rpc_urls_to_try = ([explicit_rpc] if explicit_rpc else []) + FALLBACK_RPC_URLS

    w3 = None
    connected_rpc = None
    for rpc_url in rpc_urls_to_try:
        try:
            candidate = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
            if candidate.is_connected():
                w3 = candidate
                connected_rpc = rpc_url
                break
        except Exception:
            pass

    if w3 is None:
        log.error("Could not connect to any Polygon RPC. Tried: %s", rpc_urls_to_try)
        return 1
    log.info("Connected to RPC: %s", connected_rpc)

    acct = w3.eth.account.from_key(private_key)
    eoa_addr = acct.address

    # Resolve wallet address: CLI > POLYMARKET_PROXY env > WALLET_ADDR env > EOA fallback
    wallet_addr = (
        args.wallet.strip()
        or os.getenv("POLYMARKET_PROXY", "").strip()
        or os.getenv("WALLET_ADDR", "").strip()
        or eoa_addr
    )
    wallet_addr = Web3.to_checksum_address(wallet_addr)
    chain_id = w3.eth.chain_id

    if wallet_addr.lower() == eoa_addr.lower():
        log.warning(
            "Using EOA address %s as wallet. "
            "NOTE: Polymarket uses PROXY wallets -- your CTF tokens are likely "
            "held by your proxy wallet, NOT your EOA. "
            "Set POLYMARKET_PROXY in your .env or pass --wallet.",
            wallet_addr,
        )
    else:
        log.info("Wallet (proxy): %s  (EOA signer: %s)", wallet_addr, eoa_addr)

    ctf = w3.eth.contract(address=ctf_address, abi=CTF_ABI)

    # ---- Resolve markets to redeem ----
    markets_to_redeem: List[dict] = []

    if args.slug:
        # Targeted slug search
        slug = args.slug.strip()
        log.info("Searching for market by slug: %s", slug)
        market = fetch_market_by_slug(slug)

        # If partial slug (no time suffix), scan all 24 hourly variations
        # and find which ones have tokens
        slug_lower = slug.lower()
        if market is None and "am-et" not in slug_lower and "pm-et" not in slug_lower:
            log.info("Scanning all hourly variations of '%s' and checking balances...", slug)
            all_hourly = fetch_all_hourly_slugs(slug)
            if all_hourly:
                log.info("Found %d hourly markets, checking balances on each...", len(all_hourly))
                markets_with_tokens = []
                for hm in all_hourly:
                    bal_info = check_balances_for_market(connected_rpc, wallet_addr, hm, ctf_address)
                    has_balance = any(
                        bal > 0
                        for loc_bals in bal_info.values()
                        for bal in loc_bals.values()
                    )
                    if has_balance:
                        markets_with_tokens.append(hm)
                        log.info("  ✓ %s — has tokens!", hm["slug"])
                        for loc_name, loc_bals in bal_info.items():
                            for outcome, bal in loc_bals.items():
                                if bal > 0:
                                    log.info("    %s=%d (%.4f USDC) on %s",
                                             outcome, bal, bal / (10 ** COLLATERAL_DECIMALS), loc_name)
                    else:
                        log.debug("  ✗ %s — no tokens", hm["slug"])

                if markets_with_tokens:
                    markets_to_redeem = markets_with_tokens
                    log.info("Found %d market(s) with tokens to redeem", len(markets_with_tokens))
                else:
                    log.warning("Scanned %d hourly markets for '%s' — no tokens found on any.",
                                len(all_hourly), slug)
                    return 0
            else:
                log.error("Could not find any hourly markets matching: %s", slug)
                return 1
        elif market is None:
            log.error("Could not find market for slug: %s", slug)
            log.info("TIP: Try a partial slug like 'bitcoin-up-or-down-february-28'")
            return 1
        else:
            # Exact slug found
            log.info("Found market: %s", market["question"])
            log.info("  Condition ID: %s", market["condition_id"])
            log.info("  NegRisk: %s", market.get("neg_risk", False))
            for tid, outcome in market.get("tokens", {}).items():
                log.info("  %s token_id: %s", outcome, tid[:20] + "..." if len(tid) > 20 else tid)

            # Check balances
            log.info("Checking on-chain balances for proxy %s...", wallet_addr)
            bal_info = check_balances_for_market(connected_rpc, wallet_addr, market, ctf_address)

            found_any = False
            for loc_name, loc_bals in bal_info.items():
                for outcome, bal in loc_bals.items():
                    if bal > 0:
                        found_any = True
                        log.info("  %s=%d (%.6f USDC) on %s",
                                 outcome, bal, bal / (10 ** COLLATERAL_DECIMALS), loc_name)

            if not found_any:
                log.warning("No tokens found for this market on wallet %s", wallet_addr)
                log.info("Checked contracts:")
                log.info("  CTF (API token IDs):          %s", ctf_address)
                log.info("  CTF (computed, parent=0):      %s", ctf_address)
                log.info("  NegRisk Adapter:               %s", NEG_RISK_ADAPTER)
                log.info("  NegRisk CTF Exchange:          %s", NEG_RISK_CTF_EXCHANGE)

                # Also check EOA
                if wallet_addr.lower() != eoa_addr.lower():
                    log.info("Also checking EOA %s...", eoa_addr)
                    bal_eoa = check_balances_for_market(connected_rpc, eoa_addr, market, ctf_address)
                    eoa_found = False
                    for loc_name, loc_bals in bal_eoa.items():
                        for outcome, bal in loc_bals.items():
                            if bal > 0:
                                eoa_found = True
                                log.info("  EOA %s=%d on %s", outcome, bal, loc_name)
                    if not eoa_found:
                        log.info("  No tokens on EOA either.")
                        log.info("  Tokens may already be redeemed, or held on a different address.")
                return 0

            markets_to_redeem = [market]

    elif condition_ids is not None:
        # Use explicit condition IDs from env var
        log.info("Using %d condition ID(s) from CONDITION_IDS env var", len(condition_ids))
        for cid in condition_ids:
            markets_to_redeem.append({
                "condition_id": cid,
                "tokens": {},
                "slug": "",
                "question": "",
                "neg_risk": False,
            })

    else:
        # Auto-discover from wallet
        log.info("Auto-discovering positions from wallet...")
        discovered = discover_condition_ids_from_wallet(w3, ctf, wallet_addr, connected_rpc)
        if not discovered:
            log.warning("No positions discovered with non-zero balances.")
            if wallet_addr.lower() == eoa_addr.lower():
                log.warning(
                    "HINT: You are scanning the EOA. Set POLYMARKET_PROXY in .env or try:\n"
                    "  python redeem_ctf.py --wallet 0xYOUR_PROXY --dry-run --once"
                )
            log.info("TIP: If you know the market slug, use --slug to target it directly:\n"
                     "  python redeem_ctf.py --slug bitcoin-up-or-down-february-28-11am-et --dry-run --once")
            return 0
        markets_to_redeem = discovered

    # ---- Print summary ----
    mode = "DRY RUN" if args.dry_run else "LIVE"
    log.info("=" * 60)
    log.info("  CTF Redeemer  --  %s", mode)
    log.info("  Wallet:       %s", wallet_addr)
    log.info("  EOA signer:   %s", eoa_addr)
    log.info("  CTF contract: %s", ctf_address)
    log.info("  Chain ID:     %d", chain_id)
    log.info("  Markets:      %d", len(markets_to_redeem))
    if args.once:
        log.info("  Mode:         single pass")
    else:
        log.info("  Mode:         loop every %.0f min", args.interval_min)
    log.info("=" * 60)

    interval_sec = args.interval_min * 60.0
    cycle = 0

    while True:
        cycle += 1
        log.info("[cycle %d] Starting redemption check (%d market(s))...",
                 cycle, len(markets_to_redeem))

        try:
            n = redeem_cycle(
                w3=w3, ctf=ctf,
                wallet_addr=wallet_addr,
                eoa_addr=eoa_addr,
                private_key=private_key,
                markets=markets_to_redeem,
                chain_id=chain_id,
                rpc_url=connected_rpc,
                dry_run=args.dry_run,
            )
            log.info("[cycle %d] Done. Redeemed %d market(s).", cycle, n)
        except Exception:
            log.exception("[cycle %d] Unhandled error in redemption cycle", cycle)

        if args.once or _SHUTDOWN:
            break

        log.info("[cycle %d] Next check in %.0f min...", cycle, args.interval_min)
        wake_time = time.monotonic() + interval_sec
        while time.monotonic() < wake_time:
            if _SHUTDOWN:
                break
            time.sleep(min(5.0, wake_time - time.monotonic()))

        if _SHUTDOWN:
            break

    log.info("Exiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
