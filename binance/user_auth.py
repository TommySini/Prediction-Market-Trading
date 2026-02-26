"""
Polymarket CLOB authentication and trading client.

Handles wallet connection, API credential management, balance/allowance
checks, and exposes a ready-to-trade ClobClient instance.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    AssetType,
    BalanceAllowanceParams,
    OrderArgs,
    OrderType,
)
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon Mainnet


def _load_private_key() -> str:
    """Load the Ethereum private key from the POLYMARKET_PRIVATE_KEY env var."""
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
    if pk:
        return pk

    log.error("No private key found. Set POLYMARKET_PRIVATE_KEY in binance/.env")
    sys.exit(1)


def _resolve_signature_type() -> int:
    """0 = EOA / MetaMask, 1 = email / Magic / proxy wallet."""
    raw = os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")
    return int(raw)


def _resolve_funder() -> str | None:
    return os.getenv("POLYMARKET_FUNDER_ADDRESS") or None


def build_client() -> ClobClient:
    """
    Build and return a fully authenticated ClobClient ready for trading.

    Steps performed:
      1. Load private key (env var or PEM file).
      2. Instantiate ClobClient with appropriate signature type.
      3. Create or derive API credentials and attach them.
    """
    private_key = _load_private_key()
    sig_type = _resolve_signature_type()
    funder = _resolve_funder()

    kwargs: dict = {
        "host": HOST,
        "key": private_key,
        "chain_id": CHAIN_ID,
    }
    if sig_type != 0:
        kwargs["signature_type"] = sig_type
    if funder:
        kwargs["funder"] = funder

    client = ClobClient(**kwargs)

    log.info("Deriving / creating API credentials …")
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    log.info("API key set: %s…", creds.api_key[:12])

    return client


# ── Balance & allowance helpers ──────────────────────────────────────


def get_usdc_balance(client: ClobClient) -> dict:
    params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    return client.get_balance_allowance(params)


def get_token_balance(client: ClobClient, token_id: str) -> dict:
    params = BalanceAllowanceParams(
        asset_type=AssetType.CONDITIONAL,
        token_id=token_id,
    )
    return client.get_balance_allowance(params)


def approve_allowances(client: ClobClient) -> None:
    """Set max allowance for USDC collateral so orders can execute."""
    params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    result = client.update_balance_allowance(params)
    log.info("Collateral allowance updated: %s", result)


# ── Contract address helpers ─────────────────────────────────────────


def get_contract_addresses(client: ClobClient) -> dict[str, str]:
    return {
        "collateral_usdc": client.get_collateral_address(),
        "conditional_tokens": client.get_conditional_address(),
        "exchange": client.get_exchange_address(),
    }


# ── API key management ───────────────────────────────────────────────


def list_api_keys(client: ClobClient):
    return client.get_api_keys()


def rotate_api_key(client: ClobClient) -> None:
    """Delete current key and create a fresh one."""
    client.delete_api_key()
    creds = client.create_api_key()
    client.set_api_creds(creds)
    log.info("Rotated to new API key: %s…", creds.api_key[:12])


# ── Quick-trade helpers ──────────────────────────────────────────────


def place_limit_order(
    client: ClobClient,
    token_id: str,
    side: str,
    price: float,
    size: float,
    order_type: OrderType = OrderType.GTC,
) -> dict:
    """
    Place a limit order on Polymarket.

    Args:
        token_id: The condition token ID for the market outcome.
        side: "BUY" or "SELL".
        price: Limit price (0–1 range for binary markets).
        size: Number of shares.
        order_type: GTC (default), FOK, or GTD.

    Returns:
        API response dict from post_order.
    """
    order_side = BUY if side.upper() == "BUY" else SELL
    order_args = OrderArgs(
        token_id=token_id,
        price=price,
        size=size,
        side=order_side,
    )
    signed_order = client.create_order(order_args)
    resp = client.post_order(signed_order, order_type)
    log.info("Order placed: %s", resp)
    return resp


# ── CLI smoke-test ───────────────────────────────────────────────────


def main() -> None:
    """Quick connectivity and balance check."""
    client = build_client()

    log.info("=== Contract Addresses ===")
    for name, addr in get_contract_addresses(client).items():
        log.info("  %s: %s", name, addr)

    log.info("=== USDC Balance ===")
    log.info("  %s", get_usdc_balance(client))

    log.info("=== API Keys ===")
    log.info("  %s", list_api_keys(client))

    log.info("Auth OK — client is ready for trading.")


if __name__ == "__main__":
    main()
