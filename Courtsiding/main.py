from cryptography.hazmat.primitives import serialization
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from streaming import run_stream

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
KEYID = os.getenv("KALSHI_KEY_ID")
PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH", "private_key.pem")

if not KEYID:
    raise ValueError("KALSHI_KEY_ID not found in environment variables")

if not os.path.exists(PRIVATE_KEY_PATH):
    raise FileNotFoundError(f"Private key file not found: {PRIVATE_KEY_PATH}")

# Load private key from .pem file
with open(PRIVATE_KEY_PATH, "rb") as key_file:
    private_key_pem = key_file.read()

private_key = serialization.load_pem_private_key(
    private_key_pem,
    password=None,
)

if __name__ == "__main__":
    import asyncio
    
    # Get market ticker from env or prompt
    market_ticker = os.getenv("KALSHI_MARKET_TICKER")
    if not market_ticker:
        market_ticker = input("Enter market ticker: ").strip()
        if not market_ticker:
            print("Error: Market ticker cannot be empty")
            sys.exit(1)
    
    sys.stdout.write(f"[STARTING] Streaming {market_ticker}...\n")
    sys.stdout.flush()
    
    try:
        # Run unbuffered if you want the fastest display:
        # python -u main.py
        asyncio.run(run_stream(KEYID, private_key, market_ticker))
    except KeyboardInterrupt:
        print("\nStreaming stopped by user")
    except Exception as e:
        print(f"Error: {e}")