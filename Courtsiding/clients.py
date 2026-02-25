import requests
import base64
import time
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import json

from requests.exceptions import HTTPError

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature

import ssl
import websockets
import certifi

class KalshiBaseClient:
    """Base client class for interacting with the Kalshi API."""
    def __init__(
        self,
        key_id: str,
        private_key: rsa.RSAPrivateKey,
    ):
        """Initializes the client with the provided API key and private key.

        Args:
            key_id (str): Your Kalshi API key ID.
            private_key (rsa.RSAPrivateKey): Your RSA private key.
        """
        self.key_id = key_id
        self.private_key = private_key
        self.last_api_call = datetime.now()
        self.HTTP_BASE_URL = "https://api.elections.kalshi.com"
        self.WS_BASE_URL = "wss://api.elections.kalshi.com"

    def request_headers(self, method: str, path: str) -> Dict[str, Any]:
        """Generates the required authentication headers for API requests."""
        current_time_milliseconds = int(time.time() * 1000)
        timestamp_str = str(current_time_milliseconds)

        # Remove query params from path
        path_parts = path.split('?')

        msg_string = timestamp_str + method + path_parts[0]
        signature = self.sign_pss_text(msg_string)

        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
        }
        return headers

    def sign_pss_text(self, text: str) -> str:
        """Signs the text using RSA-PSS and returns the base64 encoded signature."""
        message = text.encode('utf-8')
        try:
            signature = self.private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode('utf-8')
        except InvalidSignature as e:
            raise ValueError("RSA sign PSS failed") from e

class KalshiHttpClient(KalshiBaseClient):
    """Client for handling HTTP connections to the Kalshi API."""
    def __init__(
        self,
        key_id: str,
        private_key: rsa.RSAPrivateKey,
    ):
        super().__init__(key_id, private_key)
        self.host = self.HTTP_BASE_URL
        self.exchange_url = "/trade-api/v2/exchange"
        self.markets_url = "/trade-api/v2/markets"
        self.portfolio_url = "/trade-api/v2/portfolio"

    def rate_limit(self) -> None:
        """Built-in rate limiter to prevent exceeding API rate limits."""
        THRESHOLD_IN_MILLISECONDS = 100
        now = datetime.now()
        threshold_in_microseconds = 1000 * THRESHOLD_IN_MILLISECONDS
        threshold_in_seconds = THRESHOLD_IN_MILLISECONDS / 1000
        if now - self.last_api_call < timedelta(microseconds=threshold_in_microseconds):
            time.sleep(threshold_in_seconds)
        self.last_api_call = datetime.now()

    def raise_if_bad_response(self, response: requests.Response) -> None:
        """Raises an HTTPError if the response status code indicates an error."""
        if response.status_code not in range(200, 299):
            response.raise_for_status()

    def post(self, path: str, body: dict) -> Any:
        """Performs an authenticated POST request to the Kalshi API."""
        self.rate_limit()
        response = requests.post(
            self.host + path,
            json=body,
            headers=self.request_headers("POST", path)
        )
        self.raise_if_bad_response(response)
        return response.json()

    def get(self, path: str, params: Dict[str, Any] = {}) -> Any:
        """Performs an authenticated GET request to the Kalshi API."""
        self.rate_limit()
        full_url = self.host + path
        response = requests.get(
            full_url,
            headers=self.request_headers("GET", path),
            params=params
        )
        self.raise_if_bad_response(response)
        return response.json()

    def delete(self, path: str, params: Dict[str, Any] = {}) -> Any:
        """Performs an authenticated DELETE request to the Kalshi API."""
        self.rate_limit()
        response = requests.delete(
            self.host + path,
            headers=self.request_headers("DELETE", path),
            params=params
        )
        self.raise_if_bad_response(response)
        return response.json()

    def get_balance(self) -> Dict[str, Any]:
        """Retrieves the account balance."""
        return self.get(self.portfolio_url + '/balance')

    def get_exchange_status(self) -> Dict[str, Any]:
        """Retrieves the exchange status."""
        return self.get(self.exchange_url + "/status")

    def get_trades(
        self,
        ticker: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        max_ts: Optional[int] = None,
        min_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieves trades based on provided filters."""
        params = {
            'ticker': ticker,
            'limit': limit,
            'cursor': cursor,
            'max_ts': max_ts,
            'min_ts': min_ts,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return self.get(self.markets_url + '/trades', params=params)

    def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        client_order_id: Optional[str] = None,
        order_type: str = "limit",
    ) -> Dict[str, Any]:
        """Creates a limit order."""
        import uuid
        
        body = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }
        
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        
        return self.post(self.portfolio_url + '/orders', body)

    def get_positions(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Retrieves portfolio positions."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        return self.get(self.portfolio_url + '/positions', params=params)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Retrieves a specific order by ID."""
        return self.get(self.portfolio_url + f'/orders/{order_id}')

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancels a specific order by ID."""
        return self.post(self.portfolio_url + f'/orders/{order_id}/cancel', {})


class KalshiWebSocketClient(KalshiBaseClient):
    """Client for handling WebSocket connections to the Kalshi API."""
    def __init__(
        self,
        key_id: str,
        private_key: rsa.RSAPrivateKey,
    ):
        super().__init__(key_id, private_key)
        self.ws = None
        self.url_suffix = "/trade-api/ws/v2"
        self._msg_id = 1
        self._fill_subscribed = False

    def _next_msg_id(self) -> int:
        """Get next message ID for WebSocket commands."""
        msg_id = self._msg_id
        self._msg_id += 1
        return msg_id

    def _get_ws_auth_headers(self) -> Dict[str, str]:
        """Generate WebSocket authentication headers.
        
        The signed text must be: timestamp + "GET" + "/trade-api/ws/v2"
        """
        current_time_milliseconds = int(time.time() * 1000)
        timestamp_str = str(current_time_milliseconds)
        
        # Sign: timestamp + "GET" + "/trade-api/ws/v2"
        msg_string = timestamp_str + "GET" + self.url_suffix
        signature = self.sign_pss_text(msg_string)
        
        headers = {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
        }
        return headers

    async def subscribe_user_fills(self):
        """Subscribe to the User Fills channel.
        
        Per Kalshi docs: market specification is ignored, always sends all fills.
        Channel name is "fill" (not "user_fills").
        """
        if self.ws is None:
            raise RuntimeError("WebSocket not connected")
        
        cmd = {
            "id": self._next_msg_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["fill"]
            }
        }
        
        await self.ws.send(json.dumps(cmd))
        print(f"[WS] Subscribed to fill channel: {json.dumps(cmd)}")

    async def connect(self):
        """Establishes a WebSocket connection and processes messages continuously."""
        host = self.WS_BASE_URL + self.url_suffix
        auth_headers = self._get_ws_auth_headers()
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        async with websockets.connect(
            host,
            additional_headers=auth_headers,
            ssl=ssl_context,
            ping_interval=20,
            ping_timeout=10
        ) as websocket:
            self.ws = websocket
            await self.on_open()
            
            # Subscribe to fill channel
            await self.subscribe_user_fills()
            
            # Process messages continuously
            try:
                async for message in websocket:
                    await self.process_message(message)
            except websockets.ConnectionClosed as e:
                await self.on_close(e.code, e.reason)
            except Exception as e:
                await self.on_error(e)

    async def process_message(self, message: str):
        """Process incoming WebSocket messages."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            print(f"[WS] Failed to parse message: {e}, message: {message[:200]}")
            return
        
        msg_type = data.get("type")
        
        if msg_type == "fill":
            # User fill notification
            fill_msg = data.get("msg", {})
            await self.on_fill(fill_msg)
        elif msg_type == "error":
            # Error message
            error_data = data.get("msg", {})
            error_code = error_data.get("code")
            error_msg = error_data.get("msg")
            print(f"[WS ERROR] Code: {error_code}, Message: {error_msg}")
            await self.on_error(f"WebSocket error {error_code}: {error_msg}")
        elif msg_type == "subscribed":
            # Subscription confirmation
            sub_msg = data.get("msg", {})
            channel = sub_msg.get("channel")
            sid = sub_msg.get("sid")
            print(f"[WS] Subscribed to channel: {channel}, sid: {sid}")
            if channel == "fill":
                self._fill_subscribed = True
        elif msg_type == "ok":
            # OK response (may contain subscription info)
            subs = data.get("subscriptions", [])
            for sub in subs:
                channel = sub.get("channel")
                sid = sub.get("sid")
                print(f"[WS] Subscription confirmed: {channel}, sid: {sid}")
                if channel == "fill":
                    self._fill_subscribed = True
        else:
            # Other message types - pass to on_message for custom handling
            await self.on_message(data)

    async def on_open(self):
        """Callback when WebSocket connection is opened."""
        print("WebSocket connection opened.")

    async def on_fill(self, fill_msg: Dict[str, Any]):
        """Callback for handling fill notifications.
        
        Args:
            fill_msg: The fill message payload containing:
                - trade_id: str
                - order_id: str
                - market_ticker: str
                - is_taker: bool
                - side: "yes" or "no"
                - yes_price: int (or no_price)
                - count: int
                - action: "buy" or "sell"
                - ts: int (timestamp)
                - post_position: int
        """
        print(f"[WS FILL] {fill_msg.get('action', 'unknown').upper()} {fill_msg.get('side', 'unknown').upper()} "
              f"- {fill_msg.get('count', 0)} shares @ {fill_msg.get('yes_price', fill_msg.get('no_price', 'N/A'))} "
              f"- Order: {fill_msg.get('order_id', 'unknown')}")

    async def on_message(self, message):
        """Callback for handling other incoming messages."""
        # Override this in subclasses for custom message handling
        pass

    async def on_error(self, error):
        """Callback for handling errors."""
        print(f"WebSocket error: {error}")

    async def on_close(self, close_status_code, close_msg):
        """Callback when WebSocket connection is closed."""
        print(f"WebSocket connection closed with code: {close_status_code}, message: {close_msg}")