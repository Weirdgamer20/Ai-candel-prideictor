"""CoinDCX active spot-market discovery; no assumptions about BTC-only symbols."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import requests

MARKETS_URL = "https://api.coindcx.com/exchange/v1/markets_details"


@dataclass(frozen=True)
class SpotMarket:
    pair: str
    base_asset: str
    quote_asset: str


def discover_spot_markets(
    quote_currencies: Iterable[str] | None = ("INR",),
    *,
    session: requests.Session | None = None,
) -> list[SpotMarket]:
    """Return active market pairs from CoinDCX metadata, optionally filtered by quote.

    CoinDCX's metadata uses target_currency_short_name for the traded asset and
    base_currency_short_name for the quote currency in pairs such as BTC/INR.
    The exchange response is the source of truth; this function invents no symbols.
    """
    client = session or requests.Session()
    response = client.get(MARKETS_URL, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(
            f"Unexpected CoinDCX markets metadata: {type(payload).__name__}"
        )

    allowed_quotes = (
        {str(value).strip().upper() for value in quote_currencies}
        if quote_currencies is not None
        else None
    )
    markets: dict[str, SpotMarket] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        pair = str(row.get("pair", "")).strip()
        asset = str(row.get("target_currency_short_name", "")).strip().upper()
        quote = str(row.get("base_currency_short_name", "")).strip().upper()
        status = str(row.get("status", "active")).strip().lower()
        if not pair or not asset or not quote or status != "active":
            continue
        if allowed_quotes is not None and quote not in allowed_quotes:
            continue
        markets[pair] = SpotMarket(pair=pair, base_asset=asset, quote_asset=quote)
    return sorted(markets.values(), key=lambda market: (market.quote_asset, market.base_asset, market.pair))
