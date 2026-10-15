from src.universe import discover_spot_markets


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, timeout):
        assert "markets_details" in url
        assert timeout > 0
        return FakeResponse(self.payload)


def test_discovers_all_active_assets_in_requested_quote_currency():
    markets = discover_spot_markets(
        ("INR",),
        session=FakeSession([
            {"pair": "I-BTC_INR", "target_currency_short_name": "BTC",
             "base_currency_short_name": "INR", "status": "active"},
            {"pair": "I-ETH_INR", "target_currency_short_name": "ETH",
             "base_currency_short_name": "INR", "status": "active"},
            {"pair": "I-DOGE_INR", "target_currency_short_name": "DOGE",
             "base_currency_short_name": "INR", "status": "inactive"},
            {"pair": "B-BTC_USDT", "target_currency_short_name": "BTC",
             "base_currency_short_name": "USDT", "status": "active"},
        ]),
    )
    assert [market.base_asset for market in markets] == ["BTC", "ETH"]
    assert all(market.quote_asset == "INR" for market in markets)


def test_can_discover_all_quote_currencies_when_requested():
    markets = discover_spot_markets(
        None,
        session=FakeSession([
            {"pair": "I-BTC_INR", "target_currency_short_name": "BTC",
             "base_currency_short_name": "INR", "status": "active"},
            {"pair": "B-BTC_USDT", "target_currency_short_name": "BTC",
             "base_currency_short_name": "USDT", "status": "active"},
        ]),
    )
    assert {market.quote_asset for market in markets} == {"INR", "USDT"}
