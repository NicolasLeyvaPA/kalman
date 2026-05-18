"""
Curated list of known centralized-exchange hot wallets and Polygon
bridge contracts. Used to classify funding sources during chain tracing.

These are publicly attributed addresses gathered from Etherscan/Polygonscan
labels. The list is intentionally conservative — matches here are
high-confidence. Lower-confidence pattern matching belongs in
exchange_identifier.py.
"""

KNOWN_ADDRESSES: dict[str, dict[str, str]] = {
    # Coinbase
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": {"label": "Coinbase 10",   "category": "exchange:coinbase"},
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": {"label": "Coinbase 1",    "category": "exchange:coinbase"},
    "0x503828976d22510aad0201ac7ec88293211d23da": {"label": "Coinbase 2",    "category": "exchange:coinbase"},
    "0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740": {"label": "Coinbase 3",    "category": "exchange:coinbase"},
    "0x3cd751e6b0078be393132286c442345e5dc49699": {"label": "Coinbase 4",    "category": "exchange:coinbase"},
    "0xb5d85cbf7cb3ee0d56b3bb207d5fc4b82f43f511": {"label": "Coinbase 5",    "category": "exchange:coinbase"},
    "0xeb2629a2734e272bcc07bda959863f316f4bd4cf": {"label": "Coinbase 6",    "category": "exchange:coinbase"},
    "0xd688aea8f7d450909ade10c47faa95707b0682d9": {"label": "Coinbase 7",    "category": "exchange:coinbase"},
    "0x02466e547bfdab679fc49e96bbfc62b9747d997c": {"label": "Coinbase 8",    "category": "exchange:coinbase"},
    "0x6b76f8b1e9e59913bfe758821887311ba1805cab": {"label": "Coinbase 9",    "category": "exchange:coinbase"},

    # Binance
    "0xf977814e90da44bfa03b6295a0616a897441acec": {"label": "Binance 8",     "category": "exchange:binance"},
    "0xe7804c37c13166ff0b37f5ae0bb07a3aebb6e245": {"label": "Binance Hot",   "category": "exchange:binance"},
    "0x505e71695e9bc45943c58adec1650577bca68fd9": {"label": "Binance Cold",  "category": "exchange:binance"},
    "0x290275e3db66394c52272398959845170e4dcb88": {"label": "Binance Hot 2", "category": "exchange:binance"},

    # Kraken
    "0xa83b11093c858c86321fbc4c20fe82cdbd58e09e": {"label": "Kraken 1",      "category": "exchange:kraken"},
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": {"label": "Kraken 2",      "category": "exchange:kraken"},

    # OKX
    "0x06b1ea18b6d4e3b8ad8d9bda60ca6e4d96eb6b1d": {"label": "OKX 1",         "category": "exchange:okx"},
    "0xc708a1c712ba26dc618f972ad7a187f76c8596fd": {"label": "OKX 2",         "category": "exchange:okx"},

    # Bybit
    "0xee5b5b923ffce93a870b3104b7ca09c3db80047a": {"label": "Bybit Hot",     "category": "exchange:bybit"},
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40": {"label": "Bybit 2",       "category": "exchange:bybit"},

    # Bridges
    "0xa0c68c638235ee32657e8f720a23cec1bfc77c77": {"label": "Polygon PoS Bridge", "category": "bridge:polygon-pos"},
    "0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf": {"label": "Polygon ERC20 Bridge","category": "bridge:polygon-erc20"},
    "0x1f513585d8bb1f0b7d4d40e8b9ee6cb74d7e2e93": {"label": "LayerZero Endpoint",  "category": "bridge:layerzero"},
    "0x5a58505a96d1dbf8df91cb21b54419fc36e93fde": {"label": "Stargate Router",     "category": "bridge:stargate"},
    "0x8731d54e9d02c286767d56ac03e8037c07e01e98": {"label": "Stargate USDC Pool",  "category": "bridge:stargate"},
    "0x9a25d79ab755718e0b12bd3c927a010a543c2b31": {"label": "Synapse Bridge",      "category": "bridge:synapse"},
    "0xc0fbc4967259786c743361a5885ef49380473dcf": {"label": "Across SpokePool",    "category": "bridge:across"},

    # Polymarket contracts
    "0x4d97dcd97ec945f40cf65f87097ace5ea0476045": {"label": "Polymarket CTF Exchange", "category": "contract:polymarket"},
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e": {"label": "Polymarket Neg Risk Adapter", "category": "contract:polymarket"},
    "0xc5d563a36ae78145c45a50134d48a1215220f80a": {"label": "Polymarket Neg Risk CTF",     "category": "contract:polymarket"},
}


def lookup_address(address: str) -> dict[str, str] | None:
    """Case-insensitive lookup against the curated address registry."""
    if not address:
        return None
    return KNOWN_ADDRESSES.get(address.lower())
