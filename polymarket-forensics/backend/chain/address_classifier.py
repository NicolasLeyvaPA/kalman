"""
Classify a Polygon address as exchange / bridge / contract / wallet.
"""
from chain.alchemy_client import AlchemyClient
from chain.exchange_identifier import identify


async def classify_address(address: str, alchemy: AlchemyClient | None = None) -> dict[str, str]:
    """
    Returns:
      {
        "type":     "exchange" | "bridge" | "contract" | "wallet" | "unknown",
        "label":    human-readable label (or empty),
        "exchange": exchange name if applicable,
      }
    """
    category, label = identify(address)
    if category:
        prefix = category.split(":", 1)[0]
        return {
            "type": prefix,
            "label": label or "",
            "exchange": category.split(":", 1)[1] if prefix == "exchange" else "",
        }

    if alchemy is None:
        return {"type": "wallet", "label": "", "exchange": ""}

    try:
        code = await alchemy.get_code(address)
    except Exception:
        code = "0x"

    if code and code != "0x":
        return {"type": "contract", "label": "", "exchange": ""}
    return {"type": "wallet", "label": "", "exchange": ""}
