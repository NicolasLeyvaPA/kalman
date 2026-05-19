"""Classify a Polygon address as exchange / bridge / contract / wallet."""

from __future__ import annotations

from chain.alchemy_client import AlchemyClient
from chain.exchange_identifier import identify
from exceptions import ExternalAPIError
from utils.logging import get_logger

log = get_logger(__name__)


async def classify_address(
    address: str, alchemy: AlchemyClient | None = None,
) -> dict[str, str]:
    """Return classification metadata for an address.

    Returns:
        ``{"type": "exchange"|"bridge"|"contract"|"wallet"|"unknown",
            "label": "...", "exchange": "..."}``

        ``label`` is the human-readable name for known addresses.
        ``exchange`` is the exchange slug if ``type == "exchange"``,
        otherwise empty string.
    """
    category, label = identify(address)
    if category:
        prefix = category.split(":", 1)[0]
        exchange = category.split(":", 1)[1] if prefix == "exchange" else ""
        return {"type": prefix, "label": label or "", "exchange": exchange}

    if alchemy is None:
        return {"type": "wallet", "label": "", "exchange": ""}

    try:
        code = await alchemy.get_code(address)
    except ExternalAPIError as exc:
        log.debug("classify_get_code_failed", address=address, error=exc.message)
        return {"type": "wallet", "label": "", "exchange": ""}

    if code and code != "0x":
        return {"type": "contract", "label": "", "exchange": ""}
    return {"type": "wallet", "label": "", "exchange": ""}
