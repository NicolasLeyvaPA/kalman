"""Identify whether an address belongs to a known exchange or bridge."""

from __future__ import annotations

from data.known_exchanges import lookup_address


def identify(address: str) -> tuple[str | None, str | None]:
    """Return ``(category, label)`` for a known address, else ``(None, None)``.

    Category format: ``"exchange:coinbase"``, ``"bridge:polygon-pos"``,
    ``"contract:polymarket"``.
    """
    if not address:
        return None, None
    meta = lookup_address(address.lower())
    if not meta:
        return None, None
    return meta["category"], meta["label"]


def is_exchange(address: str) -> bool:
    cat, _ = identify(address)
    return bool(cat and cat.startswith("exchange:"))


def is_bridge(address: str) -> bool:
    cat, _ = identify(address)
    return bool(cat and cat.startswith("bridge:"))


def is_polymarket(address: str) -> bool:
    cat, _ = identify(address)
    return bool(cat and cat.startswith("contract:polymarket"))


def exchange_name(address: str) -> str | None:
    cat, _ = identify(address)
    if cat and cat.startswith("exchange:"):
        return cat.split(":", 1)[1]
    return None
