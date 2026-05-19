"""Custom exception hierarchy.

Every external boundary in the system raises a typed exception so callers
can react meaningfully (retry, alert, fail-open) instead of blanket-catching
``Exception``. The base class carries an optional ``context`` mapping which
gets logged by the structured-logging middleware.
"""

from __future__ import annotations

from typing import Any


class ForensicsError(Exception):
    """Root of the application's exception hierarchy."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})


# --- External-API errors ---------------------------------------------------

class ExternalAPIError(ForensicsError):
    """A third-party HTTP API returned a non-success response."""

    def __init__(
        self,
        service: str,
        endpoint: str,
        status: int,
        body: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.service = service
        self.endpoint = endpoint
        self.status = status
        self.body = body[:500]
        ctx = {"service": service, "endpoint": endpoint, "status": status, **(context or {})}
        super().__init__(
            f"{service} returned {status} on {endpoint}", context=ctx,
        )


class RateLimitError(ExternalAPIError):
    """Caller must back off and retry after ``retry_after`` seconds."""

    def __init__(
        self, service: str, endpoint: str, retry_after: float, body: str = "",
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            service, endpoint, 429, body, context={"retry_after": retry_after},
        )


class UpstreamUnavailableError(ExternalAPIError):
    """503/timeout — the upstream is temporarily down."""


class PolymarketAPIError(ExternalAPIError):
    """Polymarket Gamma / Data / CLOB API returned an error."""

    def __init__(
        self, endpoint: str, status: int, body: str,
        *, context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("polymarket", endpoint, status, body, context=context)


class AlchemyAPIError(ExternalAPIError):
    """Alchemy JSON-RPC returned an error."""

    def __init__(
        self, method: str, status: int, body: str,
        *, context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("alchemy", method, status, body, context=context)


class AlchemyConfigError(ForensicsError):
    """Alchemy client invoked without an API key configured."""


# --- Domain errors ---------------------------------------------------------

class InsufficientDataError(ForensicsError):
    """Not enough data to produce a meaningful result."""


class WalletNotFoundError(ForensicsError):
    """Requested wallet does not exist in the database."""

    def __init__(self, address: str) -> None:
        super().__init__(f"wallet not found: {address}", context={"address": address})


class ClusterNotFoundError(ForensicsError):
    """Requested cluster does not exist."""

    def __init__(self, cluster_id: str) -> None:
        super().__init__(f"cluster not found: {cluster_id}", context={"cluster_id": cluster_id})


class AlertNotFoundError(ForensicsError):
    """Requested alert does not exist."""

    def __init__(self, alert_id: int) -> None:
        super().__init__(f"alert not found: {alert_id}", context={"alert_id": alert_id})
