"""Backend interface. A backend loads a model artifact and runs inference;
the runner never needs to know which runtime or vendor is underneath."""

from __future__ import annotations

from typing import Any, Protocol


class Backend(Protocol):
    name: str

    def infer(self, inputs: Any) -> Any: ...

    def describe(self) -> dict[str, Any]: ...

    def provider(self) -> str: ...
