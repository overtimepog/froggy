"""ToolSelector for optional tool filtering."""

from __future__ import annotations

from .tools import ToolDef, ToolRegistry


class ToolSelector:
    """Wraps a ToolRegistry and exposes a filtered subset of tools.

    Useful when the caller wants to restrict which tools are available
    for a given session or request without modifying the underlying registry.
    """

    def __init__(self, registry: ToolRegistry, allowed: list[str] | None = None):
        """
        Args:
            registry: The full ToolRegistry to filter from.
            allowed: If provided, only these tool names will be available.
                     If None, all tools from the registry are available.
        """
        self._registry = registry
        self._allowed: set[str] | None = set(allowed) if allowed is not None else None

    def select(self, names: list[str]) -> "ToolSelector":
        """Return a new ToolSelector restricted to the given tool names."""
        return ToolSelector(self._registry, names)

    def available(self) -> list[ToolDef]:
        """Return the list of currently available ToolDef objects."""
        if self._allowed is None:
            return self._registry.all()
        return [t for t in self._registry.all() if t.name in self._allowed]

    def names(self) -> list[str]:
        return [t.name for t in self.available()]

    def get(self, name: str) -> ToolDef | None:
        if self._allowed is not None and name not in self._allowed:
            return None
        return self._registry.get(name)

    def to_registry(self) -> ToolRegistry:
        """Return a new ToolRegistry containing only the selected tools."""
        return ToolRegistry(tools=self.available())
