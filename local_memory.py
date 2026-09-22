"""Small in-process memory used for within-run agent ablations.

The store intentionally has no persistence or retrieval dependencies. A new
instance is created for each experiment configuration, so memories cannot
leak across runs or between users.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Optional


@dataclass(frozen=True)
class MemoryEntry:
    """A generated review that can be reused by a later task."""

    text: str
    stars: Optional[float] = None
    item_id: Optional[str] = None
    sequence: int = 0


class LocalMemoryStore:
    """Bounded, user-scoped memory for one experiment run."""

    def __init__(self, max_entries_per_user: int = 8):
        if max_entries_per_user < 1:
            raise ValueError("max_entries_per_user must be at least 1")
        self.max_entries_per_user = max_entries_per_user
        self._entries: dict[str, list[MemoryEntry]] = {}
        self._sequence = 0
        self._lock = RLock()

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        stars: Optional[float] = None,
        item_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Append one generated review and return the stored entry."""
        key = self._validate_user_id(user_id)
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            raise ValueError("memory text must not be empty")

        with self._lock:
            self._sequence += 1
            entry = MemoryEntry(
                text=cleaned_text,
                stars=stars,
                item_id=item_id,
                sequence=self._sequence,
            )
            entries = self._entries.setdefault(key, [])
            entries.append(entry)
            del entries[:-self.max_entries_per_user]
            return entry

    def recall(self, user_id: str, limit: int = 5) -> list[MemoryEntry]:
        """Return the user's newest entries first."""
        key = self._validate_user_id(user_id)
        if limit <= 0:
            return []

        with self._lock:
            return list(reversed(self._entries.get(key, [])[-limit:]))

    def clear(self) -> None:
        """Remove all entries from this experiment-local store."""
        with self._lock:
            self._entries.clear()
            self._sequence = 0

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        key = str(user_id or "").strip()
        if not key:
            raise ValueError("user_id must not be empty")
        return key
