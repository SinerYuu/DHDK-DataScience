"""Shared base classes for handlers."""

from __future__ import annotations

from typing import Optional

import pandas as pd


class Handler:
    """Base handler storing the database path/URL configuration."""

    def __init__(self):
        self._dbPathOrUrl = ""

    def getDbPathOrUrl(self) -> str:
        return self._dbPathOrUrl

    def setDbPathOrUrl(self, db_path_or_url: Optional[str]) -> bool:
        if not isinstance(db_path_or_url, str):
            return False
        self._dbPathOrUrl = db_path_or_url.strip()
        return True


class UploadHandler(Handler):
    """Base class for upload handlers."""

    def pushDataToDb(self, path_to_file: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class QueryHandler(Handler):
    """Base class for query handlers."""

    def getById(self, identifier: str) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

