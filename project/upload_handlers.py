"""Upload handlers delegating to the repositories."""

from __future__ import annotations

from typing import Optional

from .handlers_base import UploadHandler
from .repositories import SparqlJournalRepository, SQLiteCategoryRepository


class JournalUploadHandler(UploadHandler):
    """Handle loading CSV metadata into the graph database."""

    def __init__(self):
        super().__init__()
        self._repository: Optional[SparqlJournalRepository] = None

    def setDbPathOrUrl(self, db_path_or_url: str) -> bool:
        if not super().setDbPathOrUrl(db_path_or_url):
            return False
        self._repository = SparqlJournalRepository(self.getDbPathOrUrl())
        return True

    def pushDataToDb(self, path_to_file: str) -> bool:
        if not self._repository:
            return False
        return self._repository.load_csv(path_to_file)

    @property
    def repository(self) -> Optional[SparqlJournalRepository]:
        return self._repository


class CategoryUploadHandler(UploadHandler):
    """Handle loading JSON category data into the relational database."""

    def __init__(self):
        super().__init__()
        self._repository: Optional[SQLiteCategoryRepository] = None

    def setDbPathOrUrl(self, db_path_or_url: str) -> bool:
        if not super().setDbPathOrUrl(db_path_or_url):
            return False
        self._repository = SQLiteCategoryRepository(self.getDbPathOrUrl())
        return True

    def pushDataToDb(self, path_to_file: str) -> bool:
        if not self._repository:
            return False
        return self._repository.load_json(path_to_file)

    @property
    def repository(self) -> Optional[SQLiteCategoryRepository]:
        return self._repository

