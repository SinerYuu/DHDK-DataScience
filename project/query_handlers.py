"""Query handlers that expose DataFrame-based APIs."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .handlers_base import QueryHandler
from .repositories import SparqlJournalRepository, SQLiteCategoryRepository


class JournalQueryHandler(QueryHandler):
    """Query DOAJ journal information via the SPARQL repository."""

    COLUMNS = [
        "id",
        "title",
        "print_issn",
        "electronic_issn",
        "languages",
        "publisher",
        "doaj_seal",
        "license",
        "apc",
        "identifiers",
    ]

    def __init__(self):
        super().__init__()
        self._repository: Optional[SparqlJournalRepository] = None

    def setDbPathOrUrl(self, db_path_or_url: str) -> bool:
        if not super().setDbPathOrUrl(db_path_or_url):
            return False
        self._repository = SparqlJournalRepository(self.getDbPathOrUrl())
        return True

    def _empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.COLUMNS)

    def _repo(self) -> Optional[SparqlJournalRepository]:
        return self._repository

    def getById(self, identifier: str) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return self._empty_frame()
        return repo.fetch_by_id(identifier)

    def getAllJournals(self) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return self._empty_frame()
        return repo.fetch_all()

    def getJournalsWithTitle(self, title_part: str) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return self._empty_frame()
        return repo.fetch_by_title(title_part)

    def getJournalsPublishedBy(self, publisher_part: str) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return self._empty_frame()
        return repo.fetch_by_publisher(publisher_part)

    def getJournalsWithLicense(self, licenses: Iterable[str]) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return self._empty_frame()
        return repo.fetch_by_license(licenses)

    def getJournalsWithAPC(self) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return self._empty_frame()
        return repo.fetch_with_apc()

    def getJournalsWithDOAJSeal(self) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return self._empty_frame()
        return repo.fetch_with_doaj_seal()

    @property
    def repository(self) -> Optional[SparqlJournalRepository]:
        return self._repository


class CategoryQueryHandler(QueryHandler):
    """Query category and area data via the SQLite repository."""

    def __init__(self):
        super().__init__()
        self._repository: Optional[SQLiteCategoryRepository] = None

    def setDbPathOrUrl(self, db_path_or_url: str) -> bool:
        if not super().setDbPathOrUrl(db_path_or_url):
            return False
        self._repository = SQLiteCategoryRepository(self.getDbPathOrUrl())
        return True

    def _repo(self) -> Optional[SQLiteCategoryRepository]:
        return self._repository

    def _empty(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["type", "id", "quartiles", "areas", "categories"]
        )

    def getById(self, identifier: str) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return self._empty()
        return repo.fetch_entity_by_identifier(identifier)

    def getAllCategories(self) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return pd.DataFrame(columns=["id", "quartiles", "areas"])
        return repo.fetch_categories()

    def getAllAreas(self) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return pd.DataFrame(columns=["id", "categories"])
        return repo.fetch_areas()

    def getCategoriesWithQuartile(self, quartiles: Iterable[str]) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return pd.DataFrame(columns=["id", "quartiles", "areas"])
        return repo.fetch_categories_with_quartiles(quartiles)

    def getCategoriesAssignedToAreas(self, areas: Iterable[str]) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return pd.DataFrame(columns=["id", "quartiles", "areas"])
        return repo.fetch_categories_assigned_to_areas(areas)

    def getAreasAssignedToCategories(self, categories: Iterable[str]) -> pd.DataFrame:
        repo = self._repo()
        if not repo:
            return pd.DataFrame(columns=["id", "categories"])
        return repo.fetch_areas_assigned_to_categories(categories)

    @property
    def repository(self) -> Optional[SQLiteCategoryRepository]:
        return self._repository
