"""Query engines combining multiple handlers into higher-level objects."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .handlers_base import Handler  # noqa: F401  # for UML reference
from .models import Area, Category, IdentifiableEntity, Journal
from .query_handlers import CategoryQueryHandler, JournalQueryHandler
from .repositories import _normalise_identifier


class BasicQueryEngine:
    """Aggregate journal and category query handlers and expose domain objects."""

    def __init__(self):
        self.journalQuery: List[JournalQueryHandler] = []
        self.categoryQuery: List[CategoryQueryHandler] = []

    # -- handler management ----------------------------------------------------

    def cleanJournalHandlers(self) -> bool:
        self.journalQuery.clear()
        return True

    def cleanCategoryHandlers(self) -> bool:
        self.categoryQuery.clear()
        return True

    def addJournalHandler(self, handler: JournalQueryHandler) -> bool:
        if handler and handler not in self.journalQuery:
            self.journalQuery.append(handler)
            return True
        return False

    def addCategoryHandler(self, handler: CategoryQueryHandler) -> bool:
        if handler and handler not in self.categoryQuery:
            self.categoryQuery.append(handler)
            return True
        return False

    # -- helpers ---------------------------------------------------------------

    def _collect_journal_frames(self, method_name: str, *args, **kwargs) -> pd.DataFrame:
        frames = []
        for handler in self.journalQuery:
            method = getattr(handler, method_name, None)
            if callable(method):
                frame = method(*args, **kwargs)
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=JournalQueryHandler.COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        if "id" in combined.columns:
            combined = combined.drop_duplicates(subset="id", keep="first")
        return combined.reset_index(drop=True)

    def _collect_category_exports(self) -> Dict[str, dict]:
        combined = {
            "categories": {},
            "areas": {},
            "journal_categories": {},
            "journal_areas": {},
            "journal_alias": {},
        }

        for handler in self.categoryQuery:
            repository = handler.repository
            if not repository:
                continue
            export = repository.export_all()

            for cid, data in export.get("categories", {}).items():
                record = combined["categories"].setdefault(cid, {"quartiles": set(), "areas": set()})
                record["quartiles"].update(data.get("quartiles", set()))
                record["areas"].update(data.get("areas", set()))

            for aid, data in export.get("areas", {}).items():
                record = combined["areas"].setdefault(aid, {"categories": set()})
                record["categories"].update(data.get("categories", set()))

            for jid, data in export.get("journal_categories", {}).items():
                journal_record = combined["journal_categories"].setdefault(jid, {})
                for cid, quartiles in data.items():
                    journal_record.setdefault(cid, set()).update(quartiles)

            for jid, areas in export.get("journal_areas", {}).items():
                record = combined["journal_areas"].setdefault(jid, set())
                record.update(areas)

            for alias_norm, canonical in export.get("journal_alias", {}).items():
                combined["journal_alias"].setdefault(alias_norm, canonical)

        return combined

    def _build_taxonomy(self) -> Tuple[Dict[str, Category], Dict[str, Area], Dict[str, dict]]:
        exports = self._collect_category_exports()

        category_map: Dict[str, Category] = {}
        for cid, data in exports["categories"].items():
            category = category_map.setdefault(cid, Category(cid))
            for quartile in data.get("quartiles", set()):
                category.addQuartile(quartile)

        area_map: Dict[str, Area] = {}
        for aid, data in exports["areas"].items():
            area = area_map.setdefault(aid, Area(aid))
            for cid in data.get("categories", set()):
                category = category_map.setdefault(cid, Category(cid))
                category.addArea(area)

        return category_map, area_map, exports

    def _build_journal_objects(self, frame: pd.DataFrame) -> List[Journal]:
        if frame is None or frame.empty:
            return []

        category_map, area_map, exports = self._build_taxonomy()
        alias_map = exports.get("journal_alias", {})
        journal_categories = exports.get("journal_categories", {})
        journal_areas = exports.get("journal_areas", {})

        journals: List[Journal] = []
        for _, row in frame.iterrows():
            record = row.to_dict()
            journal = Journal(
                identifier=record.get("id"),
                title=record.get("title"),
                print_issn=record.get("print_issn"),
                electronic_issn=record.get("electronic_issn"),
                publisher=record.get("publisher"),
                languages=list(record.get("languages") or []),
                license_=record.get("license"),
                has_apc=bool(record.get("apc")),
                has_doaj_seal=bool(record.get("doaj_seal")),
            )

            identifiers = [identifier for identifier in (record.get("identifiers") or []) if identifier]

            for identifier in identifiers:
                canonical = alias_map.get(_normalise_identifier(identifier))
                if not canonical:
                    canonical = alias_map.get(_normalise_identifier(identifier.replace("-", "")))
                if not canonical:
                    continue

                for category_id in journal_categories.get(canonical, {}).keys():
                    category = category_map.get(category_id)
                    if category:
                        journal.addCategory(category)
                for area_id in journal_areas.get(canonical, set()):
                    area = area_map.get(area_id)
                    if area:
                        journal.addArea(area)

            journals.append(journal)

        return journals

    def _find_category_by_identifier(
        self,
        category_map: Dict[str, Category],
        identifier: str,
    ) -> Optional[Category]:
        norm = _normalise_identifier(identifier)
        for category in category_map.values():
            if _normalise_identifier(category.getId()) == norm:
                return category
        return None

    def _find_area_by_identifier(
        self,
        area_map: Dict[str, Area],
        identifier: str,
    ) -> Optional[Area]:
        norm = _normalise_identifier(identifier)
        for area in area_map.values():
            if _normalise_identifier(area.getId()) == norm:
                return area
        return None

    # -- public API ------------------------------------------------------------

    def getEntityById(self, identifier: str) -> Optional[IdentifiableEntity]:
        frame = self._collect_journal_frames("getById", identifier)
        journals = self._build_journal_objects(frame)
        if journals:
            return journals[0]

        category_map, area_map, _ = self._build_taxonomy()
        category = self._find_category_by_identifier(category_map, identifier)
        if category:
            return category
        area = self._find_area_by_identifier(area_map, identifier)
        if area:
            return area
        return None

    def getAllJournals(self) -> List[Journal]:
        frame = self._collect_journal_frames("getAllJournals")
        return self._build_journal_objects(frame)

    def getJournalsWithTitle(self, title_part: str) -> List[Journal]:
        frame = self._collect_journal_frames("getJournalsWithTitle", title_part)
        return self._build_journal_objects(frame)

    def getJournalsPublishedBy(self, publisher_part: str) -> List[Journal]:
        frame = self._collect_journal_frames("getJournalsPublishedBy", publisher_part)
        return self._build_journal_objects(frame)

    def getJournalsWithLicense(self, licenses: Iterable[str]) -> List[Journal]:
        frame = self._collect_journal_frames("getJournalsWithLicense", licenses)
        return self._build_journal_objects(frame)

    def getJournalsWithAPC(self) -> List[Journal]:
        frame = self._collect_journal_frames("getJournalsWithAPC")
        return self._build_journal_objects(frame)

    def getJournalsWithDOAJSeal(self) -> List[Journal]:
        frame = self._collect_journal_frames("getJournalsWithDOAJSeal")
        return self._build_journal_objects(frame)

    def getAllCategories(self) -> List[Category]:
        category_map, _, _ = self._build_taxonomy()
        return list(category_map.values())

    def getAllAreas(self) -> List[Area]:
        _, area_map, _ = self._build_taxonomy()
        return list(area_map.values())

    def getCategoriesWithQuartile(self, quartiles: Iterable[str]) -> List[Category]:
        target = {_normalise_identifier(q) for q in quartiles or [] if q}
        result = []
        for category in self.getAllCategories():
            quartile_norm = {_normalise_identifier(q) for q in category.getQuartiles()}
            if not target or quartile_norm.intersection(target):
                result.append(category)
        return result

    def getCategoriesAssignedToAreas(self, areas: Iterable[str]) -> List[Category]:
        target = {_normalise_identifier(a) for a in areas or [] if a}
        result = []
        for category in self.getAllCategories():
            area_norm = {_normalise_identifier(area.getId()) for area in category.getAreas()}
            if not target or area_norm.intersection(target):
                result.append(category)
        return result

    def getAreasAssignedToCategories(self, categories: Iterable[str]) -> List[Area]:
        target = {_normalise_identifier(c) for c in categories or [] if c}
        result = []
        for area in self.getAllAreas():
            category_norm = {_normalise_identifier(category.getId()) for category in area.getCategories()}
            if not target or category_norm.intersection(target):
                result.append(area)
        return result


class FullQueryEngine(BasicQueryEngine):
    """Extend the basic engine with mashup queries."""

    def _journal_dataframe(self) -> pd.DataFrame:
        return self._collect_journal_frames("getAllJournals")

    def _select_journals_by_license(
        self,
        df: pd.DataFrame,
        licenses: Iterable[str],
    ) -> pd.DataFrame:
        license_norm = {_normalise_identifier(l) for l in licenses or [] if l}
        if df.empty or not license_norm:
            return df
        mask = df["license"].apply(
            lambda value: _normalise_identifier(value) in license_norm if isinstance(value, str) else False
        )
        return df.loc[mask].reset_index(drop=True)

    def getJournalsInCategoriesWithQuartile(
        self,
        categories: Iterable[str],
        quartiles: Iterable[str],
    ) -> List[Journal]:
        requested_categories = {_normalise_identifier(c) for c in categories or [] if c}
        requested_quartiles = {_normalise_identifier(q) for q in quartiles or [] if q}

        df = self._journal_dataframe()
        journals = self._build_journal_objects(df)

        result = []
        for journal in journals:
            for category in journal.getCategories():
                category_id_norm = _normalise_identifier(category.getId())
                quartile_norm = {_normalise_identifier(q) for q in category.getQuartiles()}
                cat_match = not requested_categories or category_id_norm in requested_categories
                quartile_match = not requested_quartiles or quartile_norm.intersection(requested_quartiles)
                if cat_match and quartile_match:
                    result.append(journal)
                    break
        return result

    def getJournalsInAreasWithLicense(
        self,
        areas: Iterable[str],
        licenses: Iterable[str],
    ) -> List[Journal]:
        requested_areas = {_normalise_identifier(a) for a in areas or [] if a}

        df = self._select_journals_by_license(self._journal_dataframe(), licenses)
        journals = self._build_journal_objects(df)

        result = []
        for journal in journals:
            area_norm = {_normalise_identifier(area.getId()) for area in journal.getAreas()}
            if not requested_areas or area_norm.intersection(requested_areas):
                result.append(journal)
        return result

    def getDiamondJournalsInAreasAndCategoriesWithQuartile(
        self,
        areas: Iterable[str],
        categories: Iterable[str],
        quartiles: Iterable[str],
    ) -> List[Journal]:
        requested_areas = {_normalise_identifier(a) for a in areas or [] if a}
        requested_categories = {_normalise_identifier(c) for c in categories or [] if c}
        requested_quartiles = {_normalise_identifier(q) for q in quartiles or [] if q}

        df = self._journal_dataframe()
        # Filter upfront for journals without APC
        df = df.loc[df["apc"] == False].reset_index(drop=True)  # noqa: E712
        journals = self._build_journal_objects(df)

        result = []
        for journal in journals:
            area_norm = {_normalise_identifier(area.getId()) for area in journal.getAreas()}
            if requested_areas and not area_norm.intersection(requested_areas):
                continue

            categories_norm = [
                (
                    _normalise_identifier(category.getId()),
                    {_normalise_identifier(q) for q in category.getQuartiles()},
                )
                for category in journal.getCategories()
            ]

            category_match = False
            for category_id, quartile_set in categories_norm:
                cat_ok = not requested_categories or category_id in requested_categories
                quart_ok = not requested_quartiles or quartile_set.intersection(requested_quartiles)
                if cat_ok and quart_ok:
                    category_match = True
                    break

            if category_match:
                result.append(journal)

        return result
