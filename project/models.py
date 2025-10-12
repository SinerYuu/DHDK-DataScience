"""Domain model classes used across the project."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Sequence, Set


class IdentifiableEntity:
    """Base class for entities that expose an identifier and a human-friendly name."""

    def __init__(self, entity_id: Optional[str] = None, name: Optional[str] = None):
        self._id = (entity_id or "").strip()
        self._name = (name or "").strip()

    def getId(self) -> str:
        return self._id

    def hasId(self) -> bool:
        return bool(self._id)

    def setId(self, entity_id: Optional[str]) -> None:
        self._id = (entity_id or "").strip()

    def getName(self) -> str:
        return self._name

    def hasName(self) -> bool:
        return bool(self._name)

    def setName(self, name: Optional[str]) -> None:
        self._name = (name or "").strip()


class Area(IdentifiableEntity):
    """Represents a Scimago area."""

    def __init__(self, area_id: Optional[str] = None):
        super().__init__(area_id, area_id)
        self._categories: "OrderedDict[str, Category]" = OrderedDict()

    def _link_category(self, category: "Category") -> None:
        if category and category.getId():
            self._categories.setdefault(category.getId(), category)

    def addCategory(self, category: "Category") -> None:
        if category and category.getId():
            category._link_area(self)
            self._link_category(category)

    def getCategories(self) -> List["Category"]:
        return list(self._categories.values())

    def hasCategories(self) -> bool:
        return bool(self._categories)


class Category(IdentifiableEntity):
    """Represents a Scimago category associated with quartiles and areas."""

    def __init__(self, category_id: Optional[str] = None):
        super().__init__(category_id, category_id)
        self._quartiles: "OrderedDict[str, bool]" = OrderedDict()
        self._areas: "OrderedDict[str, Area]" = OrderedDict()

    def addQuartile(self, quartile: Optional[str]) -> None:
        if quartile:
            quartile_clean = quartile.strip()
            if quartile_clean:
                self._quartiles.setdefault(quartile_clean, True)

    def getQuartiles(self) -> List[str]:
        return list(self._quartiles.keys())

    def hasQuartiles(self) -> bool:
        return bool(self._quartiles)

    def _link_area(self, area: Area) -> None:
        if area and area.getId():
            self._areas.setdefault(area.getId(), area)

    def addArea(self, area: Area) -> None:
        if area and area.getId():
            self._link_area(area)
            area._link_category(self)

    def getAreas(self) -> List[Area]:
        return list(self._areas.values())

    def hasAreas(self) -> bool:
        return bool(self._areas)


class Journal(IdentifiableEntity):
    """Represents a DOAJ journal."""

    def __init__(
        self,
        identifier: Optional[str] = None,
        title: Optional[str] = None,
        print_issn: Optional[str] = None,
        electronic_issn: Optional[str] = None,
        publisher: Optional[str] = None,
        languages: Optional[Sequence[str]] = None,
        license_: Optional[str] = None,
        has_apc: bool = False,
        has_doaj_seal: bool = False,
    ):
        super().__init__(identifier, title)
        self._title = (title or "").strip()
        self._print_issn = (print_issn or "").strip()
        self._electronic_issn = (electronic_issn or "").strip()
        self._publisher = (publisher or "").strip()
        self._languages = [lang.strip() for lang in (languages or []) if lang and lang.strip()]
        self._license = (license_ or "").strip()
        self._has_apc = bool(has_apc)
        self._has_doaj_seal = bool(has_doaj_seal)
        self._categories: "OrderedDict[str, Category]" = OrderedDict()
        self._areas: "OrderedDict[str, Area]" = OrderedDict()

    def getTitle(self) -> str:
        return self._title

    def hasTitle(self) -> bool:
        return bool(self._title)

    def getPrintIssn(self) -> str:
        return self._print_issn

    def hasPrintIssn(self) -> bool:
        return bool(self._print_issn)

    def getElectronicIssn(self) -> str:
        return self._electronic_issn

    def hasElectronicIssn(self) -> bool:
        return bool(self._electronic_issn)

    def getPublisher(self) -> str:
        return self._publisher

    def hasPublisher(self) -> bool:
        return bool(self._publisher)

    def getLanguages(self) -> List[str]:
        return list(self._languages)

    def hasLanguages(self) -> bool:
        return bool(self._languages)

    def getLicense(self) -> str:
        return self._license

    def hasLicense(self) -> bool:
        return bool(self._license)

    def hasAPC(self) -> bool:
        return self._has_apc

    def hasDOAJSeal(self) -> bool:
        return self._has_doaj_seal

    def addCategory(self, category: Category) -> None:
        if category and category.getId():
            self._categories.setdefault(category.getId(), category)

    def getCategories(self) -> List[Category]:
        return list(self._categories.values())

    def hasCategories(self) -> bool:
        return bool(self._categories)

    def addArea(self, area: Area) -> None:
        if area and area.getId():
            self._areas.setdefault(area.getId(), area)

    def getAreas(self) -> List[Area]:
        return list(self._areas.values())

    def hasAreas(self) -> bool:
        return bool(self._areas)

    def getAllIdentifiers(self) -> Set[str]:
        identifiers = {self.getId()}
        if self._print_issn:
            identifiers.add(self._print_issn)
        if self._electronic_issn:
            identifiers.add(self._electronic_issn)
        if self._title:
            identifiers.add(self._title)
        return {identifier for identifier in identifiers if identifier}

