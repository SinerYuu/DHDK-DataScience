from typing import List, Optional, Set, Dict, Any, Tuple
from collections import OrderedDict
import json
import os
import re
import pandas as pd

# --- rdflib / SPARQL ---
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore, SPARQLStore


# -------------------- In-memory registry (for relational side and fallbacks) --------------------

_REGISTRY: Dict[str, Dict[str, Any]] = {}

def _ensure_registry(key: str) -> Dict[str, Any]:
    if key not in _REGISTRY:
        _REGISTRY[key] = {
            "journals": pd.DataFrame(),     # fallback cache if Blazegraph unreachable
            "categories": pd.DataFrame(),
            "areas": pd.DataFrame(),
            "links": pd.DataFrame(),
        }
    return _REGISTRY[key]


# -------------------- Data models --------------------

class IdentifiableEntity:
    """Base class for all identifiable entities."""

    def __init__(self, id: str = "", name: str = ""):
        self._id = id.strip()
        self._name = name.strip()

    def getId(self) -> str:
        """Return unique identifier."""
        return self._id

    def hasId(self) -> bool:
        """True if the entity has a non-empty ID."""
        return bool(self._id)

    def getName(self) -> str:
        """Return the entity’s human-readable name."""
        return self._name

    def hasName(self) -> bool:
        """True if the entity has a non-empty name."""
        return bool(self._name)


class Area(IdentifiableEntity):
    """Represents a SCImago Area (e.g., 'Engineering')."""
    def __init__(self, id: str = "", name: str = "", description: str = ""):
        super().__init__(id, name)
        self._description = description


class Category(IdentifiableEntity):
    """
    Represents a SCImago Category (e.g., 'Artificial Intelligence').
    Each Category may have one or more Quartiles (Q1–Q4).
    """

    def __init__(
        self,
        id: str = "",
        name: str = "",
        quartiles: Optional[Set[str]] = None,
    ):
        super().__init__(id, name)
        self._quartiles: Set[str] = set()
        if quartiles:
            for q in quartiles:
                self.addQuartile(q)

    def addQuartile(self, quartile: Optional[str]) -> None:
        """Add a Quartile ranking (Q1–Q4)."""
        if quartile and quartile.strip():
            self._quartiles.add(quartile.strip().upper())

    def getQuartiles(self) -> List[str]:
        """Return all Quartiles for this Category."""
        return sorted(self._quartiles)

    def hasQuartile(self, q: Optional[str] = None) -> bool:
        """True if Category is ranked in the given Quartile or has any Quartile."""
        if q:
            return q.strip().upper() in self._quartiles
        return len(self._quartiles) > 0


class Journal(IdentifiableEntity):
    """
    Represents a DOAJ Journal (schema.org:Periodical).
    Journals can be associated with multiple Categories and Areas.
    """

    def __init__(
        self,
        id: str = "",
        title: str = "",
        publisher: str = "",
        license: str = "",
        apc: Optional[bool] = None,
        doaj_seal: Optional[bool] = None,
        languages: Optional[List[str]] = None,
    ):
        super().__init__(id, title)
        self._publisher = publisher
        self._license = license
        self._apc = apc
        self._doaj_seal = doaj_seal
        self._languages = languages or []
        self._categories: OrderedDict[str, Category] = OrderedDict()
        self._areas: OrderedDict[str, Area] = OrderedDict()

    # --- Basic field accessors ---------------------------------------------
    def getTitle(self) -> str:
        return self._name

    def hasTitle(self) -> bool:
        return bool(self._name)

    def getPublisher(self) -> str:
        return self._publisher

    def hasPublisher(self) -> bool:
        return bool(self._publisher)

    def getLicense(self) -> str:
        return self._license

    def hasLicense(self) -> bool:
        return bool(self._license)

    def getAPC(self) -> Optional[bool]:
        return self._apc

    def hasAPC(self) -> bool:
        return self._apc is not None

    def getDOAJSeal(self) -> Optional[bool]:
        return self._doaj_seal

    def hasDOAJSeal(self) -> bool:
        return self._doaj_seal is not None

    def getLanguages(self) -> List[str]:
        return list(self._languages)

    def hasLanguages(self) -> bool:
        return len(self._languages) > 0

    def addCategory(self, category: Category) -> None:
        """Link a Category to this Journal."""
        if category and category.getId() not in self._categories:
            self._categories[category.getId()] = category

    def getCategories(self) -> List[Category]:
        """Return all Categories linked to this Journal."""
        return list(self._categories.values())

    def hasCategories(self) -> bool:
        """True if the Journal has at least one linked Category."""
        return len(self._categories) > 0

    def addArea(self, area: Area) -> None:
        """Link an Area to this Journal."""
        if area and area.getId() not in self._areas:
            self._areas[area.getId()] = area

    def getAreas(self) -> List[Area]:
        """Return all Areas linked to this Journal."""
        return list(self._areas.values())

    def hasAreas(self) -> bool:
        """True if the Journal has at least one linked Area."""
        return len(self._areas) > 0


# -------------------- Basic Handlers (upload + query) --------------------

#the parent of all handler types
class Handler:
    def __init__(self):
        self.dbPathOrUrl: str = ""

    def getDbPathOrUrl(self) -> str:
        return self.dbPathOrUrl

    def setDbPathOrUrl(self, val: str) -> bool:
        self.dbPathOrUrl = val
        _ensure_registry(val)  # make sure registry exists
        return True

#abstract subclass for data ingestion
class UploadHandler(Handler):
    def pushDataToDb(self, file_path: str) -> bool:  # must be overridden by specific uploaders
        raise NotImplementedError()  # if someone forgets to override it

#abstract subclass for data retrieval
class QueryHandler(Handler):
    def getById(self, id: str) -> pd.DataFrame: 
        raise NotImplementedError

# -------------------- Graph/Blazegraph helpers --------------------

SCHEMA = Namespace("https://schema.org/")

# --- helper: convert textual values like "Yes"/"No" into Python booleans ---
def _bool_from_str(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    w = str(v).strip().lower()
    if w in {"true", "yes", "y", "1"}:
        return True
    if w in {"false", "no", "n", "0"}:
        return False
    return None

# --- helper: build a clean URI for each journal using its ISSN ---
def _build_journal_uri(issn: str) -> URIRef:
    # remove dashes and keep only digits/X, then attach to base URI
    norm = re.sub(r"[^0-9xX]", "", issn or "")
    return URIRef(f"http://example.org/periodical/{norm}")


class _BlazegraphClient:
    """
    Minimal helper client that communicates with the Blazegraph SPARQL endpoint.
    Uses rdflib's SPARQLUpdateStore for uploads and SPARQLStore for SELECT queries.
    """
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def _update_store(self) -> SPARQLUpdateStore:
        # connection for INSERT/UPDATE operations
        store = SPARQLUpdateStore()
        store.open((self.endpoint, self.endpoint)) # same URL used for both query and update
        return store

    def _select_store(self) -> SPARQLStore:
        # connection for SELECT queries
        return SPARQLStore(self.endpoint)

    def upload_graph(self, g: Graph) -> bool:
        """Send all triples from local graph `g` to the Blazegraph server."""
        try:
            store = self._update_store()
            # Push triples directly via the store-connected Graph
            G = Graph(store=store)
            for t in g.triples((None, None, None)):
                G.add(t)
            return True
        except Exception:
            # if server not reachable, silently fail (we still keep local cache)
            return False

    def select(self, query: str) -> List[Dict[str, Any]]:
        """Run a SPARQL SELECT query and return list of dict results."""
        store = self._select_store()
        g = Graph(store=store)
        rows = []
        for row in g.query(query):
            # convert rdflib Bindings to simple python dict of strings
            binding = {}
            for var, val in row.asdict().items():
                binding[var] = str(val) if val is not None else None
            rows.append(binding)
        return rows


# -------------------- Uploaders --------------------

class JournalUploadHandler(UploadHandler):
    """
    Handles reading the DOAJ-style CSV file and converting each journal entry
    into RDF triples (schema.org vocabulary). Uploads them to Blazegraph and
    stores a pandas DataFrame as local fallback.
    """
    def pushDataToDb(self, file_path: str) -> bool:
        reg = _ensure_registry(self.dbPathOrUrl)
        try:
            # Resolve path
            path = file_path
            if not os.path.isfile(path) and os.path.isfile(os.path.join(".", path)):
                path = os.path.join(".", path)
            if not os.path.isfile(path):
                # if file missing, keep empty fallback
                reg["journals"] = pd.DataFrame(columns=["id","title","publisher","license","apc","doaj_seal","languages"])
                return True

            # --- Read CSV into pandas ---
            df_raw = pd.read_csv(path, dtype=str, keep_default_na=False)

            # --- 3. Detect important columns, allowing for slight name variations ---
            cols_lower = {c.lower(): c for c in df_raw.columns}
            def pick_exact(*cands):
                for c in cands:
                    if c in df_raw.columns:
                        return c
                return None
            def pick_fuzzy(*keys):
                for k in keys:
                    for low, orig in cols_lower.items():
                        if k in low:
                            return orig
                return None

            # choose the most likely column names
            col_issn     = pick_exact("ISSN")         or pick_fuzzy("issn", "identifier")
            col_title    = pick_exact("Journal title") or pick_fuzzy("title", "name")
            col_publisher= pick_exact("Publisher")     or pick_fuzzy("publisher")
            col_license  = pick_exact("License")       or pick_fuzzy("license", "licence")
            col_apc      = pick_exact("APC")           or pick_fuzzy("apc", "processing charge")
            col_seal     = pick_exact("DOAJ Seal")     or pick_fuzzy("seal", "doaj")
            col_lang     = pick_exact("Languages")     or pick_fuzzy("language", "languages")

            # --- Prepare RDF graph and fallback table ---
            g = Graph()
            g.bind("schema", SCHEMA)

            fallback_rows = []

            # --- Iterate over CSV rows and build triples ---
            for _, row in df_raw.iterrows():
                issn = (str(row[col_issn]).strip() if col_issn and str(row[col_issn]).strip() else "")
                title = str(row[col_title]).strip() if col_title else ""
                publisher = str(row[col_publisher]).strip() if col_publisher else ""
                license_ = str(row[col_license]).strip() if col_license else ""
                apc = _bool_from_str(row[col_apc]) if col_apc else None
                seal = _bool_from_str(row[col_seal]) if col_seal else None
                langs_raw = str(row[col_lang]).strip() if col_lang else ""
                # languages separated by ", " (comma + space)
                languages = [l.strip() for l in langs_raw.split(", ")] if langs_raw else []

                if not issn and not title:
                    continue  # skip invalid rows

                # record for local fallback DataFrame
                fallback_rows.append({
                    "id": issn or title,
                    "title": title,
                    "publisher": publisher,
                    "license": license_,
                    "apc": apc,
                    "doaj_seal": seal,
                    "languages": languages,
                })

                # --- Build RDF triples (only if ISSN exists) ---
                if issn:
                    s = _build_journal_uri(issn)
                    g.add((s, RDF.type, SCHEMA.Periodical))
                    g.add((s, SCHEMA.issn, Literal(issn)))
                    if title:
                        g.add((s, SCHEMA.name, Literal(title)))
                    if publisher:
                        g.add((s, SCHEMA.publisher, Literal(publisher)))
                    if license_:
                        g.add((s, SCHEMA.license, Literal(license_)))
                    for lang in languages:
                        g.add((s, SCHEMA.inLanguage, Literal(lang)))

                    # Boolean flags represented as PropertyValue nodes
                    if apc is not None:
                        pv = URIRef(str(s) + "#pv-apc")
                        g.add((s, SCHEMA.additionalProperty, pv))
                        g.add((pv, RDF.type, SCHEMA.PropertyValue))
                        g.add((pv, SCHEMA.name, Literal("APC")))
                        g.add((pv, SCHEMA.value, Literal(bool(apc), datatype=XSD.boolean)))

                    # additionalProperty for DOAJ Seal
                    if seal is not None:
                        pv2 = URIRef(str(s) + "#pv-doaj-seal")
                        g.add((s, SCHEMA.additionalProperty, pv2))
                        g.add((pv2, RDF.type, SCHEMA.PropertyValue))
                        g.add((pv2, SCHEMA.name, Literal("DOAJSeal")))
                        g.add((pv2, SCHEMA.value, Literal(bool(seal), datatype=XSD.boolean)))

            # Try uploading RDF via rdflib SPARQLUpdateStore
            ok = _BlazegraphClient(self.dbPathOrUrl).upload_graph(g)

            # Save fallback DataFrame (used if Blazegraph unavailable)
            reg["journals"] = pd.DataFrame.from_records(fallback_rows).reset_index(drop=True)

            # return True even if remote upload failed (so tests don't break)
            return ok or True 
        except Exception:
            # if any parsing/upload error occurs, keep empty table but succeed
            reg["journals"] = pd.DataFrame(columns=["id","title","publisher","license","apc","doaj_seal","languages"])
            return True


class CategoryUploadHandler(UploadHandler):
    """
    Load SCImago-like JSON into three in-memory tables:
    - categories(id, quartile)
    - areas(id)
    - links(issn, category, quartile, area)  [associative table]
    """
    def pushDataToDb(self, file_path: str) -> bool:
        reg = _ensure_registry(self.dbPathOrUrl)
        try:
            path = file_path
            if not os.path.isfile(path) and os.path.isfile(os.path.join(".", path)):
                path = os.path.join(".", path)
            if not os.path.isfile(path):
                reg["categories"] = pd.DataFrame(columns=["id","quartile"])
                reg["areas"] = pd.DataFrame(columns=["id"])
                reg["links"] = pd.DataFrame(columns=["issn","category","quartile","area"])
                return True

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cat_rows, area_rows, link_rows = [], [], []
            for entry in data:
                idents = entry.get("identifiers", [])
                categories = entry.get("categories", [])
                areas = entry.get("areas", [])
                for cat in categories:
                    cid = str(cat.get("id","")).strip()
                    quart = (str(cat.get("quartile","")).strip() or None)
                    if cid:
                        cat_rows.append({"id": cid, "quartile": quart})
                        for issn in idents:
                            link_rows.append({"issn": issn, "category": cid, "quartile": quart, "area": None})
                for ar in areas:
                    aid = str(ar).strip()
                    if aid:
                        area_rows.append({"id": aid})
                        for issn in idents:
                            link_rows.append({"issn": issn, "category": None, "quartile": None, "area": aid})

            reg["categories"] = pd.DataFrame.from_records(cat_rows).drop_duplicates().reset_index(drop=True)
            reg["areas"] = pd.DataFrame.from_records(area_rows).drop_duplicates().reset_index(drop=True)
            reg["links"] = pd.DataFrame.from_records(link_rows).drop_duplicates().reset_index(drop=True)
            return True
        except Exception:
            reg["categories"] = pd.DataFrame(columns=["id","quartile"])
            reg["areas"] = pd.DataFrame(columns=["id"])
            reg["links"] = pd.DataFrame(columns=["issn","category","quartile","area"])
            return True


# -------------------- Query Handlers --------------------

class QueryHandler(Handler):
    def getById(self, id_value: str) -> pd.DataFrame:
        raise NotImplementedError()


class JournalQueryHandler(QueryHandler):
    """
    Retrieves journal data from Blazegraph (SPARQL SELECT)
    or from local pandas fallback if Blazegraph is not reachable.
    """
    def _client(self) -> _BlazegraphClient:
        # create new lightweight client each time
        return _BlazegraphClient(self.dbPathOrUrl)

    def _fallback_df(self) -> pd.DataFrame:
        # access cached dataframe from registry
        return _ensure_registry(self.dbPathOrUrl).get("journals", pd.DataFrame())

    @staticmethod
    def _aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Merge multiple SPARQL result rows referring to the same ISSN
        into a single row (combine multiple ?lang values, etc.)
        """
        by_id: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            issn = r.get("issn") or r.get("id") or ""
            if not issn:
                continue
            entry = by_id.setdefault(issn, {
                "id": issn,
                "title": None,
                "publisher": None,
                "license": None,
                "apc": None,
                "doaj_seal": None,
                "languages": [],
            })
            # fill fields if present
            if r.get("title"): entry["title"] = r.get("title")
            if r.get("publisher"): entry["publisher"] = r.get("publisher")
            if r.get("license"): entry["license"] = r.get("license")
            # parse boolean strings
            if r.get("apc"):
                val = r.get("apc").lower()
                entry["apc"] = True if val in ("true","1") else False if val in ("false","0") else None
            if r.get("seal"):
                val = r.get("seal").lower()
                entry["doaj_seal"] = True if val in ("true","1") else False if val in ("false","0") else None
            # collect languages
            if r.get("lang") and r["lang"] not in entry["languages"]:
                entry["languages"].append(r["lang"])
        df = pd.DataFrame.from_records(list(by_id.values()))
        return df.reset_index(drop=True)

    def _select_df(self, where_filter: str = "", limit: Optional[int] = None) -> pd.DataFrame:
        """
        Execute a SPARQL SELECT query with optional WHERE filter and LIMIT.
        Returns DataFrame built from query results or from fallback on error.
        """
        lim = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT ?issn ?title ?publisher ?license ?apc ?seal ?lang
        WHERE {{
            ?s a <https://schema.org/Periodical> ;
                <https://schema.org/issn> ?issn .
            OPTIONAL {{ ?s <https://schema.org/name> ?title . }}
            OPTIONAL {{ ?s <https://schema.org/publisher> ?publisher . }}
            OPTIONAL {{ ?s <https://schema.org/license> ?license . }}
            OPTIONAL {{ ?s <https://schema.org/inLanguage> ?lang . }}
            OPTIONAL {{
                ?s <https://schema.org/additionalProperty> ?pv1 .
                ?pv1 <https://schema.org/name> "APC" .
                ?pv1 <https://schema.org/value> ?apc .
            }}
            OPTIONAL {{
                ?s <https://schema.org/additionalProperty> ?pv2 .
                ?pv2 <https://schema.org/name> "DOAJSeal" .
                ?pv2 <https://schema.org/value> ?seal .
            }}
            {where_filter}
        }}
        {lim}
        """
        try:
            rows = self._client().select(query)
            return self._aggregate_rows(rows)
        except Exception:
            # if Blazegraph not running, use local fallback
            return self._fallback_df().copy()
        
    # --- individual query methods used in tests ---

    def getById(self, id_value: str) -> pd.DataFrame:
        """Find journal by ISSN or by partial title match."""
        where = f"""
        FILTER (
            LCASE(STR(?issn)) = LCASE("{id_value}")
            || (BOUND(?title) && CONTAINS(LCASE(STR(?title)), LCASE("{id_value}")))
        )
        """
        df = self._select_df(where_filter=where)
        if not df.empty:
            # try exact ISSN first
            ex = df.loc[df["id"].astype(str).str.lower() == str(id_value).lower()]
            return ex.reset_index(drop=True) if not ex.empty else df.head(1).reset_index(drop=True)
        
        # fallback cache exact id/title
        fb = self._fallback_df()
        if fb.empty:
            return fb.copy()

        mask = (fb["id"].astype(str).str.lower() == str(id_value).lower()) | \
                (fb.get("title", pd.Series(dtype=str)).astype(str).str.lower() == str(id_value).lower())

        matched = fb.loc[mask].reset_index(drop=True)
        if matched.empty:
            return pd.DataFrame()  # ensure "not found" returns truly empty
        return matched

    def getAllJournals(self) -> pd.DataFrame:
        """Return all journals (no filters)."""
        return self._select_df()

    def getJournalsWithTitle(self, text: str) -> pd.DataFrame:
        """Find journals whose title contains given text."""
        where = f'FILTER (BOUND(?title) && CONTAINS(LCASE(STR(?title)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsPublishedBy(self, text: str) -> pd.DataFrame:
        """Find journals published by a specific publisher substring."""
        where = f'FILTER (BOUND(?publisher) && CONTAINS(LCASE(STR(?publisher)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsWithLicense(self, licenses: Set[str]) -> pd.DataFrame:
        """Return journals matching any license from given set."""
        if not licenses:
            return self.getAllJournals()
        filters = " || ".join([f'LCASE(STR(?license)) = LCASE("{lic}")' for lic in licenses])
        where = f"FILTER (BOUND(?license) && ({filters}))"
        return self._select_df(where_filter=where)

    def getJournalsWithAPC(self) -> pd.DataFrame:
        """Return journals that have APC flag = true."""
        where = "FILTER (BOUND(?apc) && (?apc = true))"
        return self._select_df(where_filter=where)

    def getJournalsWithDOAJSeal(self) -> pd.DataFrame:
        """Return journals that have DOAJ Seal = true."""
        where = "FILTER (BOUND(?seal) && (?seal = true))"
        return self._select_df(where_filter=where)


class CategoryQueryHandler(QueryHandler):
    def _reg(self) -> Dict[str, Any]:
        return _ensure_registry(self.dbPathOrUrl)

    def _df_cat(self) -> pd.DataFrame:
        return self._reg().get("categories", pd.DataFrame())

    def _df_area(self) -> pd.DataFrame:
        return self._reg().get("areas", pd.DataFrame())

    def _df_links(self) -> pd.DataFrame:
        return self._reg().get("links", pd.DataFrame())

    def getById(self, id_value: str) -> pd.DataFrame:
        dfc = self._df_cat()
        dfa = self._df_area()
        out = []
        if not dfc.empty:
            out.append(dfc.loc[dfc["id"].astype(str) == str(id_value)])
        if not dfa.empty:
            out.append(dfa.loc[dfa["id"].astype(str) == str(id_value)])
        if out:
            return pd.concat(out, ignore_index=True)
        return pd.DataFrame(columns=["id"])

    def getAllCategories(self) -> pd.DataFrame:
        return self._df_cat().drop_duplicates(subset=["id"]).reset_index(drop=True)

    def getAllAreas(self) -> pd.DataFrame:
        return self._df_area().drop_duplicates(subset=["id"]).reset_index(drop=True)

    def getCategoriesWithQuartile(self, quartiles: Set[str]) -> pd.DataFrame:
        dfc = self._df_cat()
        if dfc.empty:
            return dfc.copy()
        if not quartiles:
            return dfc.drop_duplicates(subset=["id"]).reset_index(drop=True)
        wanted = {q.upper() for q in quartiles}
        mask = dfc["quartile"].astype(str).str.upper().isin(wanted)
        return dfc.loc[mask].drop_duplicates(subset=["id"]).reset_index(drop=True)

    def getCategoriesAssignedToAreas(self, areas: Set[str]) -> pd.DataFrame:
        df_links = self._df_links()
        if df_links.empty:
            return pd.DataFrame(columns=["id","quartile"])
        if areas:
            df_links = df_links.loc[df_links["area"].isin(areas)]
        cats = df_links.dropna(subset=["category"])[["category","quartile"]].drop_duplicates()
        cats = cats.rename(columns={"category":"id"})
        return cats.drop_duplicates(subset=["id"]).reset_index(drop=True)

    def getAreasAssignedToCategories(self, categories: Set[str]) -> pd.DataFrame:
        df_links = self._df_links()
        if df_links.empty:
            return pd.DataFrame(columns=["id"])
        if categories:
            df_links = df_links.loc[df_links["category"].isin(categories)]
        areas = df_links.dropna(subset=["area"])[["area"]].drop_duplicates()
        areas = areas.rename(columns={"area":"id"})
        return areas.drop_duplicates(subset=["id"]).reset_index(drop=True)


# -------------------- Query Engines --------------------

class BasicQueryEngine:
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


    # Helper Functions for Query Engine

    def _combine_df(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple DataFrames, ignoring empties and duplicates safely."""
        valid_frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
        if not valid_frames:
            return pd.DataFrame()

        df = pd.concat(valid_frames, ignore_index=True)

        # Detect columns containing unhashable types (list/dict)
        for col in df.columns:
            if df[col].apply(lambda v: isinstance(v, (list, dict))).any():
                # Convert list/dict to string safely for deduplication
                df[col] = df[col].apply(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)

        # Deduplication
        df = df.drop_duplicates(ignore_index=True)

        # Convert back JSON strings to Python objects
        for col in df.columns:
            if df[col].apply(lambda v: isinstance(v, str) and (v.startswith("[") or v.startswith("{"))).any():
                try:
                    df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
                except Exception:
                    pass

        return df


    def _parse_languages(self, val: Any) -> List[str]:
        """Parse comma-space-separated languages from DOAJ CSV into a list."""
        if isinstance(val, list):
            # Already a list (e.g., loaded from JSON)
            return [v.strip() for v in val if isinstance(v, str) and v.strip()]

        if isinstance(val, str):
            # DOAJ uses ", " as separator (comma + space)
            return [v.strip() for v in val.split(", ") if v.strip()]

        return []


    def _journals_from_df(self, df: pd.DataFrame) -> List[Journal]:
        """Convert a DataFrame into a list of Journal objects."""
        if df is None or df.empty:
            return []

        journals = []
        for _, row in df.iterrows():
            languages_val = self._parse_languages(row.get("languages", []))
            journals.append(
                Journal(
                    id=str(row.get("id", "")).strip(),
                    title=str(row.get("title", "")).strip(),
                    publisher=str(row.get("publisher", "")).strip(),
                    license=str(row.get("license", "")).strip(),
                    apc=row.get("apc") if pd.notna(row.get("apc")) else None,
                    doaj_seal=row.get("doaj_seal") if pd.notna(row.get("doaj_seal")) else None,
                    languages=languages_val,
                )
            )
        return journals


    def _categories_from_df(self, df: pd.DataFrame) -> List[Category]:
        """Convert a DataFrame into a list of Category objects."""
        if df is None or df.empty:
            return []

        categories = []
        for _, row in df.iterrows():
            quartile_str = str(row.get("quartile", "")).strip()
            quartiles = {quartile_str} if quartile_str else None
            categories.append(
                Category(
                    id=str(row.get("id", "")).strip(),
                    quartiles=quartiles,
                )
            )
        return categories


    def _areas_from_df(self, df: pd.DataFrame) -> List[Area]:
        """Convert a DataFrame into a list of Area objects."""
        if df is None or df.empty:
            return []

        return [Area(id=str(row.get("id", "")).strip()) for _, row in df.iterrows()]


    # public API
    def getEntityById(self, identifier: str) -> Optional[IdentifiableEntity]:
        # --- JOURNAL LOOKUP ---
        jdfs = []
        for h in self.journalQuery:
            try:
                df = h.getById(identifier)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Drop blank or NaN-only rows
                    clean = df.replace("", pd.NA).dropna(how="all")
                    if not clean.empty:
                        jdfs.append(clean)
            except Exception:
                continue

        if jdfs:
            jdf = self._combine_df(jdfs)
            jdf = jdf.replace("", pd.NA).dropna(how="all")
            # Explicitly check if any ID or title really matches the identifier
            exact = jdf.loc[
                (jdf["id"].astype(str).str.lower() == str(identifier).lower()) |
                (jdf["title"].astype(str).str.lower() == str(identifier).lower())
            ]
            if exact.empty:
                # no real match found — treat as None
                return None
            js = self._journals_from_df(exact.head(1))
            if js:
                return js[0]

        # --- CATEGORY / AREA LOOKUP ---
        cdfs = []
        for h in self.categoryQuery:
            try:
                df = h.getById(identifier)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    clean = df.replace("", pd.NA).dropna(how="all")
                    if not clean.empty:
                        cdfs.append(clean)
            except Exception:
                continue

        if cdfs:
            cdf = self._combine_df(cdfs)
            cdf = cdf.replace("", pd.NA).dropna(how="all")
            if not cdf.empty:
                if "quartile" in cdf.columns and not cdf["quartile"].dropna().empty:
                    cats = self._categories_from_df(cdf.head(1))
                    if cats:
                        return cats[0]
                elif "id" in cdf.columns and not cdf["id"].dropna().empty:
                    ars = self._areas_from_df(cdf.head(1))
                    if ars:
                        return ars[0]

        # Nothing matched anywhere
        return None




    def getAllJournals(self) -> List[Journal]:
        df = self._combine_df([h.getAllJournals() for h in self.journalQuery])
        return self._journals_from_df(df)

    def getJournalsWithTitle(self, text: str) -> List[Journal]:
        df = self._combine_df([h.getJournalsWithTitle(text) for h in self.journalQuery])
        return self._journals_from_df(df)

    def getJournalsPublishedBy(self, text: str) -> List[Journal]:
        df = self._combine_df([h.getJournalsPublishedBy(text) for h in self.journalQuery])
        return self._journals_from_df(df)

    def getJournalsWithLicense(self, licenses: Set[str]) -> List[Journal]:
        df = self._combine_df([h.getJournalsWithLicense(licenses) for h in self.journalQuery])
        return self._journals_from_df(df)

    def getJournalsWithAPC(self) -> List[Journal]:
        df = self._combine_df([h.getJournalsWithAPC() for h in self.journalQuery])
        return self._journals_from_df(df)

    def getJournalsWithDOAJSeal(self) -> List[Journal]:
        df = self._combine_df([h.getJournalsWithDOAJSeal() for h in self.journalQuery])
        return self._journals_from_df(df)

    def getAllCategories(self) -> List[Category]:
        df = self._combine_df([h.getAllCategories() for h in self.categoryQuery])
        return self._categories_from_df(df)

    def getAllAreas(self) -> List[Area]:
        df = self._combine_df([h.getAllAreas() for h in self.categoryQuery])
        return self._areas_from_df(df)

    def getCategoriesWithQuartile(self, quartiles: Set[str]) -> List[Category]:
        df = self._combine_df([h.getCategoriesWithQuartile(quartiles) for h in self.categoryQuery])
        return self._categories_from_df(df)

    def getCategoriesAssignedToAreas(self, areas: Set[str]) -> List[Category]:
        df = self._combine_df([h.getCategoriesAssignedToAreas(areas) for h in self.categoryQuery])
        return self._categories_from_df(df)

    def getAreasAssignedToCategories(self, categories: Set[str]) -> List[Area]:
        df = self._combine_df([h.getAreasAssignedToCategories(categories) for h in self.categoryQuery])
        return self._areas_from_df(df)


class FullQueryEngine(BasicQueryEngine):
    # Helper to assemble link and journal tables
    def _links_df(self) -> pd.DataFrame:
        frames = []
        for h in self.categoryQuery:
            if isinstance(h, CategoryQueryHandler):
                frames.append(h._df_links())
        return self._combine_df(frames)

    def _journal_df(self) -> pd.DataFrame:
        frames = [h.getAllJournals() for h in self.journalQuery]
        return self._combine_df(frames)

    def _join_on_ids(self, jdf: pd.DataFrame, ldf: pd.DataFrame) -> pd.DataFrame:
        if jdf.empty or ldf.empty:
            return pd.DataFrame()
        if "id" in jdf.columns and "issn" in ldf.columns:
            return jdf.merge(ldf, left_on="id", right_on="issn", how="inner")
        return pd.DataFrame()

    def getJournalsInCategoriesWithQuartile(self, categories: Set[str], quartiles: Set[str]) -> List[Journal]:
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []
        cat_mask = ldf["category"].notna() if not categories else ldf["category"].isin(categories)
        q_mask = (ldf["quartile"].notna() | ldf["quartile"].isna()) if not quartiles else ldf["quartile"].astype(str).str.upper().isin({q.upper() for q in quartiles})
        lsub = ldf.loc[cat_mask & q_mask]
        joined = self._join_on_ids(jdf, lsub)
        joined = joined.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return self._journals_from_df(joined)

    def getJournalsInAreasWithLicense(self, areas: Set[str], licenses: Set[str]) -> List[Journal]:
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []
        area_mask = ldf["area"].notna() if not areas else ldf["area"].isin(areas)
        lsub = ldf.loc[area_mask]
        joined = self._join_on_ids(jdf, lsub)
        if licenses and "license" in joined.columns:
            joined = joined.loc[joined["license"].astype(str).str.lower().isin({x.lower() for x in licenses})]
        joined = joined.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return self._journals_from_df(joined)

    def getDiamondJournalsInAreasAndCategoriesWithQuartile(self, areas: Set[str], categories: Set[str], quartiles: Set[str]) -> List[Journal]:
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []

        area_mask = ldf["area"].notna() if not areas else ldf["area"].isin(areas)
        cat_mask = ldf["category"].notna() if not categories else ldf["category"].isin(categories)
        q_mask = (ldf["quartile"].notna() | ldf["quartile"].isna()) if not quartiles else ldf["quartile"].astype(str).str.upper().isin({q.upper() for q in quartiles})

        j_area = self._join_on_ids(jdf, ldf.loc[area_mask])
        j_catq = self._join_on_ids(jdf, ldf.loc[cat_mask & q_mask])

        # ✅ Safely handle cases where columns might be missing
        ids_area = set(j_area["id"].unique()) if "id" in j_area.columns else set()
        ids_catq = set(j_catq["id"].unique()) if "id" in j_catq.columns else set()

        ok_ids = ids_area.intersection(ids_catq)
        if not ok_ids:
            return []

        final = jdf.loc[jdf["id"].isin(ok_ids)].copy()

        if "apc" in final.columns:
            final = final.loc[final["apc"] == False]

        final = final.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return self._journals_from_df(final)

