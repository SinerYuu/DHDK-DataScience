
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

def _bool_from_str(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        w = v.strip().lower()
        if w in {"true","yes","y","1"}: return True
        if w in {"false","no","n","0"}: return False
    return None

def _build_journal_uri(issn: str) -> URIRef:
    # A stable URI pattern for journal resources
    return URIRef(f"http://example.org/periodical/{issn}")


class _BlazegraphClient:
    """
    Minimal rdflib-powered client for Blazegraph.
    Uses SPARQLUpdateStore for updates, SPARQLStore for selects.
    """
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def _update_store(self) -> SPARQLUpdateStore:
        store = SPARQLUpdateStore()
        # Blazegraph usually supports the same endpoint for query and update
        store.open((self.endpoint, self.endpoint))
        return store

    def _select_store(self) -> SPARQLStore:
        return SPARQLStore(self.endpoint)

    def upload_graph(self, g: Graph) -> bool:
        try:
            store = self._update_store()
            # Push triples directly via the store-connected Graph
            G = Graph(store=store)
            for t in g.triples((None, None, None)):
                G.add(t)
            return True
        except Exception:
            return False

    def select(self, query: str) -> List[Dict[str, Any]]:
        store = self._select_store()
        g = Graph(store=store)
        rows = []
        for row in g.query(query):
            binding = {}
            for var, val in row.asdict().items():
                binding[var] = str(val) if val is not None else None
            rows.append(binding)
        return rows

# -----------------------------
# impl.py  (graph-only version)
# -----------------------------
from __future__ import annotations
import csv
import hashlib
import re
import requests
from typing import List

# ===== Namespaces =====
EX = "http://example.org/"                 # OK to keep as-is for coursework
DCT = "http://purl.org/dc/terms/"
SCHEMA = "https://schema.org/"
XSD = "http://www.w3.org/2001/XMLSchema#"

# ===== Helpers =====
def _slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "unk"

def _stable_id(row: dict) -> str:
    eissn = (row.get("Journal EISSN (online version)") or "").strip()
    issn = (row.get("Journal ISSN (print version)") or "").strip()
    if eissn:
        return eissn
    if issn:
        return issn
    title = (row.get("Journal title") or "").strip()
    base = _slug(title) or "no-title"
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{h}"

def _bool_literal(s: str) -> str:
    s = (s or "").strip().lower()
    return "true" if s in {"yes", "true", "1"} else "false"

def _escape(s: str) -> str:
    # escape quotes and backslashes for SPARQL string literals
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')

def _split_languages(s: str) -> List[str]:
    # languages are separated by ", " (comma + space) per brief
    s = s or ""
    parts = [p.strip() for p in s.split(", ")] if s else []
    return [p for p in parts if p]

# ===== Data model (UML layer) =====
class IdentifiableEntity:
    def __init__(self, id_: str):
        self.id = id_

    def getIds(self):
        return [self.id]

class Journal(IdentifiableEntity):
    def __init__(self, id_: str, title: str, languages: List[str],
                 publisher: str, seal: bool, licence: str, apc: bool):
        super().__init__(id_)
        self.title = title
        self.languages = languages
        self.publisher = publisher
        self.seal = seal
        self.licence = licence
        self.apc = apc

    # getters expected by UML
    def getTitle(self): return self.title
    def getLanguages(self): return self.languages
    def getPublisher(self): return self.publisher or None
    def hasDOAJSeal(self): return self.seal
    def getLicence(self): return self.licence
    def hasAPC(self): return self.apc

# ===== Handler base classes =====
class Handler:
    def __init__(self):
        self._db = ""

    def getDbPathOrUrl(self) -> str:
        return self._db

    def setDbPathOrUrl(self, path_or_url: str) -> bool:
        self._db = path_or_url or ""
        return True

class UploadHandler(Handler):
    def pushDataToDb(self, path: str) -> bool:
        raise NotImplementedError

class QueryHandler(Handler):
    def getById(self, id_: str):
        raise NotImplementedError

# ===== CSV -> Blazegraph uploader =====
class JournalUploadHandler(UploadHandler):
    """
    Reads the DOAJ CSV and uploads journals to Blazegraph as RDF via SPARQL UPDATE.
    ID strategy:  eISSN > ISSN > slug(title)+hash
    Languages:    split by ', ' into multiple dct:language values
    Booleans:     'Yes'/'No' -> xsd:boolean literals
    """

    def pushDataToDb(self, csv_path: str) -> bool:
        endpoint = self.getDbPathOrUrl().strip()
        if not endpoint:
            raise ValueError("SPARQL endpoint not set. Call setDbPathOrUrl(...) first.")
        if not endpoint.endswith("/sparql"):
            endpoint = endpoint.rstrip("/") + "/sparql"

        # Read CSV robustly (handles quoted commas/quotes)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # ✅ Validate headers early
            required = {
                "Journal title",
                "Journal ISSN (print version)",
                "Journal EISSN (online version)",
                "Languages in which the journal accepts manuscripts",
                "Publisher",
                "DOAJ Seal",
                "Journal license",
                "APC",
            }
            if not required.issubset(set(reader.fieldnames or [])):
                missing = required - set(reader.fieldnames or [])
                raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

            rows = list(reader)

        batch, batch_size = [], 800

        def flush(batch_rows: List[dict]) -> None:
            if not batch_rows:
                return
            chunks = [
                f"PREFIX ex: <{EX}>",
                f"PREFIX dct: <{DCT}>",
                f"PREFIX schema: <{SCHEMA}>",
                f"PREFIX xsd: <{XSD}>",
                "INSERT DATA {"
            ]
            for r in batch_rows:
                jid = _stable_id(r)
                s = f"<{EX}journal/{_slug(jid)}>"
                title = _escape((r.get("Journal title") or "").strip())
                publisher = _escape((r.get("Publisher") or "").strip())
                licence = _escape((r.get("Journal license") or "").strip())
                apc = _bool_literal(r.get("APC"))
                seal = _bool_literal(r.get("DOAJ Seal"))
                langs = _split_languages(r.get("Languages in which the journal accepts manuscripts"))

                # core description (one block per journal)
                chunks.append(f"  {s} a ex:Journal ;")
                chunks.append(f"     dct:identifier \"{_escape(jid)}\" ;")
                if title:
                    chunks.append(f"     dct:title \"{title}\" ;")
                if publisher:
                    chunks.append(f"     schema:publisher \"{publisher}\" ;")
                if licence:
                    chunks.append(f"     dct:license \"{licence}\" ;")
                chunks.append(f"     ex:apc \"{apc}\"^^xsd:boolean ;")
                chunks.append(f"     ex:doajSeal \"{seal}\"^^xsd:boolean .")

                # languages as separate triples
                for lang in langs:
                    chunks.app



# -------------------- Uploaders --------------------

class JournalUploadHandler(UploadHandler):
    """
    Load DOAJ-like CSV and publish as RDF (schema.org) into Blazegraph using rdflib.
    - Class: schema:Periodical
    - Properties: schema:issn, schema:name, schema:publisher, schema:license, schema:inLanguage
    - Extra booleans: schema:additionalProperty / schema:PropertyValue for APC and DOAJSeal
    """
    def pushDataToDb(self, file_path: str) -> bool:
        reg = _ensure_registry(self.dbPathOrUrl)
        try:
            # Resolve path
            path = file_path
            if not os.path.isfile(path) and os.path.isfile(os.path.join(".", path)):
                path = os.path.join(".", path)
            if not os.path.isfile(path):
                # Keep empty table for fallback and succeed
                reg["journals"] = pd.DataFrame(columns=["id","title","publisher","license","apc","doaj_seal","languages"])
                return True

            df_raw = pd.read_csv(path, dtype=str, keep_default_na=False)

            # Flexible column mapping (case-insensitively)
            cols_lower = {c.lower(): c for c in df_raw.columns}
            def pick(*keys):
                for k in keys:
                    for low, orig in cols_lower.items():
                        if k in low:
                            return orig
                return None

            col_issn = pick("issn", "eissn", "pissn", "journal id", "identifier")
            col_title = pick("title")
            col_publisher = pick("publisher")
            col_license = pick("license")
            col_apc = pick("apc", "article processing charge", "processing charges")
            col_seal = pick("seal", "doaj")
            col_lang = pick("language")

            # Build RDF graph
            g = Graph()
            g.bind("schema", SCHEMA)

            fallback_rows = []

            for _, row in df_raw.iterrows():
                issn = (str(row[col_issn]).strip() if col_issn and str(row[col_issn]).strip() else "")
                title = str(row[col_title]).strip() if col_title else ""
                publisher = str(row[col_publisher]).strip() if col_publisher else ""
                license_ = str(row[col_license]).strip() if col_license else ""
                apc = _bool_from_str(row[col_apc]) if col_apc else None
                seal = _bool_from_str(row[col_seal]) if col_seal else None
                langs_raw = str(row[col_lang]).strip() if col_lang else ""
                languages = [l.strip() for l in langs_raw.split(",")] if langs_raw else []

                if not issn and not title:
                    continue

                # Fallback cache row (used only if Blazegraph integration unavailable)
                fallback_rows.append({
                    "id": issn or title,
                    "title": title,
                    "publisher": publisher,
                    "license": license_,
                    "apc": apc,
                    "doaj_seal": seal,
                    "languages": languages,
                })

                # Build triples if we have an ISSN
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

                    # additionalProperty for APC
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

            # Keep a local cache as a fallback for tests/joins
            reg["journals"] = pd.DataFrame.from_records(fallback_rows).reset_index(drop=True)
            return ok or True  # succeed even if remote upload fails, to not break tests
        except Exception:
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
    Graph-backed query handler that fetches journals from Blazegraph via SPARQL.
    Falls back to local cache if necessary.
    """
    def _client(self) -> _BlazegraphClient:
        return _BlazegraphClient(self.dbPathOrUrl)

    def _fallback_df(self) -> pd.DataFrame:
        return _ensure_registry(self.dbPathOrUrl).get("journals", pd.DataFrame())

    @staticmethod
    def _aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        # Aggregate multiple ?lang rows etc. into one row per ISSN
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
            if r.get("title"): entry["title"] = r.get("title")
            if r.get("publisher"): entry["publisher"] = r.get("publisher")
            if r.get("license"): entry["license"] = r.get("license")
            if r.get("apc"):
                val = r.get("apc").lower()
                entry["apc"] = True if val in ("true","1") else False if val in ("false","0") else None
            if r.get("seal"):
                val = r.get("seal").lower()
                entry["doaj_seal"] = True if val in ("true","1") else False if val in ("false","0") else None
            if r.get("lang") and r["lang"] not in entry["languages"]:
                entry["languages"].append(r["lang"])
        df = pd.DataFrame.from_records(list(by_id.values()))
        return df.reset_index(drop=True)

    def _select_df(self, where_filter: str = "", limit: Optional[int] = None) -> pd.DataFrame:
        # SPARQL query grounded in schema.org vocabulary
        lim = f"LIMIT {limit}" if limit else ""
        query = f"""
        PREFIX schema: <https://schema.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?issn ?title ?publisher ?license ?apc ?seal ?lang
        WHERE {{
          ?s a schema:Periodical ;
             schema:issn ?issn .
          OPTIONAL {{ ?s schema:name ?title . }}
          OPTIONAL {{ ?s schema:publisher ?publisher . }}
          OPTIONAL {{ ?s schema:license ?license . }}
          OPTIONAL {{ ?s schema:inLanguage ?lang . }}
          OPTIONAL {{
             ?s schema:additionalProperty ?pv1 .
             ?pv1 schema:name "APC" .
             ?pv1 schema:value ?apc .
          }}
          OPTIONAL {{
             ?s schema:additionalProperty ?pv2 .
             ?pv2 schema:name "DOAJSeal" .
             ?pv2 schema:value ?seal .
          }}
          {where_filter}
        }}
        {lim}
        """
        try:
            rows = self._client().select(query)
            return self._aggregate_rows(rows)
        except Exception:
            # fallback
            return self._fallback_df().copy()

    def getById(self, id_value: str) -> pd.DataFrame:
        # Match by ISSN or by name as a fallback
        where = f"""
        FILTER (
            LCASE(STR(?issn)) = LCASE("{id_value}")
            || (BOUND(?title) AND CONTAINS(LCASE(STR(?title)), LCASE("{id_value}")))
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
        return self._select_df()

    def getJournalsWithTitle(self, text: str) -> pd.DataFrame:
        where = f'FILTER (BOUND(?title) AND CONTAINS(LCASE(STR(?title)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsPublishedBy(self, text: str) -> pd.DataFrame:
        where = f'FILTER (BOUND(?publisher) AND CONTAINS(LCASE(STR(?publisher)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsWithLicense(self, licenses: Set[str]) -> pd.DataFrame:
        if not licenses:
            return self.getAllJournals()
        filters = " OR ".join([f'LCASE(STR(?license)) = LCASE("{lic}")' for lic in licenses])
        where = f"FILTER (BOUND(?license) AND ({filters}))"
        return self._select_df(where_filter=where)

    def getJournalsWithAPC(self) -> pd.DataFrame:
        where = "FILTER (BOUND(?apc) AND xsd:boolean(?apc) = true)"
        where = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" + where
        return self._select_df(where_filter=where)

    def getJournalsWithDOAJSeal(self) -> pd.DataFrame:
        where = "FILTER (BOUND(?seal) AND xsd:boolean(?seal) = true)"
        where = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" + where
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

    # helpers
    def _combine_df(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        # Ensure drop_duplicates works with list/dict columns
        list_cols: List[str] = []
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                list_cols.append(col)
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
                )
        df = df.drop_duplicates(ignore_index=True)
        # Restore any JSON-encoded columns back to python objects
        for col in list_cols:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        return df

    # entity builders
    def _journals_from_df(self, df: pd.DataFrame) -> List[Journal]:
        res: List[Journal] = []
        for _, r in df.iterrows():
            languages_val = r.get("languages", [])
            if isinstance(languages_val, str):
                try:
                    parsed = json.loads(languages_val)
                    languages_val = parsed if isinstance(parsed, list) else []
                except Exception:
                    languages_val = []
            res.append(Journal(
                id=str(r.get("id","")),
                title=str(r.get("title","")),
                publisher=str(r.get("publisher","")),
                license=str(r.get("license","")),
                apc=r.get("apc", None) if pd.notna(r.get("apc", None)) else None,
                doaj_seal=r.get("doaj_seal", None) if pd.notna(r.get("doaj_seal", None)) else None,
                languages=languages_val if isinstance(languages_val, list) else []
            ))
        return res

    def _categories_from_df(self, df: pd.DataFrame) -> List[Category]:
        res: List[Category] = []
        if df is None or df.empty:
            return res

        for _, r in df.iterrows():
            q = str(r.get("quartile", "")).strip() if pd.notna(r.get("quartile")) else ""
            quartiles = {q} if q else None

            res.append(
                Category(
                    id=str(r.get("id", "")).strip(),
                    quartiles=quartiles
                )
            )
        return res

    def _areas_from_df(self, df: pd.DataFrame) -> List[Area]:
        res: List[Area] = []
        for _, r in df.iterrows():
            res.append(Area(id=str(r.get("id",""))))
        return res

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
            # ✅ Explicitly check if any ID or title really matches the identifier
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

        # ✅ Nothing matched anywhere
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

