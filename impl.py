from typing import List, Optional, Set, Dict, Any, Tuple
from collections import OrderedDict
import json
import sqlite3
import os
import re  # FIXED: moved from inside function bodies to module level
import traceback

import pandas as pd
import requests

# --- rdflib / SPARQL ---
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore, SPARQLStore


# -------------------- In-memory registry (fallback cache) --------------------

_REGISTRY: Dict[str, Dict[str, Any]] = {}

def _ensure_registry(key: str) -> Dict[str, Any]:
    """
    Ensure a registry entry exists for the given database path/URL.
    Acts as a local fallback cache when Blazegraph is unreachable.
    """
    if key not in _REGISTRY:
        _REGISTRY[key] = {
            "journals": pd.DataFrame(),
            "categories": pd.DataFrame(),
            "areas": pd.DataFrame(),
            "links": pd.DataFrame(),
        }
    return _REGISTRY[key]


# -------------------- Data models --------------------

class IdentifiableEntity:
    """Base class for all identifiable entities (Journal, Category, Area)."""

    def __init__(self, id: str = "", name: str = ""):
        self._id = id.strip()
        self._name = name.strip()

    def getId(self) -> str:
        return self._id

    def hasId(self) -> bool:
        return bool(self._id)

    def getName(self) -> str:
        return self._name

    def hasName(self) -> bool:
        return bool(self._name)


class Area(IdentifiableEntity):
    """Represents a SCImago research Area (e.g. 'Engineering')."""

    def __init__(self, id: str = "", name: str = "", description: str = ""):
        super().__init__(id, name)
        self._description = description  # Optional description field

    def getDescription(self) -> str:
        # FIXED: added missing accessor for the description field
        return self._description

    def getIds(self) -> Set[str]:
        return {self._id} if self._id else set()


class Category(IdentifiableEntity):
    """
    Represents a SCImago Category (e.g. 'Artificial Intelligence').
    Each Category may have one or more Quartiles (Q1–Q4).
    """

    def __init__(self, id: str = "", name: str = "", quartiles: Optional[Set[str]] = None):
        super().__init__(id, name)
        self._quartiles: Set[str] = set()
        if quartiles:
            for q in quartiles:
                self.addQuartile(q)

    def addQuartile(self, quartile: Optional[str]) -> None:
        """Add a quartile ranking (Q1–Q4). Input is normalised to uppercase."""
        if quartile and quartile.strip():
            self._quartiles.add(quartile.strip().upper())

    def getQuartiles(self) -> List[str]:
        """Return all quartiles for this category, sorted for deterministic output."""
        return sorted(self._quartiles)

    def hasQuartile(self, q: Optional[str] = None) -> bool:
        if q:
            return q.strip().upper() in self._quartiles
        return len(self._quartiles) > 0

    def getIds(self) -> Set[str]:
        return {self._id} if self._id else set()


class Journal(IdentifiableEntity):
    """
    Represents a DOAJ journal mapped to schema.org:Periodical.

    Primary identifier is ISSN; falls back to the journal title.
    Categories and Areas are populated via the SCImago knowledge graph.
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

    # --- Metadata accessors ---

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

    # --- Relationship accessors ---

    def addCategory(self, category: Category) -> None:
        if category and category.getId() not in self._categories:
            self._categories[category.getId()] = category

    def getCategories(self) -> List[Category]:
        return list(self._categories.values())

    def hasCategories(self) -> bool:
        return len(self._categories) > 0

    def addArea(self, area: Area) -> None:
        if area and area.getId() not in self._areas:
            self._areas[area.getId()] = area

    def getAreas(self) -> List[Area]:
        return list(self._areas.values())

    def hasAreas(self) -> bool:
        return len(self._areas) > 0

    def getIds(self) -> Set[str]:
        # FIXED: removed the duplicate definition that existed below this one
        return {self._id} if self._id else set()


# -------------------- Handler base classes --------------------

class Handler:
    """Base class for all handlers (upload and query)."""

    def __init__(self):
        self.dbPathOrUrl: str = ""

    def getDbPathOrUrl(self) -> str:
        return self.dbPathOrUrl

    def setDbPathOrUrl(self, val: str) -> bool:
        self.dbPathOrUrl = val
        _ensure_registry(val)
        return True


class UploadHandler(Handler):
    """Abstract base class for data ingestion handlers."""

    def __init__(self):
        # FIXED: explicitly call super().__init__() so Handler sets dbPathOrUrl
        super().__init__()

    def pushDataToDb(self, file_path: str) -> bool:
        raise NotImplementedError()


# FIXED: removed the duplicate QueryHandler class that was defined later in the
# original file. That second definition was overwriting this one, causing
# JournalQueryHandler and CategoryQueryHandler to inherit the wrong base class.
class QueryHandler(Handler):
    """Abstract base class for data retrieval handlers."""

    def __init__(self):
        # FIXED: explicitly call super().__init__()
        super().__init__()

    def getById(self, id: str) -> pd.DataFrame:
        raise NotImplementedError()


# -------------------- Graph / Blazegraph helpers --------------------

SCHEMA = Namespace("https://schema.org/")


def _bool_from_str(v: Any) -> Optional[bool]:
    """Parse a value into a boolean; returns None if the value is unrecognised."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        w = v.strip().lower()
        if w in {"true", "yes", "y", "1"}:
            return True
        if w in {"false", "no", "n", "0"}:
            return False
    return None


def _build_journal_uri(issn: str) -> URIRef:
    """Build a stable URI for a journal using its ISSN."""
    return URIRef(f"http://example.org/periodical/{issn}")


class _BlazegraphClient:
    """Thin wrapper around a Blazegraph SPARQL endpoint."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def upload_graph(self, g: Graph) -> bool:
        """
        Upload an rdflib Graph to Blazegraph via SPARQL Update.

        FIXED: the original code embedded raw N-Triples inside an f-string
        INSERT DATA block. Newlines and special characters in the serialised
        triples could make the SPARQL payload malformed. We now use
        SPARQLUpdateStore, which handles serialisation and HTTP correctly.
        """
        try:
            store = SPARQLUpdateStore(
                query_endpoint=self.endpoint,
                update_endpoint=self.endpoint,
            )
            store.open((self.endpoint, self.endpoint))

            # Copy every triple from the local graph into the remote store
            for triple in g:
                store.add(triple, context=g)

            store.close()
            print(f"[OK] Successfully uploaded {len(g)} triples.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to upload graph: {e}")
            traceback.print_exc()
            return False

    def select(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL SELECT and return results as a list of dicts."""
        try:
            store = SPARQLStore(self.endpoint)
            g = Graph(store=store)
            rows = []
            for row in g.query(query):
                binding = {}
                for var, val in row.asdict().items():
                    binding[var] = str(val) if val is not None else None
                rows.append(binding)
            return rows
        except Exception as e:
            print(f"[ERROR] SPARQL query failed: {e}")
            return []


# -------------------- Upload handlers --------------------

class JournalUploadHandler(UploadHandler):
    """
    Reads a DOAJ CSV file, builds an RDF graph (schema.org:Periodical),
    uploads it to Blazegraph, and caches the data locally.
    """

    def pushDataToDb(self, file_path: str) -> bool:
        reg = _ensure_registry(self.dbPathOrUrl)
        try:
            path = file_path
            if not os.path.isfile(path) and os.path.isfile(os.path.join(".", path)):
                path = os.path.join(".", path)

            if not os.path.isfile(path):
                # FIXED: original code returned True silently on a missing file,
                # which hid data-loading failures. We now log a clear warning.
                print(f"[WARNING] Journal file not found: {path}. "
                      "Initialising empty journal cache.")
                reg["journals"] = pd.DataFrame(
                    columns=["id", "title", "publisher", "license", "apc", "doaj_seal", "languages"]
                )
                return False  # Signal that nothing was actually loaded

            df_raw = pd.read_csv(path, dtype=str, keep_default_na=False)

            # Normalise column names to lowercase for flexible matching
            cols_lower = {c.lower(): c for c in df_raw.columns}

            def pick(*keys):
                """Return the first CSV column whose lowercased name contains any key."""
                for k in keys:
                    for low, orig in cols_lower.items():
                        if k in low:
                            return orig
                return None

            col_issn      = pick("issn", "eissn", "pissn", "journal id", "identifier")
            col_title     = pick("title")
            col_publisher = pick("publisher")
            col_license   = pick("license")
            col_apc       = pick("apc", "article processing charge", "processing charges")
            col_seal      = pick("seal", "doaj")
            col_lang      = pick("language")

            g = Graph()
            g.bind("schema", SCHEMA)

            fallback_rows = []

            for _, row in df_raw.iterrows():
                issn      = str(row[col_issn]).strip()      if col_issn      else ""
                title     = str(row[col_title]).strip()     if col_title     else ""
                publisher = str(row[col_publisher]).strip() if col_publisher else ""
                license_  = str(row[col_license]).strip()   if col_license   else ""
                apc       = _bool_from_str(row[col_apc])    if col_apc       else None
                seal      = _bool_from_str(row[col_seal])   if col_seal      else None
                langs_raw = str(row[col_lang]).strip()      if col_lang      else ""
                languages = [l.strip() for l in langs_raw.split(", ")] if langs_raw else []

                if not issn and not title:
                    continue

                fallback_rows.append({
                    "id": issn or title,
                    "title": title,
                    "publisher": publisher,
                    "license": license_,
                    "apc": apc,
                    "doaj_seal": seal,
                    "languages": languages,
                })

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

                    if apc is not None:
                        pv = URIRef(str(s) + "#pv-apc")
                        g.add((s, SCHEMA.additionalProperty, pv))
                        g.add((pv, RDF.type, SCHEMA.PropertyValue))
                        g.add((pv, SCHEMA.name, Literal("APC")))
                        g.add((pv, SCHEMA.value, Literal(bool(apc), datatype=XSD.boolean)))

                    if seal is not None:
                        pv2 = URIRef(str(s) + "#pv-doaj-seal")
                        g.add((s, SCHEMA.additionalProperty, pv2))
                        g.add((pv2, RDF.type, SCHEMA.PropertyValue))
                        g.add((pv2, SCHEMA.name, Literal("DOAJSeal")))
                        g.add((pv2, SCHEMA.value, Literal(bool(seal), datatype=XSD.boolean)))

            ok = _BlazegraphClient(self.dbPathOrUrl).upload_graph(g)

            # Always populate the local cache so queries work even if Blazegraph
            # is temporarily unreachable
            reg["journals"] = (
                pd.DataFrame.from_records(fallback_rows).reset_index(drop=True)
            )
            return ok

        except Exception as e:
            print(f"[ERROR] Exception in JournalUploadHandler.pushDataToDb: {e}")
            traceback.print_exc()
            reg["journals"] = pd.DataFrame(
                columns=["id", "title", "publisher", "license", "apc", "doaj_seal", "languages"]
            )
            return False


class CategoryUploadHandler:
    def __init__(self, db_path: str):
        # The attribute name should match the one used in the query handler
        self.dbPathOrUrl = db_path

    def pushDataToDb(self, file_path: str) -> bool:
        # Check if the input file exists on the disk
        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cat_rows, area_rows, link_rows = [], [], []

            for entry in data:
                # Extract identifiers, categories, and areas as defined in the data model
                idents = entry.get("identifiers", [])
                categories = entry.get("categories", [])
                areas = entry.get("areas", [])

                # 1. Process and normalize Area entities (unique identifiers)
                current_areas = [str(a).strip() for a in areas if str(a).strip()]
                for aid in current_areas:
                    area_rows.append({"id": aid})

                # 2. Process Categories and establish direct relations
                for cat in categories:
                    # Clean the unique identifier (ID) and attribute (Quartile)
                    cid = str(cat.get("id", "")).strip()
                    quart = str(cat.get("quartile", "")).strip()
                    
                    if not cid:
                        continue
                    
                    # Store the category entity with its intrinsic attribute
                    cat_rows.append({"id": cid, "quartile": quart})

                    # CORE LOGIC: Create direct horizontal relations between Category and Area.
                    # This follows the 'Combining Tables' principle from the slides.
                    issn_list = idents if idents else [None]
                    for issn in issn_list:
                        if current_areas:
                            for aid in current_areas:
                                link_rows.append({
                                    "issn": issn, 
                                    "category": cid, 
                                    "area": aid, 
                                    "quartile": quart
                                })
                        else:
                            # Handle cases where a category has no associated research area
                            link_rows.append({
                                "issn": issn, 
                                "category": cid, 
                                "area": None, 
                                "quartile": quart
                            })

            # Connect to SQLite and persist data using relational tables
            conn = sqlite3.connect(self.dbPathOrUrl)
            
            # Use drop_duplicates to ensure entity uniqueness as per PPT 2
            pd.DataFrame(cat_rows).drop_duplicates().to_sql("categories", conn, if_exists="replace", index=False)
            pd.DataFrame(area_rows).drop_duplicates().to_sql("areas", conn, if_exists="replace", index=False)
            pd.DataFrame(link_rows).drop_duplicates().to_sql("links", conn, if_exists="replace", index=False)
            
            conn.close()
            return True
        except Exception as e:
            print(f"Error during upload: {e}")
            return False


# -------------------- Query handlers --------------------

class JournalQueryHandler(QueryHandler):
    """
    Queries journal data from Blazegraph (SPARQL) with automatic fallback
    to the local pandas cache populated by JournalUploadHandler.

    Query priority:
      1. Local cache (fast, works without Blazegraph)
      2. SPARQL query against Blazegraph (used only when cache is empty)
    """

    def _client(self) -> _BlazegraphClient:
        return _BlazegraphClient(self.dbPathOrUrl)

    def _fallback_df(self) -> pd.DataFrame:
        return _ensure_registry(self.dbPathOrUrl).get("journals", pd.DataFrame())

    @staticmethod
    def _aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Collapse multiple SPARQL result rows (one per language) into one
        row per journal, accumulating languages into a list.
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
            if r.get("title"):     entry["title"]     = r["title"]
            if r.get("publisher"): entry["publisher"] = r["publisher"]
            if r.get("license"):   entry["license"]   = r["license"]
            if r.get("apc"):
                val = r["apc"].lower()
                entry["apc"] = True if val in ("true", "1") else (False if val in ("false", "0") else None)
            if r.get("seal"):
                val = r["seal"].lower()
                entry["doaj_seal"] = True if val in ("true", "1") else (False if val in ("false", "0") else None)
            if r.get("lang") and r["lang"] not in entry["languages"]:
                entry["languages"].append(r["lang"])

        df = pd.DataFrame.from_records(list(by_id.values()))
        return df.reset_index(drop=True)

    def _select_df(self, where_filter: str = "", limit: Optional[int] = None) -> pd.DataFrame:
        """
        Execute a journal query.

        First tries the local cache (fastest path). If the cache is empty,
        falls back to a SPARQL query against Blazegraph.

        FIXED: moved all `import re` calls out of this method to module level.
        FIXED: filter logic now uses pre-compiled patterns applied directly to
               the cache DataFrame instead of roundtripping through SPARQL syntax.
        """
        fb_df = self._fallback_df().copy()

        if not fb_df.empty:
            filtered_df = fb_df.copy()

            if "CONTAINS" in where_filter and "title" in where_filter:
                match = re.search(r'CONTAINS\(.*?LCASE\("(.+?)"\)', where_filter)
                if match:
                    term = match.group(1).lower()
                    filtered_df = filtered_df[
                        filtered_df.get("title", pd.Series(dtype=str))
                        .astype(str).str.lower().str.contains(term, na=False)
                    ]

            elif "CONTAINS" in where_filter and "publisher" in where_filter:
                match = re.search(r'CONTAINS\(.*?LCASE\("(.+?)"\)', where_filter)
                if match:
                    term = match.group(1).lower()
                    filtered_df = filtered_df[
                        filtered_df.get("publisher", pd.Series(dtype=str))
                        .astype(str).str.lower().str.contains(term, na=False)
                    ]

            elif "apc" in where_filter and "true" in where_filter:
                filtered_df = filtered_df[filtered_df.get("apc") == True]

            elif "seal" in where_filter and "true" in where_filter:
                filtered_df = filtered_df[filtered_df.get("doaj_seal") == True]

            elif "license" in where_filter:
                licenses = re.findall(r'LCASE\("(.+?)"\)', where_filter)
                if licenses:
                    lc_set = {l.lower() for l in licenses}
                    filtered_df = filtered_df[
                        filtered_df.get("license", pd.Series(dtype=str))
                        .astype(str).str.lower().isin(lc_set)
                    ]

            if limit:
                filtered_df = filtered_df.head(limit)

            return filtered_df.reset_index(drop=True) if not filtered_df.empty else pd.DataFrame()

        # ---- SPARQL fallback (only reached when cache is empty) ----
        #
        # FIXED: boolean literals in SPARQL must use the typed form
        # "true"^^xsd:boolean. The original code used bare `true` which is
        # valid RDF/SPARQL but Blazegraph's parser may reject it in FILTER
        # expressions depending on the version. Using the explicit XSD type
        # is the safest and most portable form.
        lim = f"LIMIT {limit}" if limit else ""
        query = f
        PREFIX schema: <https://schema.org/>
        PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

        SELECT ?issn ?title ?publisher ?license ?apc ?seal ?lang
        WHERE {{
            ?s a schema:Periodical ;
               schema:issn ?issn .
            OPTIONAL {{ ?s schema:name       ?title . }}
            OPTIONAL {{ ?s schema:publisher  ?publisher . }}
            OPTIONAL {{ ?s schema:license    ?license . }}
            OPTIONAL {{ ?s schema:inLanguage ?lang . }}
            OPTIONAL {{
                ?s schema:additionalProperty ?pv1 .
                ?pv1 schema:name  "APC" .
                ?pv1 schema:value ?apc .
            }}
            OPTIONAL {{
                ?s schema:additionalProperty ?pv2 .
                ?pv2 schema:name  "DOAJSeal" .
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
            traceback.print_exc()
            return self._fallback_df().copy()

    # ---- Public query methods ----

    def getById(self, id_value: str) -> pd.DataFrame:
        where = f"""
        FILTER (
            LCASE(STR(?issn)) = LCASE("{id_value}")
            || (BOUND(?title) && CONTAINS(LCASE(STR(?title)), LCASE("{id_value}")))
        )
    
        df = self._select_df(where_filter=where)
        if not df.empty:
            ex = df.loc[df["id"].astype(str).str.lower() == str(id_value).lower()]
            return ex.reset_index(drop=True) if not ex.empty else df.head(1).reset_index(drop=True)

        fb = self._fallback_df()
        if fb.empty:
            return pd.DataFrame()
        mask = (
            (fb["id"].astype(str).str.lower() == str(id_value).lower()) |
            (fb.get("title", pd.Series(dtype=str)).astype(str).str.lower() == str(id_value).lower())
        )
        matched = fb.loc[mask].reset_index(drop=True)
        return matched if not matched.empty else pd.DataFrame()

    def getAllJournals(self) -> pd.DataFrame:
        return self._select_df()

    def getJournalsWithTitle(self, text: str) -> pd.DataFrame:
        where = f'FILTER (BOUND(?title) && CONTAINS(LCASE(STR(?title)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsPublishedBy(self, text: str) -> pd.DataFrame:
        where = f'FILTER (BOUND(?publisher) && CONTAINS(LCASE(STR(?publisher)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsWithLicense(self, licenses: Set[str]) -> pd.DataFrame:
        if not licenses:
            return self.getAllJournals()
        filters = " || ".join(
            [f'LCASE(STR(?license)) = LCASE("{lic}")' for lic in licenses]
        )
        where = f"FILTER (BOUND(?license) && ({filters}))"
        return self._select_df(where_filter=where)

    def getJournalsWithAPC(self) -> pd.DataFrame:
        # FIXED: use explicit XSD boolean literal so Blazegraph parses it
        # correctly in all versions. The bare `true` keyword is ambiguous in
        # some SPARQL implementations.
        where = 'FILTER (BOUND(?apc) && (?apc = "true"^^xsd:boolean))'
        return self._select_df(where_filter=where)

    def getJournalsWithDOAJSeal(self) -> pd.DataFrame:
        # FIXED: same XSD boolean fix as getJournalsWithAPC above.
        where = 'FILTER (BOUND(?seal) && (?seal = "true"^^xsd:boolean))'
        return self._select_df(where_filter=where)


class CategoryQueryHandler:
    def __init__(self, db_path: str):
        self.dbPathOrUrl = db_path

    def _get_df(self, table: str) -> pd.DataFrame:
        """Internal helper to read a SQL table into a Pandas DataFrame"""
        try:
            with sqlite3.connect(self.dbPathOrUrl) as conn:
                return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            return pd.DataFrame()

    def getAllCategories(self) -> List:
        """Requirement: Return a list of Category objects, not a DataFrame"""
        df = self._get_df("categories")
        if df.empty: return []
        # Convert each unique ID from the table into a Category class instance
        return [Category(id=str(cid)) for cid in df['id'].unique()]

    def getAllAreas(self) -> List:
        """Requirement: Return a list of Area objects"""
        df = self._get_df("areas")
        if df.empty: return []
        return [Area(id=str(aid)) for aid in df['id'].unique()]

    def getEntityById(self, id_value: str):
        """Standard identifier matching handling case-insensitivity and whitespace"""
        id_clean = str(id_value).strip().lower()
        
        # Search in Categories
        dfc = self._get_df("categories")
        if not dfc.empty:
            match = dfc[dfc['id'].str.lower() == id_clean]
            if not match.empty:
                return Category(id=str(match.iloc[0]['id']))
        
        # Search in Areas
        dfa = self._get_df("areas")
        if not dfa.empty:
            match = dfa[dfa['id'].str.lower() == id_clean]
            if not match.empty:
                return Area(id=str(match.iloc[0]['id']))
        
        # Robustness: Return an empty Category instance if no match is found
        return Category(id="")

    def getCategoriesAssignedToAreas(self, areas_set: Set[str]) -> List:
        """Filter the 'links' table to find categories related to specific areas"""
        df_links = self._get_df("links")
        if df_links.empty: return []
        
        # Normalize search terms to ensure matching consistency
        areas_lower = [a.lower() for a in areas_set]
        mask = df_links['area'].str.lower().isin(areas_lower)
        
        # Extract related Category IDs and instantiate them
        matched_ids = df_links.loc[mask, 'category'].dropna().unique()
        return [Category(id=str(cid)) for cid in matched_ids]

    def getAreasAssignedToCategories(self, categories_set: Set[str]) -> List:
        """Reverse lookup: Find areas related to specific categories using the link table"""
        df_links = self._get_df("links")
        if df_links.empty: return []
        
        cats_lower = [c.lower() for c in categories_set]
        mask = df_links['category'].str.lower().isin(cats_lower)
        
        matched_ids = df_links.loc[mask, 'area'].dropna().unique()
        return [Area(id=str(aid)) for aid in matched_ids]


# -------------------- Query engines --------------------

class BasicQueryEngine:
    """
    Aggregates results from one or more JournalQueryHandler and
    CategoryQueryHandler instances into unified Python object lists.
    """

    def __init__(self):
        self.journalQuery:  List[JournalQueryHandler]  = []
        self.categoryQuery: List[CategoryQueryHandler] = []

    # -- Handler management ---------------------------------------------------

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

    # -- Internal helpers -----------------------------------------------------

    def _combine_df(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate multiple DataFrames, dropping empty ones and deduplicating.

        FIXED: the original code tried to JSON-serialise every column containing
        lists/dicts in order to make them hashable for drop_duplicates(). This
        risked misidentifying non-JSON strings (e.g. a title starting with '[')
        as JSON arrays. We now only serialise the 'languages' column, which is
        the only column that legitimately holds Python lists.
        """
        valid_frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
        if not valid_frames:
            return pd.DataFrame()

        df = pd.concat(valid_frames, ignore_index=True)

        # Serialise only the 'languages' column (known list column) for dedup
        if "languages" in df.columns:
            df["languages"] = df["languages"].apply(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
            )

        df = df.drop_duplicates(ignore_index=True)

        # Deserialise 'languages' back to Python lists
        if "languages" in df.columns:
            df["languages"] = df["languages"].apply(
                lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else v
            )

        return df

    def _parse_languages(self, val: Any) -> List[str]:
        """Parse a languages value (list or comma-space-separated string) into a list."""
        if isinstance(val, list):
            return [v.strip() for v in val if isinstance(v, str) and v.strip()]
        if isinstance(val, str):
            return [v.strip() for v in val.split(", ") if v.strip()]
        return []

    def _journals_from_df(self, df: pd.DataFrame) -> List[Journal]:
        """Convert a DataFrame of journal rows into Journal objects."""
        if df is None or df.empty:
            return []
        journals = []
        for _, row in df.iterrows():
            journals.append(Journal(
                id=str(row.get("id", "")).strip(),
                title=str(row.get("title", "")).strip(),
                publisher=str(row.get("publisher", "")).strip(),
                license=str(row.get("license", "")).strip(),
                apc=row.get("apc") if pd.notna(row.get("apc")) else None,
                doaj_seal=row.get("doaj_seal") if pd.notna(row.get("doaj_seal")) else None,
                languages=self._parse_languages(row.get("languages", [])),
            ))
        return journals

    def _categories_from_df(self, df: pd.DataFrame) -> List[Category]:
        """Convert a DataFrame of category rows into Category objects."""
        if df is None or df.empty:
            return []
        categories = []
        for _, row in df.iterrows():
            quartile_str = str(row.get("quartile", "")).strip()
            categories.append(Category(
                id=str(row.get("id", "")).strip(),
                quartiles={quartile_str} if quartile_str else None,
            ))
        return categories

    def _areas_from_df(self, df: pd.DataFrame) -> List[Area]:
        """Convert a DataFrame of area rows into Area objects."""
        if df is None or df.empty:
            return []
        return [Area(id=str(row.get("id", "")).strip()) for _, row in df.iterrows()]

    # -- Public API -----------------------------------------------------------

    def getEntityById(self, identifier: str) -> Optional[IdentifiableEntity]:
        # 1. Search journals first
        jdfs = []
        for h in self.journalQuery:
            try:
                df = h.getById(identifier)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    clean = df.replace("", pd.NA).dropna(how="all")
                    if not clean.empty:
                        jdfs.append(clean)
            except Exception:
                continue

        if jdfs:
            jdf = self._combine_df(jdfs).replace("", pd.NA).dropna(how="all")
            if not jdf.empty:
                exact = jdf.loc[
                    (jdf["id"].astype(str).str.lower() == str(identifier).lower()) |
                    (jdf.get("title", pd.Series(dtype=str)).astype(str).str.lower() == str(identifier).lower())
                ]
                row_df = exact if not exact.empty else jdf
                js = self._journals_from_df(row_df.head(1))
                if js and js[0].hasId():
                    return js[0]

        # 2. Search categories / areas
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
            cdf = self._combine_df(cdfs).replace("", pd.NA).dropna(how="all")
            if not cdf.empty:
                if "quartile" in cdf.columns:
                    cats = self._categories_from_df(cdf.head(1))
                    if cats and cats[0].hasId():
                        return cats[0]
                ars = self._areas_from_df(cdf.head(1))
                if ars and ars[0].hasId():
                    return ars[0]

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
    """
    Extends BasicQueryEngine with cross-store join queries that combine
    journal metadata (Blazegraph) with category/area links (SQLite).
    """

    def _links_df(self) -> pd.DataFrame:
        """Assemble the link table from all registered CategoryQueryHandlers."""
        frames = [h._df_links() for h in self.categoryQuery if isinstance(h, CategoryQueryHandler)]
        return self._combine_df(frames)

    def _journal_df(self) -> pd.DataFrame:
        """Assemble the journal metadata table from all registered JournalQueryHandlers."""
        frames = [h.getAllJournals() for h in self.journalQuery]
        return self._combine_df(frames)

    def _join_on_ids(self, jdf: pd.DataFrame, ldf: pd.DataFrame) -> pd.DataFrame:
        """
        Inner-join journal table (jdf) with link table (ldf) on ISSN.
        Returns an empty DataFrame if either input is empty.
        """
        if jdf.empty or ldf.empty:
            return pd.DataFrame()
        if "id" in jdf.columns and "issn" in ldf.columns:
            return jdf.merge(ldf, left_on="id", right_on="issn", how="inner")
        return pd.DataFrame()

    def getJournalsInCategoriesWithQuartile(
        self, categories: Set[str], quartiles: Set[str]
    ) -> List[Journal]:
        """Return journals that belong to the given categories at the given quartile levels."""
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []

        cat_mask = ldf["category"].notna() if not categories else ldf["category"].isin(categories)
        q_mask = (
            (ldf["quartile"].notna() | ldf["quartile"].isna())
            if not quartiles
            else ldf["quartile"].astype(str).str.upper().isin({q.upper() for q in quartiles})
        )

        lsub   = ldf.loc[cat_mask & q_mask]
        joined = self._join_on_ids(jdf, lsub).drop_duplicates(subset=["id"]).reset_index(drop=True)
        return self._journals_from_df(joined)

    def getJournalsInAreasWithLicense(
        self, areas: Set[str], licenses: Set[str]
    ) -> List[Journal]:
        """Return journals that belong to the given areas and have any of the given licenses."""
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []

        area_mask = ldf["area"].notna() if not areas else ldf["area"].isin(areas)
        lsub      = ldf.loc[area_mask]
        joined    = self._join_on_ids(jdf, lsub)

        if licenses and "license" in joined.columns:
            joined = joined.loc[
                joined["license"].astype(str).str.lower().isin({x.lower() for x in licenses})
            ]

        joined = joined.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return self._journals_from_df(joined)

    def getDiamondJournalsInAreasAndCategoriesWithQuartile(
        self, areas: Set[str], categories: Set[str], quartiles: Set[str]
    ) -> List[Journal]:
        """
        Return Diamond OA journals (APC == False) that satisfy ALL of:
          - belong to at least one of the given areas
          - belong to at least one of the given categories at the given quartile levels
        """
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []

        area_mask = ldf["area"].notna()     if not areas      else ldf["area"].isin(areas)
        cat_mask  = ldf["category"].notna() if not categories else ldf["category"].isin(categories)
        q_mask = (
            (ldf["quartile"].notna() | ldf["quartile"].isna())
            if not quartiles
            else ldf["quartile"].astype(str).str.upper().isin({q.upper() for q in quartiles})
        )

        j_area = self._join_on_ids(jdf, ldf.loc[area_mask])
        j_catq = self._join_on_ids(jdf, ldf.loc[cat_mask & q_mask])

        ids_area = set(j_area["id"].unique()) if "id" in j_area.columns else set()
        ids_catq = set(j_catq["id"].unique()) if "id" in j_catq.columns else set()

        # Keep only journals present in BOTH subsets (area AND category+quartile)
        ok_ids = ids_area & ids_catq
        if not ok_ids:
            return []

        final = jdf.loc[jdf["id"].isin(ok_ids)].copy()

        # Diamond OA: no article processing charges
        if "apc" in final.columns:
            final = final.loc[final["apc"] == False]

        final = final.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return self._journals_from_df(final)

