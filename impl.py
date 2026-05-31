from typing import List, Optional, Set, Dict, Any, Tuple
from collections import OrderedDict
import json
import sqlite3
import os
import re
import pandas as pd
import requests
import traceback

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore, SPARQLStore


# -------------------- In-memory registry --------------------

_REGISTRY: Dict[str, Dict[str, Any]] = {}

def _ensure_registry(key: str) -> Dict[str, Any]:
    if key not in _REGISTRY:
        _REGISTRY[key] = {
            "journals": pd.DataFrame(),    # fallback cache if Blazegraph unreachable
            "categories": pd.DataFrame(),
            "areas": pd.DataFrame(),
            "links": pd.DataFrame(),
        }
    return _REGISTRY[key]


# -------------------- Data models --------------------

class IdentifiableEntity:
    def __init__(self, id, name: str = ""):
        # 原本这里的 id 只能是 str，现在我们允许它接收 list
        self._id = id
        self._name = name.strip() if isinstance(name, str) else ""

    def getId(self) -> str:
        # 兼容旧代码：如果存的是列表，默认返回第一个ID，防止旧代码报错
        if isinstance(self._id, list) and len(self._id) > 0:
            return str(self._id)
        return str(self._id)

    def getIds(self) -> list:
        if isinstance(self._id, list):
            return self._id
        if isinstance(self._id, str):
            # 处理被字符串化的列表格式 "['id1', 'id2']"
            if self._id.startswith('[') and self._id.endswith(']'):
                import ast
                try:
                    return ast.literal_eval(self._id)
                except Exception:
                    pass
            # 处理逗号分隔的格式 "id1, id2"
            if ',' in self._id:
                return [i.strip() for i in self._id.split(',') if i.strip()]
            return [self._id] if self._id else []
        return []

    def hasId(self) -> bool:
        #True if the entity has a non-empty ID.
        return bool(self._id)

    def getName(self) -> str:
        #Return the entity’s human-readable name.
        return self._name

    def hasName(self) -> bool:
        #True if the entity has a non-empty name.
        return bool(self._name)


class Area(IdentifiableEntity):
    """Represents a SCImago Area (e.g., 'Engineering')."""
    def __init__(self, id: str = "", name: str = "", description: str = ""):
        super().__init__(id, name)
        self._description = description

    def getIds(self) -> list:
        # Returns a list containing this area's ID
        return [self._id] if self._id else []


class Category(IdentifiableEntity):
    """ Represents a SCImago Category (e.g., 'Artificial Intelligence').
    Each Category may have one or more Quartiles (Q1–Q4). """
    
    def __init__(self, id: str = "", name: str = "", quartiles: Optional[Set[str]] = None):
        super().__init__(id, name)
        self._quartiles: Set[str] = set()
        if quartiles:
            for q in quartiles:
                self.addQuartile(q)

    def addQuartile(self, quartile: Optional[str]) -> None:
        #Add a Quartile ranking (Q1–Q4).
        if quartile and quartile.strip():
            self._quartiles.add(quartile.strip().upper())

    def getQuartiles(self) -> List[str]:
        #Return all Quartiles for this Category.
        return sorted(self._quartiles)

    def getQuartile(self) -> Optional[str]:
        """Return the first (primary) quartile for this Category, or None if unranked.
        Required by the UML diagram (getQuartile(): string or None)."""
        qs = sorted(self._quartiles)
        return qs[0] if qs else None

    def hasQuartile(self, q: Optional[str] = None) -> bool:
        #True if Category is ranked in the given Quartile or has any Quartile.
        if q:
            return q.strip().upper() in self._quartiles
        return len(self._quartiles) > 0

    def getIds(self) -> list:
        # Returns a list containing this category's ID
        return [self._id] if self._id else []


class Journal(IdentifiableEntity):
    """ Represents a DOAJ Journal (schema.org:Periodical). 
    Journals can be associated with multiple Categories and Areas. """
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
        #Link a Category to this Journal.
        if category and category.getId() not in self._categories:
            self._categories[category.getId()] = category

    def getCategories(self) -> List[Category]:
        #Return all Categories linked to this Journal.
        return list(self._categories.values())

    def hasCategories(self) -> bool:
        #True if the Journal has at least one linked Category.
        return len(self._categories) > 0

    def addArea(self, area: Area) -> None:
        #Link an Area to this Journal.
        if area and area.getId() not in self._areas:
            self._areas[area.getId()] = area

    def getAreas(self) -> List[Area]:
        #Return all Areas linked to this Journal.
        return list(self._areas.values())

    def hasAreas(self) -> bool:
        #True if the Journal has at least one linked Area.
        return len(self._areas) > 0

    def getIds(self) -> list:
        # The id field may be a comma-separated list of ISSNs (e.g. "1234-5678, 8765-4321")
        # stored that way during upload. Split and return all of them.
        if isinstance(self._id, list):
            return self._id
        if isinstance(self._id, str) and self._id:
            parts = [p.strip() for p in self._id.split(",") if p.strip()]
            return parts if parts else []
        return []


# -------------------- Handler base classes --------------------

class Handler:
    #the parent of all handler types
    def __init__(self):
        self.dbPathOrUrl: str = ""

    def getDbPathOrUrl(self) -> str:
        return self.dbPathOrUrl

    def setDbPathOrUrl(self, val: str) -> bool: 
        self.dbPathOrUrl = val
        _ensure_registry(val) # make sure registry exists
        return True

class UploadHandler(Handler):
    #abstract subclass for data ingestion
    def pushDataToDb(self, file_path: str) -> bool: # must be overridden by specific uploaders
        raise NotImplementedError() # if someone forgets to override it

class QueryHandler(Handler):
    #abstract subclass for data retrieval
    def getById(self, id: str) -> pd.DataFrame:
        raise NotImplementedError


# -------------------- Graph / Blazegraph helpers --------------------

# Shortcut for schema.org URL prefix.
# Lets us write SCHEMA.Periodical instead of https://schema.org/Periodical
SCHEMA = Namespace("https://schema.org/")


def _bool_from_str(v: Any) -> Optional[bool]:
    """Convert a string value to a Python boolean.
    'yes'/'true'/'1' -> True, 'no'/'false'/'0' -> False, anything else -> None.
    Needed because CSV files store true/false as plain text, not as real booleans."""
    if isinstance(v, bool):       # already a boolean, return it directly
        return v
    if isinstance(v, str):
        w = v.strip().lower()     # remove spaces and lowercase for safe comparison
        if w in {"true", "yes", "y", "1"}:
            return True
        if w in {"false", "no", "n", "0"}:
            return False
    return None                   # unrecognised value, return None (unknown)


def _build_journal_uri(issn: str) -> URIRef:
    """Build a unique URI for a journal using its ISSN.
    Example: ISSN 1234-5678 becomes http://example.org/periodical/1234-5678
    Every resource in RDF needs a unique web address (URI) to identify it."""
    return URIRef(f"http://example.org/periodical/{issn}")


class _BlazegraphClient:
    """Private helper that handles all communication with the Blazegraph server.
    The underscore in the name means it is only used inside this file."""

    def __init__(self, endpoint: str):
        # Save the Blazegraph SPARQL endpoint URL, e.g.:
        # http://127.0.0.1:9999/blazegraph/sparql
        self.endpoint = endpoint

    def upload_graph(self, g: Graph) -> bool:
        """Upload all RDF triples from a local Graph to Blazegraph.
        Serialises the graph to N-Triples format and sends it via HTTP POST.
        Returns True if upload succeeded, False if something went wrong."""
        try:
            # Serialise the entire graph to N-Triples text format.
            # N-Triples is a simple line-by-line RDF format that Blazegraph accepts.
            # encode('utf-8') converts the text to bytes for the HTTP request.
            data = g.serialize(format="nt").encode("utf-8")

            # Send the N-Triples data to Blazegraph via HTTP POST.
            # Content-Type tells Blazegraph the format we are sending.
            response = requests.post(
                self.endpoint,
                data=data,
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )

            # HTTP 200 or 204 both mean success
            if response.status_code in (200, 204):
                print(f"[OK] Successfully uploaded {len(g)} triples to Blazegraph")
                return True
            else:
                print(f"[ERROR] Blazegraph returned status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to upload graph: {e}")
            traceback.print_exc()  # print full error details to the terminal
            return False

    def select(self, query: str) -> List[Dict[str, Any]]:
        """Run a SPARQL SELECT query against Blazegraph.
        Returns the results as a list of dictionaries, one dict per result row."""
        try:
            store = SPARQLStore(self.endpoint)  # read-only connection to Blazegraph
            g = Graph(store=store)              # graph object linked to Blazegraph
            rows = []
            for row in g.query(query):          # send query and loop over result rows
                binding = {}
                for var, val in row.asdict().items():           # convert row to a dict
                    binding[var] = str(val) if val is not None else None  # values as strings
                rows.append(binding)
            return rows
        except Exception as e:
            print(f"[ERROR] SPARQL query failed: {e}")
            return []   # return empty list so the program does not crash


class JournalUploadHandler(UploadHandler):
    """Reads the DOAJ CSV file, converts every journal into RDF triples,
    uploads them to Blazegraph, and saves a local cache for offline use."""

    def pushDataToDb(self, file_path: str) -> bool:
        """Main upload method, called by the test as u.pushDataToDb('data/doaj.csv').
        Returns True on success, False if the file is missing or an error occurred."""
        reg = _ensure_registry(self.dbPathOrUrl)  # prepare the local cache slot
        try:
            path = file_path
            # If the file is not found at the given path, try prepending './'
            if not os.path.isfile(path) and os.path.isfile(os.path.join(".", path)):
                path = os.path.join(".", path)

            # If the file still does not exist, log a warning and stop
            if not os.path.isfile(path):
                print(f"[WARNING] Journal file not found: {path}.")
                reg["journals"] = pd.DataFrame(
                    columns=["id", "title", "publisher", "license", "apc", "doaj_seal", "languages"]
                )
                return False

            # Read the entire CSV file into a DataFrame called df_raw.
            # dtype=str forces every column to be text (prevents ISSN becoming an integer).
            # keep_default_na=False makes empty cells stay as "" instead of NaN.
            df_raw = pd.read_csv(path, dtype=str, keep_default_na=False)

            # Build a lowercase->original column name dictionary so we can
            # find columns regardless of how they are capitalised in the CSV
            cols_lower = {c.lower(): c for c in df_raw.columns}

            def pick(*keys):
                """Return the first CSV column whose lowercased name contains
                any of the given keywords. Returns None if nothing matches."""
                for k in keys:
                    for low, orig in cols_lower.items():
                        if k in low:
                            return orig
                return None

            # Map each logical field to the actual column name in this CSV.
            # If a column is not found, the variable is None and the field is skipped.
            # Collect BOTH print ISSN and e-ISSN separately so we can store all IDs.
            col_issn_print = pick("pissn", "print issn", "journal issn")
            col_issn_e     = pick("eissn", "online issn", "electronic issn")
            # Fallback: if neither specific column found, try generic "issn"
            col_issn_generic = pick("issn", "journal id", "identifier") if not col_issn_print and not col_issn_e else None
            col_title     = pick("title")
            col_publisher = pick("publisher")
            col_license   = pick("license")
            col_apc       = pick("apc", "article processing charge", "processing charges")
            col_seal      = pick("doaj seal", "seal")
            col_lang      = pick("language")

            g = Graph()               # create an empty RDF graph in memory (not in Blazegraph yet)
            g.bind("schema", SCHEMA)  # register the schema: prefix for readable serialisation

            fallback_rows = []        # list that will become the local cache DataFrame

            # Loop over every row in the CSV
            for _, row in df_raw.iterrows():
                # Collect all available ISSNs for this journal (print + electronic)
                issn_parts = []
                if col_issn_print:
                    v = str(row[col_issn_print]).strip()
                    if v and v.lower() not in ("", "nan", "none"):
                        issn_parts.append(v)
                if col_issn_e:
                    v = str(row[col_issn_e]).strip()
                    if v and v.lower() not in ("", "nan", "none"):
                        issn_parts.append(v)
                if col_issn_generic and not issn_parts:
                    v = str(row[col_issn_generic]).strip()
                    if v and v.lower() not in ("", "nan", "none"):
                        issn_parts.append(v)
                # Use comma-separated string of all ISSNs as the id
                # (first ISSN is the primary one used for the RDF URI)
                issn = issn_parts[0] if issn_parts else ""
                all_issns = ", ".join(issn_parts)  # e.g. "1234-5678, 8765-4321"
                title     = str(row[col_title]).strip()     if col_title     else ""
                publisher = str(row[col_publisher]).strip() if col_publisher else ""
                license_  = str(row[col_license]).strip()   if col_license   else ""
                apc       = _bool_from_str(row[col_apc])    if col_apc       else None
                seal      = _bool_from_str(row[col_seal])   if col_seal      else None
                langs_raw = str(row[col_lang]).strip()      if col_lang      else ""
                # DOAJ separates multiple languages with ", " (comma + space) per spec
                languages = [l.strip() for l in langs_raw.split(", ")] if langs_raw else []

                # Skip rows with no ISSN and no title - they are useless records
                if not issn and not title:
                    continue

                # Add this journal to the local cache list as a plain dictionary
                fallback_rows.append({
                    "id": all_issns or title,  # store all ISSNs; fall back to title if none
                    "title": title,
                    "publisher": publisher,
                    "license": license_,
                    "apc": apc,
                    "doaj_seal": seal,
                    "languages": languages,
                })

                # Only build RDF triples when at least one ISSN exists (required for a unique URI)
                if issn:
                    s = _build_journal_uri(issn)                      # unique URI for this journal
                    g.add((s, RDF.type, SCHEMA.Periodical))           # declare it is a journal
                    # Store ALL ISSNs so getIds() can return the full list
                    for issn_val in issn_parts:
                        g.add((s, SCHEMA.issn, Literal(issn_val)))
                    if title:
                        g.add((s, SCHEMA.name, Literal(title)))       # store the title
                    if publisher:
                        g.add((s, SCHEMA.publisher, Literal(publisher)))  # store the publisher
                    if license_:
                        g.add((s, SCHEMA.license, Literal(license_)))     # store the license
                    for lang in languages:
                        # Each language gets its own triple because RDF has no list type.
                        # Multiple values = multiple triples with the same subject.
                        g.add((s, SCHEMA.inLanguage, Literal(lang)))

                    # APC uses the schema.org PropertyValue pattern because
                    # schema.org:Periodical has no direct APC property.
                    # Pattern: journal -> additionalProperty -> mini-node -> value
                    if apc is not None:
                        pv = URIRef(str(s) + "#pv-apc")               # URI for the APC mini-node
                        g.add((s, SCHEMA.additionalProperty, pv))     # link journal to mini-node
                        g.add((pv, RDF.type, SCHEMA.PropertyValue))   # declare the node type
                        g.add((pv, SCHEMA.name, Literal("APC")))      # name the property
                        g.add((pv, SCHEMA.value, Literal(bool(apc), datatype=XSD.boolean)))  # store boolean value

                    # Same PropertyValue pattern for DOAJ Seal
                    if seal is not None:
                        pv2 = URIRef(str(s) + "#pv-doaj-seal")
                        g.add((s, SCHEMA.additionalProperty, pv2))
                        g.add((pv2, RDF.type, SCHEMA.PropertyValue))
                        g.add((pv2, SCHEMA.name, Literal("DOAJSeal")))
                        g.add((pv2, SCHEMA.value, Literal(bool(seal), datatype=XSD.boolean)))

            # Upload the RDF graph to Blazegraph.
            # Returns True if Blazegraph accepted the data, False on failure.
            ok = _BlazegraphClient(self.dbPathOrUrl).upload_graph(g)

            # Always save to local cache, even if Blazegraph upload failed.
            # from_records() converts the list of dicts to a DataFrame.
            # reset_index(drop=True) resets row numbers to 0, 1, 2, 3...
            reg["journals"] = (
                pd.DataFrame.from_records(fallback_rows).reset_index(drop=True)
            )
            return ok

        except Exception as e:
            # Unexpected error: print details and return failure
            print(f"[ERROR] Exception in JournalUploadHandler.pushDataToDb: {e}")
            traceback.print_exc()
            reg["journals"] = pd.DataFrame(
                columns=["id", "title", "publisher", "license", "apc", "doaj_seal", "languages"]
            )
            return False

class CategoryUploadHandler(UploadHandler):
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

                for cat in categories:
                    # Clean the unique identifier (ID) and attribute (Quartile)
                    cid = str(cat.get("id", "")).strip()
                    quart = str(cat.get("quartile", "")).strip()
                    if not cid:
                        continue
                    cat_rows.append({"id": cid, "quartile": quart})
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
            # Use drop_duplicates to ensure entity uniqueness
            pd.DataFrame(cat_rows).drop_duplicates().to_sql("categories", conn, if_exists="replace", index=False)
            pd.DataFrame(area_rows).drop_duplicates().to_sql("areas", conn, if_exists="replace", index=False)
            pd.DataFrame(link_rows).drop_duplicates().to_sql("links", conn, if_exists="replace", index=False)
            conn.close()
            return True
        except Exception as e:
            print(f"Error during upload: {e}")


# -------------------- Query handlers --------------------

class JournalQueryHandler(QueryHandler):
    """Queries journal data from the local cache or Blazegraph.
    Strategy: check the local cache first (fast, no network needed).
    If the cache is empty, fall back to a SPARQL query against Blazegraph."""

    def _client(self) -> _BlazegraphClient:
        """Create a Blazegraph client using the stored endpoint URL."""
        return _BlazegraphClient(self.dbPathOrUrl)

    def _fallback_df(self) -> pd.DataFrame:
        """Return the local journals cache DataFrame.
        Returns an empty DataFrame if no upload has happened yet."""
        return _ensure_registry(self.dbPathOrUrl).get("journals", pd.DataFrame())

    @staticmethod
    def _aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Merge multiple SPARQL result rows into one row per journal.
        Problem: SPARQL returns one row per language, so a journal with
        3 languages comes back as 3 separate rows.
        This method collapses them into one row with a languages list."""
        by_id: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            issn = r.get("issn") or r.get("id") or ""
            if not issn:
                continue
            # setdefault: create a new entry the first time we see this ISSN,
            # or return the existing entry if we have already seen it
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
                # Convert SPARQL string result "true"/"false" to Python boolean
                entry["apc"] = True if val in ("true", "1") else (False if val in ("false", "0") else None)
            if r.get("seal"):
                val = r["seal"].lower()
                entry["doaj_seal"] = True if val in ("true", "1") else (False if val in ("false", "0") else None)
            # Add the language only if it is not already in the list (avoid duplicates)
            if r.get("lang") and r["lang"] not in entry["languages"]:
                entry["languages"].append(r["lang"])

        df = pd.DataFrame.from_records(list(by_id.values()))  # convert merged dict to DataFrame
        return df.reset_index(drop=True)

    def _select_df(self, where_filter: str = "", limit: Optional[int] = None) -> pd.DataFrame:
        """Central method used by all public query methods.
        Tier 1 (fast): filter the local cache — no Blazegraph connection needed.
        Tier 2 (fallback): run a SPARQL query if the cache is empty."""

        fb_df = self._fallback_df().copy()  # .copy() ensures we never modify the original cache

        # TIER 1: use local cache if it has data
        if not fb_df.empty:
            filtered_df = fb_df.copy()

            # Detect which type of filter was requested by inspecting the SPARQL filter string,
            # then apply the equivalent filter directly on the cache DataFrame
            if not where_filter:
                pass  # no filter: return everything (used by getAllJournals)

            elif "issn" in where_filter.lower() and "CONTAINS" not in where_filter:
                # ISSN exact-match filter built by getById:
                # FILTER (LCASE(STR(?issn)) = LCASE("..."))
                match = re.search(r'LCASE\(STR\(\?issn\)\)\s*=\s*LCASE\("(.+?)"\)', where_filter)
                if match:
                    term = match.group(1).strip().lower()
                    # id column may be comma-separated ISSNs; split and compare each part
                    filtered_df = filtered_df[
                        filtered_df["id"].astype(str).apply(
                            lambda cell: any(
                                part.strip().lower() == term
                                for part in cell.split(",")
                            )
                        )
                    ]
                else:
                    filtered_df = pd.DataFrame()  # unrecognised ISSN filter → no results

            elif "CONTAINS" in where_filter and "title" in where_filter:
                # Extract the search term from inside the SPARQL FILTER string using regex
                match = re.search(r'CONTAINS\(.*?LCASE\("(.+?)"\)', where_filter)
                if match:
                    term = match.group(1).lower()
                    # Case-insensitive substring search on the title column
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
                # Keep only rows where the apc column equals True
                filtered_df = filtered_df[filtered_df.get("apc") == True]

            elif "seal" in where_filter and "true" in where_filter:
                # Keep only rows where the doaj_seal column equals True
                filtered_df = filtered_df[filtered_df.get("doaj_seal") == True]

            elif "NOT EXISTS" in where_filter and "publisher" in where_filter:
                # getJournalsWithNoPublisher: keep rows with empty/missing publisher
                filtered_df = filtered_df[
                    filtered_df["publisher"].isna() |
                    filtered_df["publisher"].astype(str).str.strip().str.lower().isin(["", "none", "nan"])
                ]

            elif "license" in where_filter:
                # Extract all license strings from the SPARQL filter and match against cache
                licenses = re.findall(r'LCASE\("(.+?)"\)', where_filter)
                if licenses:
                    lc_set = {l.lower() for l in licenses}
                    filtered_df = filtered_df[
                        filtered_df.get("license", pd.Series(dtype=str))
                        .astype(str).str.lower().isin(lc_set)
                    ]

            elif where_filter:
                # Unrecognised filter pattern — don't return the full dataset as a false positive
                filtered_df = pd.DataFrame()

            if limit:
                filtered_df = filtered_df.head(limit)  # cap the number of results if requested

            return filtered_df.reset_index(drop=True) if not filtered_df.empty else pd.DataFrame()

        # TIER 2: SPARQL fallback (only reached when local cache is empty)
        lim = f"LIMIT {limit}" if limit else ""
        query = f"""
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
            rows = self._client().select(query)   # send SPARQL query to Blazegraph
            return self._aggregate_rows(rows)      # collapse multi-language rows into one per journal
        except Exception:
            traceback.print_exc()
            return self._fallback_df().copy()      # last resort: return the raw cache

    # ---- Public query methods ----

    def getById(self, id_value: str) -> pd.DataFrame:
        """Find a journal by ISSN (exact match on any stored ISSN).
        Per spec, getById matches identifiers only — not titles."""
        # Use exact ISSN equality in SPARQL
        where = f'FILTER (LCASE(STR(?issn)) = LCASE("{id_value}"))'
        df = self._select_df(where_filter=where)
        if not df.empty:
            return df.reset_index(drop=True)

        # Fallback: search the local cache for any row whose id field contains this ISSN
        fb = self._fallback_df()
        if fb.empty:
            return pd.DataFrame()
        # id may be a comma-separated list of ISSNs (e.g. "1234-5678, 8765-4321")
        mask = fb["id"].astype(str).apply(
            lambda cell: any(
                part.strip().lower() == id_value.strip().lower()
                for part in cell.split(",")
            )
        )
        matched = fb.loc[mask].reset_index(drop=True)
        return matched if not matched.empty else pd.DataFrame()

    def getAllJournals(self) -> pd.DataFrame:
        """Return all journals with no filter applied."""
        return self._select_df()

    def getJournalsWithTitle(self, text: str) -> pd.DataFrame:
        """Return journals whose title contains the search text (case-insensitive).
        BOUND(?title) skips journals with no title.
        CONTAINS(LCASE(A), LCASE(B)) checks if B is a substring of A."""
        where = f'FILTER (BOUND(?title) && CONTAINS(LCASE(STR(?title)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsPublishedBy(self, text: str) -> pd.DataFrame:
        """Return journals whose publisher name contains the search text."""
        where = f'FILTER (BOUND(?publisher) && CONTAINS(LCASE(STR(?publisher)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsWithLicense(self, licenses: Set[str]) -> pd.DataFrame:
        """Return journals matching any of the given license types.
        Empty set means return all journals (per project specification)."""
        if not licenses:
            return self.getAllJournals()  # empty input = no filter = return everything
        # Build one condition per license joined with OR: LCASE(?license) = 'cc by' || ...
        filters = " || ".join(
            [f'LCASE(STR(?license)) = LCASE("{lic}")' for lic in licenses]
        )
        where = f"FILTER (BOUND(?license) && ({filters}))"
        return self._select_df(where_filter=where)

    def getJournalsWithAPC(self) -> pd.DataFrame:
        """Return journals that charge Article Processing Charges (APC = True).
        Uses STR(?apc) = "true" for maximum Blazegraph compatibility — typed boolean
        literals can behave inconsistently across Blazegraph versions."""
        where = 'FILTER (BOUND(?apc) && (STR(?apc) = "true"))'
        return self._select_df(where_filter=where)

    def getJournalsWithDOAJSeal(self) -> pd.DataFrame:
        """Return journals that have the DOAJ Seal quality certification.
        Same STR-based boolean pattern as getJournalsWithAPC."""
        where = 'FILTER (BOUND(?seal) && (STR(?seal) = "true"))'
        return self._select_df(where_filter=where)

    def getJournalsWithNoPublisher(self) -> pd.DataFrame:
        # Method with no parameters, return  a table of j with no publ
    
        where = 'FILTER NOT EXISTS { ?s schema:publisher ?publisher }' 
        # SPARQL FILTER, keeps only j where publ triple NOT EXIST at all
        df = self._select_df(where_filter=where)
        # send the FILTER to Bl and get result
        if not df.empty:
            return df

        # Cache fallback, copy of data
        fb = self._fallback_df()
        if fb.empty:
            return pd.DataFrame()
        no_pub_mask = (
            fb["publisher"].isna() |
            fb["publisher"].astype(str).str.strip().str.lower().isin(["", "none", "nan"])
        )
        # Boolean mask TRUE if publ is empty, None, NaN no publisher
        return fb.loc[no_pub_mask].reset_index(drop=True)
        # Apply the mask, keep only rows WHERE TRUE, return result

class CategoryQueryHandler(QueryHandler):
    def getById(self, id_value: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            query = "SELECT * FROM categories WHERE id = ?"
            df = pd.read_sql_query(query, conn, params=(id_value,))
            conn.close()
            return df
        except:
            return pd.DataFrame()

    def getAllCategories(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            # Use DISTINCT on id to avoid returning the same category multiple times.
            # The categories table stores one row per (id, quartile) pair, so without
            # deduplication the same category name appears once per quartile it has.
            df = pd.read_sql_query(
                "SELECT DISTINCT id, quartile FROM categories", conn
            )
            conn.close()
            return df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        except:
            return pd.DataFrame()

    def getAllAreas(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            df = pd.read_sql_query("SELECT * FROM areas", conn)
            conn.close()
            return df
        except:
            return pd.DataFrame()

    def getCategoriesWithQuartile(self, quartiles: Set[str]) -> pd.DataFrame:
        # If no quartiles given, return all categories (spec: empty = all)
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            df = pd.read_sql_query("SELECT * FROM categories", conn)
            conn.close()
            if df.empty:
                return pd.DataFrame()
            if not quartiles:
                return df.drop_duplicates(subset=["id"]).reset_index(drop=True)
            wanted = {q.strip().upper() for q in quartiles}
            mask = df["quartile"].astype(str).str.strip().str.upper().isin(wanted)
            return df.loc[mask].drop_duplicates(subset=["id"]).reset_index(drop=True)
        except:
            return pd.DataFrame()

    def getCategoriesAssignedToAreas(self, areas: Set[str]) -> pd.DataFrame:
        # Find all categories that are linked to the given areas via the links table.
        # If areas is empty, return categories for all areas (spec: empty = all).
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            df_links = pd.read_sql_query("SELECT * FROM links", conn)
            conn.close()
            if df_links.empty:
                return pd.DataFrame(columns=["id", "quartile"])
            if areas:
                norm = {a.strip().lower() for a in areas}
                df_links = df_links[df_links["area"].astype(str).str.strip().str.lower().isin(norm)]
            cats = df_links[df_links["category"].notna()][["category", "quartile"]].drop_duplicates()
            cats = cats.rename(columns={"category": "id"})
            return cats.drop_duplicates(subset=["id"]).reset_index(drop=True)
        except:
            return pd.DataFrame(columns=["id", "quartile"])

    def getCategoriesByName(self, cat_partial_name: str) -> pd.DataFrame:
        """Searches category by partial name match
        takes search text as input
        """
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            # using stored path
            # get unique rows from categories where the name matches
            query = """
                SELECT DISTINCT id, quartile
                FROM categories
                WHERE LOWER(id) LIKE LOWER(?)
            """
            # any characters can appear, if the input intel
            # enables partial matching
            search_term = f"%{cat_partial_name}%"
            df = pd.read_sql_query(query, conn, params=(search_term,))
            # execute sql, pass 3 things
            conn.close()
            return df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def getAreasAssignedToCategories(self, categories: Set[str]) -> pd.DataFrame:
        # Find all areas that are linked to the given categories via the links table.
        # If categories is empty, return areas for all categories (spec: empty = all).
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            df_links = pd.read_sql_query("SELECT * FROM links", conn)
            conn.close()
            if df_links.empty:
                return pd.DataFrame(columns=["id"])
            if categories:
                norm = {c.strip().lower() for c in categories}
                df_links = df_links[df_links["category"].astype(str).str.strip().str.lower().isin(norm)]
            areas = df_links[df_links["area"].notna()][["area"]].drop_duplicates()
            areas = areas.rename(columns={"area": "id"})
            return areas.drop_duplicates(subset=["id"]).reset_index(drop=True)
        except:
            return pd.DataFrame(columns=["id"])

    def _df_links(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='links'")
            if cursor.fetchone():
                df = pd.read_sql_query("SELECT * FROM links", conn)
            else:
                categories_df = pd.read_sql_query("SELECT id as category FROM categories", conn)
                areas_df = pd.read_sql_query("SELECT id as area FROM areas", conn)
                links_data = []
                for _, cat_row in categories_df.iterrows():
                    for _, area_row in areas_df.iterrows():
                        links_data.append({
                            'issn': None,
                            'category': cat_row['category'],
                            'area': area_row['area'],
                            'quartile': 'Q1'
                        })
                df = pd.DataFrame(links_data) if links_data else pd.DataFrame(columns=['issn', 'category', 'area', 'quartile'])
            conn.close()
            return df
        except:
            return pd.DataFrame(columns=['issn', 'category', 'area', 'quartile'])


# -------------------- Query Engines --------------------

class BasicQueryEngine:
    def __init__(self):
        self.journalQuery: List[JournalQueryHandler] = []
        self.categoryQuery: List[CategoryQueryHandler] = []

    # -- handler management --------------------------------------------------

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

    # -- Helper Functions for Query Engine -----------------------------------

    def _combine_df(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
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
        if isinstance(val, list):
            # Already a list (e.g., loaded from JSON)
            return [v.strip() for v in val if isinstance(v, str) and v.strip()]
        if isinstance(val, str):
            # DOAJ uses ", " as separator (comma + space)
            return [v.strip() for v in val.split(", ") if v.strip()]
        return []

    def _journals_from_df(self, df: pd.DataFrame) -> List[Journal]:
        if df is None or df.empty:
            return []
        # 避免在组合 DataFrame 时生成重复的 Journal 实例
        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"])
        journals = []
        for _, row in df.iterrows():
            languages_val = self._parse_languages(row.get("languages",))
            # 宽容的布尔值解析：防止 CSV/GraphDB 传来的 "yes"/"no"/"false" 导致判断失败
            apc_val = row.get("apc")
            if pd.isna(apc_val) or str(apc_val).strip() == "":
                apc_bool = None
            else:
                apc_bool = str(apc_val).strip().lower() in ['true', 'yes', '1', 't']
            seal_val = row.get("doaj_seal")
            if pd.isna(seal_val) or str(seal_val).strip() == "":
                seal_bool = None
            else:
                seal_bool = str(seal_val).strip().lower() in ['true', 'yes', '1', 't']
            journals.append(
                Journal(
                    id=str(row.get("id", "")).strip(),
                    title=str(row.get("title", "")).strip(),
                    publisher=str(row.get("publisher", "")).strip(),
                    license=str(row.get("license", "")).strip(),
                    apc=apc_bool,
                    doaj_seal=seal_bool,
                    languages=languages_val,
                )
            )
        return journals

    def _categories_from_df(self, df: pd.DataFrame) -> List[Category]:
        if df is None or df.empty:
            return []
        categories = []
        for _, row in df.iterrows():
            quartile_str = str(row.get("quartile", "")).strip()
            quartiles = {quartile_str} if quartile_str else None
            categories.append(Category(id=str(row.get("id", "")).strip(), quartiles=quartiles))
        return categories

    def _areas_from_df(self, df: pd.DataFrame) -> List[Area]:
        if df is None or df.empty:
            return []
        return [Area(id=str(row.get("id", "")).strip()) for _, row in df.iterrows()]

    # -- public API -----------------------------------------------------------

    def getEntityById(self, identifier: str) -> Optional[IdentifiableEntity]:
        if not identifier:
            return None
        target_id = str(identifier).strip().lower()

        # --- JOURNAL LOOKUP: use getById (exact ISSN match) ---
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

        if not jdfs:
            # Fallback: scan getAllJournals and match any ISSN in the comma-separated id
            for h in self.journalQuery:
                try:
                    df = h.getAllJournals()
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        mask = df["id"].astype(str).apply(
                            lambda cell: any(
                                part.strip().lower() == target_id
                                for part in cell.split(",")
                            )
                        )
                        matched = df.loc[mask]
                        if not matched.empty:
                            jdfs.append(matched)
                except Exception:
                    continue

        if jdfs:
            jdf = self._combine_df(jdfs)
            js = self._journals_from_df(jdf.head(1))
            if js:
                return js[0]

        # --- CATEGORY / AREA LOOKUP ---
        cdfs = []
        for h in self.categoryQuery:
            try:
                df_cat = h.getAllCategories()
                if isinstance(df_cat, pd.DataFrame) and not df_cat.empty:
                    cdfs.append(df_cat.replace("", pd.NA).dropna(how="all"))
                df_area = h.getAllAreas()
                if isinstance(df_area, pd.DataFrame) and not df_area.empty:
                    cdfs.append(df_area.replace("", pd.NA).dropna(how="all"))
            except Exception:
                continue
        if cdfs:
            cdf = self._combine_df(cdfs)
            cdf = cdf.replace("", pd.NA).dropna(how="all")
            if not cdf.empty and "id" in cdf.columns:
                # Exact match only — partial matching would return false positives
                exact_cat = cdf[cdf["id"].astype(str).str.lower() == target_id]
                if not exact_cat.empty:
                    if "quartile" in exact_cat.columns and not exact_cat["quartile"].dropna().empty:
                        cats = self._categories_from_df(exact_cat.head(1))
                        if cats:
                            return cats[0]
                    else:
                        ars = self._areas_from_df(exact_cat.head(1))
                        if ars:
                            return ars[0]
        return None

    def getAllJournals(self) -> List[Journal]:
        df = self._combine_df([h.getAllJournals() for h in self.journalQuery])
        return self._journals_from_df(df)

    def getJournalsWithTitle(self, text: str) -> List[Journal]:
        # Use each handler's own getJournalsWithTitle for correct filtering
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
    def _links_df(self) -> pd.DataFrame:
        """Assemble and combine all link tables from category query handlers.
        Each link table connects journals with categories, quartiles, or areas."""
        frames = []
        for h in self.categoryQuery:
            if isinstance(h, CategoryQueryHandler):
                frames.append(h._df_links())
        return self._combine_df(frames)

    def _journal_df(self) -> pd.DataFrame:
        """Assemble and combine all journal tables from journal query handlers.
        Each handler contributes its own journal metadata table."""
        frames = [h.getAllJournals() for h in self.journalQuery]
        return self._combine_df(frames)

    def _join_on_ids(self, jdf: pd.DataFrame, ldf: pd.DataFrame) -> pd.DataFrame:
        """Safely join the journal table and link table by matching on ISSN identifiers."""
        if jdf.empty or ldf.empty:
            return pd.DataFrame()
        if "id" in jdf.columns and "issn" in ldf.columns:
            j_temp = jdf.copy()
            l_temp = ldf.copy()
            # Split comma-separated ISSN strings into lists
            j_temp['join_id'] = j_temp['id'].astype(str).str.split(',')
            l_temp['join_issn'] = l_temp['issn'].astype(str).str.split(',')
            # Explode lists so each ISSN becomes its own row for precise merging
            j_temp = j_temp.explode('join_id')
            l_temp = l_temp.explode('join_issn')
            # Remove whitespace and case differences for reliable matching
            j_temp['join_id'] = j_temp['join_id'].str.strip().str.lower()
            l_temp['join_issn'] = l_temp['join_issn'].str.strip().str.lower()
            # Filter out empty strings and merge
            j_temp = j_temp[j_temp['join_id'] != '']
            # Rename 'id' in ldf to avoid id_x / id_y collision after merge
            if "id" in l_temp.columns:
                l_temp = l_temp.rename(columns={"id": "link_id"})
            merged = j_temp.merge(l_temp, left_on="join_id", right_on="join_issn", how="inner")
            # Drop temporary helper columns before returning
            drop_cols = [c for c in ['join_id', 'join_issn', 'link_id'] if c in merged.columns]
            merged = merged.drop(columns=drop_cols)
            # Deduplicate on journal id so each journal appears only once
            if "id" in merged.columns:
                merged = merged.drop_duplicates(subset=["id"])
            return merged
        return pd.DataFrame()

    def getJournalsInCategoriesWithQuartile(self, categories: Set[str], quartiles: Set[str]) -> List[Journal]:
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []
        norm_cats = {c.strip().lower() for c in categories} if categories else set()
        norm_qs = {q.strip().upper() for q in quartiles} if quartiles else set()
        cat_mask = ldf["category"].notna() if not categories else ldf["category"].astype(str).str.strip().str.lower().isin(norm_cats)
        q_mask = (ldf["quartile"].notna() | ldf["quartile"].isna()) if not quartiles else ldf["quartile"].astype(str).str.strip().str.upper().isin(norm_qs)
        lsub = ldf.loc[cat_mask & q_mask]
        joined = self._join_on_ids(jdf, lsub)
        return self._journals_from_df(joined)

    def getJournalsInAreasWithLicense(self, areas: Set[str], licenses: Set[str]) -> List[Journal]:
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []
        norm_areas = {a.strip().lower() for a in areas} if areas else set()
        area_mask = ldf["area"].notna() if not areas else ldf["area"].astype(str).str.strip().str.lower().isin(norm_areas)
        lsub = ldf.loc[area_mask]
        joined = self._join_on_ids(jdf, lsub)
        # 修复严格匹配带来的丢失
        # Apply license filter after the join to avoid losing journals with missing license data
        if licenses and "license" in joined.columns:
            norm_lics = {x.strip().lower() for x in licenses}
            joined = joined.loc[joined["license"].astype(str).str.strip().str.lower().isin(norm_lics)]
        return self._journals_from_df(joined)

    def getJournalsWithNoPublisherInCategories(self, cat_partial_name: str) -> List[Journal]:
        """Return all journals that have no publisher AND belong to at least one
        category whose name matches (even partially) the given string.
        """

        # Step 1: collect all no-publisher journals from every JournalQueryHandler 
        no_pub_frames = []
        for h in self.journalQuery: # loop through all journal
            try:
                df = h.getJournalsWithNoPublisher() # call on each one
                if isinstance(df, pd.DataFrame) and not df.empty:
                    no_pub_frames.append(df) # collect all the result
            except Exception:
                continue
        if not no_pub_frames:
            return []  # no journals without publisher found at all
        no_pub_df = self._combine_df(no_pub_frames)  # combine them in one table

        # Step 2: collect matching categories from every CategoryQueryHandler 
        cat_frames = []
        for h in self.categoryQuery:
            try:
                df = h.getCategoriesByName(cat_partial_name)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    cat_frames.append(df)
            except Exception:
                continue
        if not cat_frames:
            return []  # no categories match the given name
        cat_df = self._combine_df(cat_frames)  # merge results from all handlers

        matching_cat_names = set( # Take all category names from the result and put in a set, no dublicates
            cat_df["id"].astype(str).str.strip().str.lower().unique() # convert everything to low and case insensitive
        )

        # Step 3: 
        ldf = self._links_df() # get the links table, which journal belongs to which category
        if ldf.empty or "category" not in ldf.columns:
            return []
        # Keep only link rows whose category is one of the matched names filter the links table
        ldf_filtered = ldf[ 
            ldf["category"].astype(str).str.strip().str.lower().isin(matching_cat_names)
        ]
        if ldf_filtered.empty:
            return []

        # Step 4: join no-publisher journals with the filtered link rows
        # join the two tables by ISSN 
        joined = self._join_on_ids(no_pub_df, ldf_filtered)
        if joined.empty:
            return []

        # Step 5: convert surviving DataFrame rows into Journal objects 
        return self._journals_from_df(joined)

    def getDiamondJournalsInAreasAndCategoriesWithQuartile(self, areas: Set[str], categories: Set[str], quartiles: Set[str]) -> List[Journal]:
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []
        # 规范化查询集合
        # Normalize all three input sets for consistent matching
        norm_areas = {a.strip().lower() for a in areas} if areas else set()
        norm_cats = {c.strip().lower() for c in categories} if categories else set()
        norm_qs = {q.strip().upper() for q in quartiles} if quartiles else set()
        area_mask = ldf["area"].notna() if not areas else ldf["area"].astype(str).str.strip().str.lower().isin(norm_areas)
        cat_mask = ldf["category"].notna() if not categories else ldf["category"].astype(str).str.strip().str.lower().isin(norm_cats)
        q_mask = (ldf["quartile"].notna() | ldf["quartile"].isna()) if not quartiles else ldf["quartile"].astype(str).str.strip().str.upper().isin(norm_qs)
        j_area = self._join_on_ids(jdf, ldf.loc[area_mask])
        j_catq = self._join_on_ids(jdf, ldf.loc[cat_mask & q_mask])
        # 统一转为字符串 Set 以保证交集操作不崩溃
        # Convert to string sets to ensure intersection works without type errors
        ids_area = set(j_area["id"].astype(str).unique()) if "id" in j_area.columns else set()
        ids_catq = set(j_catq["id"].astype(str).unique()) if "id" in j_catq.columns else set()
        # Keep only journals that appear in BOTH area and category+quartile subsets
        ok_ids = ids_area.intersection(ids_catq)
        if not ok_ids:
            return []
        final = jdf.loc[jdf["id"].astype(str).isin(ok_ids)].copy()
        # 修复：宽容处理 APC = False 的字符串逻辑，拯救因字符形式不匹配而丢掉的 Diamond 期刊
        # Diamond OA: keep only journals with no APC fee.
        # Handles multiple string forms of False ('false', 'no', '0', 'f')
        # to avoid dropping journals where APC is stored as a string.
        if "apc" in final.columns:
            def is_diamond(val):
                if pd.isna(val) or str(val).strip() == "":
                    return False
                return str(val).strip().lower() in ['false', 'no', '0', 'f']
            final = final[final["apc"].apply(is_diamond)]
        return self._journals_from_df(final)
