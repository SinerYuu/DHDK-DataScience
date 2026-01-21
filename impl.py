from typing import List, Optional, Set, Dict, Any, Tuple
from collections import OrderedDict
import json
import sqlite3
import os
import re
import pandas as pd
import requests
import traceback

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

    def getIds(self) -> Set[str]:
        """Return a set containing this area's ID."""
        return {self._id} if self._id else set()


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

    def getIds(self) -> Set[str]:
        """Return a set containing this category's ID."""
        return {self._id} if self._id else set()


class Journal(IdentifiableEntity):
    """
    Represents a DOAJ (Directory of Open Access Journals) Journal mapped to schema.org:Periodical.
    
    Data Model:
    - Represents a single academic journal with metadata about open access publishing
    - Can be associated with multiple academic Categories and Areas through the knowledge graph
    - Primary identifier is either an ISSN (International Standard Serial Number) or the journal title
    
    Key Properties:
    - id: ISSN (print or electronic) or journal title as fallback
    - title: Official journal name
    - publisher: Organization that publishes the journal
    - license: Open access license (e.g., CC BY, CC BY-NC, etc.)
    - apc: Boolean indicating if journal charges Article Processing Charges
    - doaj_seal: Boolean indicating if journal has DOAJ Seal (high open access quality)
    - languages: List of manuscript acceptance languages
    - categories: Academic subject categories (populated via knowledge graph links)
    - areas: Broader research areas (populated via knowledge graph links)
    
    Relationships:
    - Many-to-many with Category (journals belong to multiple categories, categories have multiple journals)
    - Many-to-many with Area (journals belong to multiple areas, areas contain multiple journals)
    - These relationships are maintained in the SCImago knowledge graph
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
        # Initialize parent IdentifiableEntity with id and title (name)
        super().__init__(id, title)
        
        # Store journal metadata from DOAJ CSV
        self._publisher = publisher  # Publishing organization
        self._license = license  # Open access license type
        self._apc = apc  # Whether journal charges for article processing
        self._doaj_seal = doaj_seal  # Quality certification from DOAJ
        self._languages = languages or []  # Supported manuscript languages
        
        # Store relationships to other entities (populated by knowledge graph)
        self._categories: OrderedDict[str, Category] = OrderedDict()  # Maps category ID -> Category object
        self._areas: OrderedDict[str, Area] = OrderedDict()  # Maps area ID -> Area object

    # --- Basic field accessors (DOAJ journal metadata) ---
    
    def getTitle(self) -> str:
        """Returns the official journal title from DOAJ data."""
        return self._name

    def hasTitle(self) -> bool:
        """Checks if journal has a non-empty title."""
        return bool(self._name)

    def getPublisher(self) -> str:
        """Returns the publishing organization extracted from DOAJ CSV."""
        return self._publisher

    def hasPublisher(self) -> bool:
        """Checks if publisher information is available."""
        return bool(self._publisher)

    def getLicense(self) -> str:
        """
        Returns the open access license type.
        Common values from DOAJ:
        - "CC BY": Creative Commons Attribution (most permissive)
        - "CC BY-NC": Non-commercial use only
        - "CC BY-SA": Share-alike requirement
        - "Publisher's own license": Custom license
        """
        return self._license

    def hasLicense(self) -> bool:
        """Checks if license information is available."""
        return bool(self._license)

    def getAPC(self) -> Optional[bool]:
        """
        Returns Article Processing Charge status.
        - True: Journal charges authors for publishing
        - False: Journal is free to publish (True OA)
        - None: Information not available
        
        Importance: Diamond OA journals have APC=False (free for authors)
        """
        return self._apc

    def hasAPC(self) -> bool:
        """Checks if APC information is available (not None)."""
        return self._apc is not None

    def getDOAJSeal(self) -> Optional[bool]:
        """
        Returns DOAJ Seal status.
        - True: Journal meets DOAJ's high standards for open access
        - False: Journal doesn't have DOAJ Seal
        - None: Information not available
        
        The DOAJ Seal is awarded to journals meeting strict criteria for:
        - Open access compliance
        - Publishing best practices
        - Transparency
        """
        return self._doaj_seal

    def hasDOAJSeal(self) -> bool:
        """Checks if DOAJ Seal information is available (not None)."""
        return self._doaj_seal is not None

    def getLanguages(self) -> List[str]:
        """
        Returns list of languages in which the journal accepts manuscripts.
        Example: ['English', 'Spanish', 'Portuguese']
        """
        return list(self._languages)

    def hasLanguages(self) -> bool:
        """Checks if any manuscript languages are specified."""
        return len(self._languages) > 0

    # --- Knowledge Graph relationship accessors ---

    def addCategory(self, category: Category) -> None:
        """
        Links a Category to this Journal (many-to-many relationship).
        Each ISSN from SCImago can be assigned to multiple categories.
        
        Example: A journal "Nature" might be in:
        - Biology (Q1)
        - Environmental Science (Q2)
        - Multidisciplinary (Q1)
        """
        if category and category.getId() not in self._categories:
            self._categories[category.getId()] = category

    def getCategories(self) -> List[Category]:
        """
        Returns all Categories linked to this Journal via the knowledge graph.
        Used to find journal's research domains and quartile rankings.
        """
        return list(self._categories.values())

    def hasCategories(self) -> bool:
        """Checks if the Journal has at least one linked Category."""
        return len(self._categories) > 0

    def addArea(self, area: Area) -> None:
        """
        Links an Area to this Journal (many-to-many relationship).
        Each ISSN from SCImago can be assigned to broader research areas.
        
        Example: Categories like "Hematology" belong to Area "Medicine"
        """
        if area and area.getId() not in self._areas:
            self._areas[area.getId()] = area

    def getAreas(self) -> List[Area]:
        """
        Returns all Research Areas linked to this Journal via the knowledge graph.
        Useful for finding journals in broad discipline areas like Engineering or Medicine.
        """
        return list(self._areas.values())

    def hasAreas(self) -> bool:
        """Checks if the Journal has at least one linked Area."""
        return len(self._areas) > 0

    def getIds(self) -> Set[str]:
        """Return a set containing this journal's ID."""
        return {self._id} if self._id else set()

    def getIds(self) -> Set[str]:
        """Return a set containing this journal's ID."""
        return {self._id} if self._id else set()

# -------------------- Basic Handlers (upload + query) --------------------

# The parent class for all handler types.
class Handler:
    def __init__(self):
        self.dbPathOrUrl: str = ""  # Initialize the database path or URL.

    def getDbPathOrUrl(self) -> str:
        return self.dbPathOrUrl  # Return the current database path or URL.

    def setDbPathOrUrl(self, val: str) -> bool:
        self.dbPathOrUrl = val  # Set the database path or URL.
        _ensure_registry(val)  # Ensure registry exists for the new path.
        return True  # Return True indicating the operation succeeded.

# Abstract subclass for data ingestion (uploading data).
class UploadHandler(Handler):
    def pushDataToDb(self, file_path: str) -> bool:  # Must be implemented by specific uploaders.
        raise NotImplementedError()  # Raise an error if not overridden.

# Abstract subclass for data retrieval (querying data).
class QueryHandler(Handler):
    def getById(self, id: str) -> pd.DataFrame:  # Must be implemented by subclasses.
        raise NotImplementedError()  # Raise an error if not overridden.

# -------------------- Graph/Blazegraph helpers --------------------

SCHEMA = Namespace("https://schema.org/")  # Define the schema namespace.

def _bool_from_str(v: Any) -> Optional[bool]:
    if isinstance(v, bool):  # If value is already a boolean, return it.
        return v
    if isinstance(v, str):  # If value is a string, parse it into a boolean.
        w = v.strip().lower()
        if w in {"true", "yes", "y", "1"}: return True
        if w in {"false", "no", "n", "0"}: return False
    return None  # Return None if the string is not valid.

def _build_journal_uri(issn: str) -> URIRef:
    # Constructs a URI for a journal using its ISSN.
    return URIRef(f"http://example.org/periodical/{issn}")

class _BlazegraphClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint  # Set the Blazegraph endpoint URL.

    def upload_graph(self, g: Graph) -> bool:
        try:
            triples_data = g.serialize(format='nt')  # Serialize RDF graph to N-Triples format.
            insert_query = f"INSERT DATA {{ {triples_data} }}"  # Create SPARQL insert query.
            response = requests.post(self.endpoint, data=insert_query, headers={'Content-Type': 'application/sparql-update'})  # Send query to Blazegraph.
            
            # Check if the upload was successful (HTTP status 200 or 204).
            if response.status_code in [200, 204]:
                print(f"[OK] Successfully uploaded {len(g)} triples.")
                return True
            else:
                print(f"[ERROR] Blazegraph returned status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to upload graph: {e}")
            return False

    def select(self, query: str) -> List[Dict[str, Any]]:
        try:
            store = SPARQLStore(self.endpoint)  # Connect to Blazegraph using SPARQL store.
            g = Graph(store=store)  # Create an RDF graph using the SPARQL store.
            rows = []

            # Execute the SPARQL query and store the results.
            for row in g.query(query):
                binding = {}
                for var, val in row.asdict().items():
                    binding[var] = str(val) if val is not None else None
                rows.append(binding)
            return rows
        except Exception as e:
            print(f"[ERROR] SPARQL query failed: {e}")
            return []  # Return an empty list if the query fails.

# -------------------- Uploaders --------------------

class JournalUploadHandler(UploadHandler):
    def pushDataToDb(self, file_path: str) -> bool:
        reg = _ensure_registry(self.dbPathOrUrl)  # Ensure registry is set.
        try:
            path = file_path  # Set the file path.
            if not os.path.isfile(path) and os.path.isfile(os.path.join(".", path)):
                path = os.path.join(".", path)  # Resolve relative file paths.
            if not os.path.isfile(path):  # Check if the file exists.
                print(f"[WARNING] File not found: {path}")
                reg["journals"] = pd.DataFrame(columns=["id","title","publisher","license","apc","doaj_seal","languages"])
                return True  # Return success even if file is missing.

            df_raw = pd.read_csv(path, dtype=str, keep_default_na=False)  # Read CSV file as strings.

            cols_lower = {c.lower(): c for c in df_raw.columns}  # Normalize column names to lowercase.
            def pick(*keys):
                """Find column name by checking multiple possible names."""
                for k in keys:
                    for low, orig in cols_lower.items():
                        if k in low:
                            return orig
                return None

            # Map columns to standardized fields.
            col_issn = pick("issn", "eissn", "pissn", "journal id", "identifier")
            col_title = pick("title")
            col_publisher = pick("publisher")
            col_license = pick("license")
            col_apc = pick("apc", "article processing charge", "processing charges")
            col_seal = pick("seal", "doaj")
            col_lang = pick("language")

            # Build RDF graph.
            g = Graph()
            g.bind("schema", SCHEMA)  # Bind schema.org prefix.

            fallback_rows = []  # List to store rows for local cache.

            for _, row in df_raw.iterrows():
                # Extract and normalize each journal's data.
                issn = (str(row[col_issn]).strip() if col_issn and str(row[col_issn]).strip() else "")
                title = str(row[col_title]).strip() if col_title else ""
                publisher = str(row[col_publisher]).strip() if col_publisher else ""
                license_ = str(row[col_license]).strip() if col_license else ""
                apc = _bool_from_str(row[col_apc]) if col_apc else None
                seal = _bool_from_str(row[col_seal]) if col_seal else None
                langs_raw = str(row[col_lang]).strip() if col_lang else ""
                languages = [l.strip() for l in langs_raw.split(", ")] if langs_raw else []

                # Skip rows with missing data (ISSN or title).
                if not issn and not title:
                    continue

                # Add row to fallback cache.
                fallback_rows.append({
                    "id": issn or title,  # Use ISSN as primary ID.
                    "title": title,
                    "publisher": publisher,
                    "license": license_,
                    "apc": apc,
                    "doaj_seal": seal,
                    "languages": languages,
                })

                if issn:  # Only process if ISSN is available.
                    s = _build_journal_uri(issn)  # Build URI for the journal.
                    g.add((s, RDF.type, SCHEMA.Periodical))  # Add type (journal).
                    g.add((s, SCHEMA.issn, Literal(issn)))  # Add ISSN property.
                    if title:
                        g.add((s, SCHEMA.name, Literal(title)))  # Add title property.
                    if publisher:
                        g.add((s, SCHEMA.publisher, Literal(publisher)))  # Add publisher property.
                    if license_:
                        g.add((s, SCHEMA.license, Literal(license_)))  # Add license property.
                    for lang in languages:
                        g.add((s, SCHEMA.inLanguage, Literal(lang)))  # Add language property.

                    # Add additional properties (APC and DOAJ Seal).
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

            ok = _BlazegraphClient(self.dbPathOrUrl).upload_graph(g)  # Upload RDF graph.

            # Cache the data locally regardless of upload success.
            reg["journals"] = pd.DataFrame.from_records(fallback_rows).reset_index(drop=True)

            return ok
        except Exception as e:
            print(f"[ERROR] Exception in pushDataToDb: {e}")
            reg["journals"] = pd.DataFrame(columns=["id","title","publisher","license","apc","doaj_seal","languages"])
            return False  # Return failure on exception.
            
class CategoryUploadHandler(UploadHandler):
    """
    Handles loading SCImago-like JSON data into a relational (SQLite) database.
    This version keeps the same method names and behavior as the project requires,
    but writes data both to the in-memory registry and to SQLite for persistence.
    """

    def pushDataToDb(self, file_path: str) -> bool:
        reg = _ensure_registry(self.dbPathOrUrl)
        db_path = self.dbPathOrUrl
        path = file_path

        # Resolve file path
        if not os.path.isfile(path) and os.path.isfile(os.path.join(".", path)):
            path = os.path.join(".", path)

        # If file missing, create empty tables
        if not os.path.isfile(path):
            reg["categories"] = pd.DataFrame(columns=["id", "quartile"])
            reg["areas"] = pd.DataFrame(columns=["id"])
            reg["links"] = pd.DataFrame(columns=["issn", "category", "quartile", "area"])
            conn = sqlite3.connect(db_path)
            reg["categories"].to_sql("categories", conn, if_exists="replace", index=False)
            reg["areas"].to_sql("areas", conn, if_exists="replace", index=False)
            reg["links"].to_sql("links", conn, if_exists="replace", index=False)
            conn.close()
            return True

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cat_rows, area_rows, link_rows = [], [], []

            for entry in data:
                idents = entry.get("identifiers", [])
                categories = entry.get("categories", [])
                areas = entry.get("areas", [])

                for cat in categories:
                    cid = str(cat.get("id", "")).strip()
                    quart = (str(cat.get("quartile", "")).strip() or None)
                    if cid:
                        cat_rows.append({"id": cid, "quartile": quart})
                        for issn in idents:
                            link_rows.append(
                                {"issn": issn, "category": cid, "quartile": quart, "area": None}
                            )

                for ar in areas:
                    aid = str(ar).strip()
                    if aid:
                        area_rows.append({"id": aid})
                        for issn in idents:
                            link_rows.append(
                                {"issn": issn, "category": None, "quartile": None, "area": aid}
                            )

            # Build in-memory DataFrames
            reg["categories"] = pd.DataFrame.from_records(cat_rows).drop_duplicates().reset_index(drop=True)
            reg["areas"] = pd.DataFrame.from_records(area_rows).drop_duplicates().reset_index(drop=True)
            reg["links"] = pd.DataFrame.from_records(link_rows).drop_duplicates().reset_index(drop=True)

            # Write to SQLite
            conn = sqlite3.connect(db_path)
            reg["categories"].to_sql("categories", conn, if_exists="replace", index=False)
            reg["areas"].to_sql("areas", conn, if_exists="replace", index=False)
            reg["links"].to_sql("links", conn, if_exists="replace", index=False)
            conn.close()

            print(f"[OK] Category data successfully stored in {db_path}")
            return True

        except Exception as e:
            print(f"[Error] Failed to process {file_path}: {e}")
            return False


# -------------------- Query Handlers --------------------

class QueryHandler(Handler):
    def getById(self, id_value: str) -> pd.DataFrame:
        raise NotImplementedError()


class JournalQueryHandler(QueryHandler):
    """
    Query handler for journals using Blazegraph with fallback to local cache.
    
    Architecture:
    1. Primary: Queries Blazegraph using SPARQL over RDF triples
    2. Fallback: Uses local pandas DataFrame cache
    
    Query Strategy:
    - First attempts to use cached data (most reliable, fastest)
    - Falls back to SPARQL queries if cache is empty
    - Returns empty DataFrame if both fail
    
    Benefits of this approach:
    - Works without Blazegraph server running
    - Caching improves performance
    - Graceful degradation
    """
    
    def _client(self) -> _BlazegraphClient:
        """Create a Blazegraph client for this database URL."""
        return _BlazegraphClient(self.dbPathOrUrl)

    def _fallback_df(self) -> pd.DataFrame:
        """Get the cached journal DataFrame from registry."""
        return _ensure_registry(self.dbPathOrUrl).get("journals", pd.DataFrame())

    @staticmethod
    def _aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Aggregate multiple SPARQL result rows into a single row per journal.
        
        Why aggregation is needed:
        - SPARQL queries may return multiple rows for one journal
        - Example: A journal in multiple languages returns one row per language
        - This method consolidates them into one journal row
        
        Process:
        1. Group results by ISSN (unique journal identifier)
        2. For each ISSN, collect all values (title, languages, etc.)
        3. Merge multiple rows into single comprehensive row
        4. Handle boolean conversions for APC and DOAJ Seal
        
        Example input:
        [
            {'issn': '1542-4863', 'title': 'Nature', 'lang': 'English'},
            {'issn': '1542-4863', 'title': 'Nature', 'lang': 'Spanish'},
        ]
        
        Example output:
        [
            {'id': '1542-4863', 'title': 'Nature', 'languages': ['English', 'Spanish']}
        ]
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
        """
        Execute a journal query by filtering the cache or running SPARQL.
        
        Two-tier query strategy:
        1. First: Filter local cache (fastest, always works)
        2. Fallback: Execute SPARQL query (more powerful but requires Blazegraph)
        
        Args:
            where_filter: SPARQL FILTER clause (e.g., "FILTER (BOUND(?title) && ...)")
            limit: Maximum number of results to return
            
        Returns:
            DataFrame with journal data, or empty DataFrame if no results
            
        Filtering in cache uses regex pattern matching on text fields.
        This provides fast, local filtering without server dependency.
        """
        # Try fallback cache first (most reliable)
        fb_df = self._fallback_df().copy()
        if not fb_df.empty:
            # Apply filters directly to fallback dataframe
            filtered_df = fb_df.copy()
            
            # Parse filter if needed (extract search terms from SPARQL filter syntax)
            if "CONTAINS" in where_filter and "title" in where_filter:
                # Extract search text from FILTER
                import re
                match = re.search(r'CONTAINS\(.*?LCASE\("(.+?)"\)', where_filter)
                if match:
                    search_text = match.group(1).lower()
                    filtered_df = filtered_df[
                        filtered_df.get("title", pd.Series(dtype=str))
                        .astype(str).str.lower().str.contains(search_text, na=False)
                    ]
            elif "CONTAINS" in where_filter and "publisher" in where_filter:
                # Extract search text for publisher
                import re
                match = re.search(r'CONTAINS\(.*?LCASE\("(.+?)"\)', where_filter)
                if match:
                    search_text = match.group(1).lower()
                    filtered_df = filtered_df[
                        filtered_df.get("publisher", pd.Series(dtype=str))
                        .astype(str).str.lower().str.contains(search_text, na=False)
                    ]
            elif "apc" in where_filter and "true" in where_filter:
                # Filter journals with APC = true
                filtered_df = filtered_df[filtered_df.get("apc", False) == True]
            elif "seal" in where_filter and "true" in where_filter:
                # Filter journals with DOAJ Seal = true
                filtered_df = filtered_df[filtered_df.get("doaj_seal", False) == True]
            elif "license" in where_filter:
                # Extract licenses from filter and match case-insensitively
                import re
                licenses = re.findall(r'LCASE\("(.+?)"\)', where_filter)
                if licenses:
                    filtered_df = filtered_df[
                        filtered_df.get("license", pd.Series(dtype=str))
                        .astype(str).str.lower().isin([l.lower() for l in licenses])
                    ]
            
            if limit:
                filtered_df = filtered_df.head(limit)
            
            return filtered_df.reset_index(drop=True) if not filtered_df.empty else pd.DataFrame()
        
        # Fallback to SPARQL if cache is empty
        # (This section executes if Blazegraph is available and cache is empty)
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
            df = self._aggregate_rows(rows)
            return df
        except Exception as e:
            traceback.print_exc()
            # fallback
            return self._fallback_df().copy()

    def getById(self, id_value: str) -> pd.DataFrame:
        """
        Find a journal by ISSN or title.
        
        Search strategy:
        1. Query Blazegraph/cache for matches on ISSN or title
        2. Prioritize exact ISSN matches
        3. Fall back to title matching if ISSN match not found
        4. Return first matching journal or empty DataFrame if no match
        
        Args:
            id_value: ISSN or journal title to search for
            
        Returns:
            DataFrame with single journal row, or empty if not found
        """
        # Match by ISSN or by name as a fallback
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
        """
        Retrieve all journals in the database.
        
        Returns:
            DataFrame with all journals (21,307+ journals from DOAJ)
        """
        return self._select_df()

    def getJournalsWithTitle(self, text: str) -> pd.DataFrame:
        """
        Find journals whose title contains the search text (case-insensitive).
        
        Example: getJournalsWithTitle('Nature') returns all journals with 'Nature' in title
        
        Args:
            text: Text to search for in journal titles
            
        Returns:
            DataFrame with matching journals
        """
        where = f'FILTER (BOUND(?title) && CONTAINS(LCASE(STR(?title)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsPublishedBy(self, text: str) -> pd.DataFrame:
        """
        Find journals published by organizations matching the search text.
        
        Example: getJournalsPublishedBy('Springer') returns journals published by Springer
        
        Args:
            text: Text to search for in publisher names
            
        Returns:
            DataFrame with matching journals
        """
        where = f'FILTER (BOUND(?publisher) && CONTAINS(LCASE(STR(?publisher)), LCASE("{text}")))'
        return self._select_df(where_filter=where)

    def getJournalsWithLicense(self, licenses: Set[str]) -> pd.DataFrame:
        """
        Find journals with specific open access licenses.
        
        Common licenses:
        - "CC BY": Most permissive, allows commercial use
        - "CC BY-NC": Non-commercial only
        - "CC BY-SA": Must share-alike
        - "CC BY-NC-SA": Combination of NC and SA restrictions
        
        Args:
            licenses: Set of license types to search for
            
        Returns:
            DataFrame with journals having any of the specified licenses
        """
        if not licenses:
            return self.getAllJournals()
        filters = " || ".join([f'LCASE(STR(?license)) = LCASE("{lic}")' for lic in licenses])
        where = f"FILTER (BOUND(?license) && ({filters}))"
        return self._select_df(where_filter=where)

    def getJournalsWithAPC(self) -> pd.DataFrame:
        """
        Find journals that charge Article Processing Charges (APC).
        
        These are journals that charge authors for publishing their papers.
        Opposite of these are Diamond OA (free for authors and readers).
        
        Returns:
            DataFrame with journals where APC = true
        """
        where = "FILTER (BOUND(?apc) && (?apc = true))"
        return self._select_df(where_filter=where)

    def getJournalsWithDOAJSeal(self) -> pd.DataFrame:
        """
        Find journals with DOAJ Seal (high-quality open access indicators).
        
        DOAJ Seal is awarded to journals demonstrating:
        - Compliance with DOAJ standards
        - Best practices in open access publishing
        - Strong editorial processes
        - Full text access
        
        Returns:
            DataFrame with journals where DOAJ Seal = true
        """
        where = "FILTER (BOUND(?seal) && (?seal = true))"
        return self._select_df(where_filter=where)


class CategoryQueryHandler(QueryHandler):
    """
    Query handler for categories and areas.
    Reads data directly from SQLite but maintains the original interface.
    """

    def _reg(self) -> Dict[str, Any]:
        return _ensure_registry(self.dbPathOrUrl)

    def _df_cat(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            df = pd.read_sql_query("SELECT * FROM categories", conn)
            conn.close()
            return df
        except Exception:
            return self._reg().get("categories", pd.DataFrame())

    def _df_area(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            df = pd.read_sql_query("SELECT * FROM areas", conn)
            conn.close()
            return df
        except Exception:
            return self._reg().get("areas", pd.DataFrame())

    def _df_links(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.dbPathOrUrl)
            df = pd.read_sql_query("SELECT * FROM links", conn)
            conn.close()
            return df
        except Exception:
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
            return pd.DataFrame(columns=["id", "quartile"])
        
        # Get all ISSNs that are assigned to the specified areas
        if areas:
            area_issns = df_links.loc[df_links["area"].isin(areas), "issn"].unique()
        else:
            # If no areas specified, get all ISSNs that have any area
            area_issns = df_links.loc[df_links["area"].notna(), "issn"].unique()
        
        if len(area_issns) == 0:
            return pd.DataFrame(columns=["id", "quartile"])
        
        # Get all categories for those ISSNs
        cats = df_links.loc[
            (df_links["issn"].isin(area_issns)) & 
            (df_links["category"].notna())
        ][["category", "quartile"]].drop_duplicates()
        
        cats = cats.rename(columns={"category": "id"})
        return cats.drop_duplicates(subset=["id"]).reset_index(drop=True)

    def getAreasAssignedToCategories(self, categories: Set[str]) -> pd.DataFrame:
        df_links = self._df_links()
        if df_links.empty:
            return pd.DataFrame(columns=["id"])
        
        # Get all ISSNs that are assigned to the specified categories
        if categories:
            cat_issns = df_links.loc[df_links["category"].isin(categories), "issn"].unique()
        else:
            # If no categories specified, get all ISSNs that have any category
            cat_issns = df_links.loc[df_links["category"].notna(), "issn"].unique()
        
        if len(cat_issns) == 0:
            return pd.DataFrame(columns=["id"])
        
        # Get all areas for those ISSNs
        areas = df_links.loc[
            (df_links["issn"].isin(cat_issns)) & 
            (df_links["area"].notna())
        ][["area"]].drop_duplicates()
        
        areas = areas.rename(columns={"area": "id"})
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
                    # Drop rows that are completely NA
                    clean = df.replace("", pd.NA).dropna(how="all")
                    if not clean.empty:
                        jdfs.append(clean)
            except Exception:
                continue

        if jdfs:
            jdf = self._combine_df(jdfs)
            jdf = jdf.replace("", pd.NA).dropna(how="all")
            if not jdf.empty:
                # Try exact match first
                exact = jdf.loc[
                    (jdf["id"].astype(str).str.lower() == str(identifier).lower()) |
                    (jdf.get("title", pd.Series(dtype=str)).astype(str).str.lower() == str(identifier).lower())
                ]
                if not exact.empty:
                    js = self._journals_from_df(exact.head(1))
                    if js and js[0].hasId():
                        return js[0]
                # If no exact match, return first journal if it has an ID
                js = self._journals_from_df(jdf.head(1))
                if js and js[0].hasId():
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
                # Check if it's a category (has quartile column)
                if "quartile" in cdf.columns:
                    cats = self._categories_from_df(cdf.head(1))
                    if cats and cats[0].hasId():
                        return cats[0]
                # Otherwise treat as area
                ars = self._areas_from_df(cdf.head(1))
                if ars and ars[0].hasId():
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
        """
        Assemble and combine all 'link tables' from category query handlers.
        Each link table connects journals with categories, quartiles, or areas.
        """
        frames = []
        for h in self.categoryQuery:
            if isinstance(h, CategoryQueryHandler):
                frames.append(h._df_links())
        return self._combine_df(frames)

    def _journal_df(self) -> pd.DataFrame:
        """
        Assemble and combine all journal tables from journal query handlers.
        Each handler contributes its own journal metadata table.
        """
        frames = [h.getAllJournals() for h in self.journalQuery]
        return self._combine_df(frames)

    def _join_on_ids(self, jdf: pd.DataFrame, ldf: pd.DataFrame) -> pd.DataFrame:
        """
        Join the journal table (jdf) and link table (ldf) using matching identifiers.
        """
        if jdf.empty or ldf.empty:
            return pd.DataFrame()
        if "id" in jdf.columns and "issn" in ldf.columns:
            return jdf.merge(ldf, left_on="id", right_on="issn", how="inner")
        return pd.DataFrame()

    def getJournalsInCategoriesWithQuartile(self, categories: Set[str], quartiles: Set[str]) -> List[Journal]:
        """
        Return journals that belong to given categories and quartile levels.
        """
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []
        
        # Category mask
        # If 'categories' is empty, keep all rows with a non-null category.
        # Otherwise, keep rows whose category value is inside the provided set.
        cat_mask = ldf["category"].notna() if not categories else ldf["category"].isin(categories)
        
        # Quartile mask
        # If 'quartiles' is empty, keep all rows (including missing ones).
        # Otherwise, convert to uppercase and compare (case-insensitive).
        q_mask = (ldf["quartile"].notna() | ldf["quartile"].isna()) if not quartiles else ldf["quartile"].astype(str).str.upper().isin({q.upper() for q in quartiles})
        
        # Combine both masks, keeps only rows that satisfy BOTH conditions
        lsub = ldf.loc[cat_mask & q_mask] 
        
        # Join filtered links with journal table on ID/ISSN
        joined = self._join_on_ids(jdf, lsub)

        # Drop duplicate journal IDs to avoid multiple matches
        joined = joined.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return self._journals_from_df(joined)

    def getJournalsInAreasWithLicense(self, areas: Set[str], licenses: Set[str]) -> List[Journal]:
        """
        Return journals belonging to specific areas and having the specified license type.
        """
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []
        
        # Area mask
        # If 'areas' is empty, keep all non-null rows.
        # Otherwise, keep rows whose 'area' matches one of the specified values.
        area_mask = ldf["area"].notna() if not areas else ldf["area"].isin(areas)

        # Filter the link table using the mask
        lsub = ldf.loc[area_mask]

        # Join filtered links with journal table on ID/ISSN
        joined = self._join_on_ids(jdf, lsub)
        
        # Additional license filtering (case-insensitive)
        if licenses and "license" in joined.columns:
            joined = joined.loc[joined["license"].astype(str).str.lower().isin({x.lower() for x in licenses})]
        
        # Remove duplicates by journal ID and convert to Journal objects
        joined = joined.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return self._journals_from_df(joined)

    def getDiamondJournalsInAreasAndCategoriesWithQuartile(self, areas: Set[str], categories: Set[str], quartiles: Set[str]) -> List[Journal]:
        """
        Return journals that:
        - belong to the given areas
        - belong to the given categories and quartiles
        - have no APC fee (Diamond Open Access)
        """
        jdf = self._journal_df()
        ldf = self._links_df()
        if jdf.empty or ldf.empty:
            return []

        area_mask = ldf["area"].notna() if not areas else ldf["area"].isin(areas)
        cat_mask = ldf["category"].notna() if not categories else ldf["category"].isin(categories)
        q_mask = (ldf["quartile"].notna() | ldf["quartile"].isna()) if not quartiles else ldf["quartile"].astype(str).str.upper().isin({q.upper() for q in quartiles})

        j_area = self._join_on_ids(jdf, ldf.loc[area_mask])
        j_catq = self._join_on_ids(jdf, ldf.loc[cat_mask & q_mask])

        # Get unique journal IDs from each subset
        ids_area = set(j_area["id"].unique()) if "id" in j_area.columns else set()
        ids_catq = set(j_catq["id"].unique()) if "id" in j_catq.columns else set()

        # Keep only IDs that appear in BOTH sets (intersection)
        ok_ids = ids_area.intersection(ids_catq)
        if not ok_ids:
            return []

        # Extract matching journals from the main journal table
        final = jdf.loc[jdf["id"].isin(ok_ids)].copy()

        # Keep only journals where APC == False (no author fees)
        if "apc" in final.columns:
            final = final.loc[final["apc"] == False]

        final = final.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return self._journals_from_df(final)
