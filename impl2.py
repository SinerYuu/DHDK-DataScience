# impl.py
# Minimal reference implementation for the DS project described in the prompt.
# Requires: pandas, requests (pip install pandas requests)

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Dict, Tuple
import json
import csv
import re
import sqlite3
import requests
import pandas as pd
from io import StringIO

# =======================
# Utilities & Namespaces
# =======================

EX = "http://example.org/"
DCT = "http://purl.org/dc/terms/"
SCHEMA = "https://schema.org/"

def _slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "unk"

def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    vs = str(v).strip().lower()
    return vs in {"1","true","t","yes","y","✓","✔","x"}

def _split_multi(s: Optional[str]) -> List[str]:
    if not s:
        return []
    # DOAJ FAQ says languages are joined by ", " (comma+space)
    # but be robust to other separators
    if ", " in s:
        parts = s.split(", ")
    elif ";" in s:
        parts = s.split(";")
    else:
        parts = s.split(",")
    return [p.strip() for p in parts if p.strip()]

def _first_non_empty(*vals) -> str:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""

# =======================
# Data model classes (UML)
# =======================

@dataclass
class IdentifiableEntity:
    ids: List[str]  # [1..*] in UML
    def getIds(self) -> List[str]:
        return self.ids

@dataclass
class Area(IdentifiableEntity):
    # Area has only 'id' in UML
    def __init__(self, id: str):
        super().__init__([id])

@dataclass
class Category(IdentifiableEntity):
    quartile: Optional[str] = None
    def __init__(self, id: str, quartile: Optional[str] = None):
        super().__init__([id])
        self.quartile = quartile
    def getQuartile(self) -> Optional[str]:
        return self.quartile

@dataclass
class Journal(IdentifiableEntity):
    title: str = ""
    languages: List[str] = None
    publisher: Optional[str] = None
    seal: bool = False
    licence: str = ""
    apc: bool = False
    _categories: List[Category] = None
    _areas: List[Area] = None

    def getTitle(self) -> str:
        return self.title
    def getLanguages(self) -> List[str]:
        return self.languages or []
    def getPublisher(self) -> Optional[str]:
        return self.publisher
    def hasDOAJSeal(self) -> bool:
        return self.seal
    def getLicence(self) -> str:
        return self.licence
    def hasAPC(self) -> bool:
        return self.apc
    def getCategories(self) -> List[Category]:
        return self._categories or []
    def getAreas(self) -> List[Area]:
        return self._areas or []

# =======================
# Base handlers (UML)
# =======================

class Handler:
    def __init__(self) -> None:
        self._dbPathOrUrl: str = ""
    def getDbPathOrUrl(self) -> str:
        return self._dbPathOrUrl
    def setDbPathOrUrl(self, path_or_url: str) -> None:
        self._dbPathOrUrl = path_or_url

class UploadHandler(Handler):
    def pushDataToDb(self, path: str) -> bool:
        raise NotImplementedError

class QueryHandler(Handler):
    def getById(self, idd: str) -> pd.DataFrame:
        """Return a DataFrame with the identifiable entity matching id."""
        raise NotImplementedError

# ============================================
# Relational side: Categories, Areas (SQLite)
# ============================================

class CategoryUploadHandler(UploadHandler):
    """
    Expects a JSON file (scimago-like). We support two simple shapes:

    A) list of records:
       [{"area":"Computer Science", "category":"Artificial Intelligence", "quartile":"Q1"}, ...]

    B) dict with arrays:
       {"areas":[...], "categories":[...], "links":[{"area":"...","category":"..."}], ...}

    We normalize to three tables:
      area(id TEXT PRIMARY KEY)
      category(id TEXT PRIMARY KEY, quartile TEXT)
      area_category(area_id TEXT, category_id TEXT, PRIMARY KEY(area_id, category_id))
    """
    def _conn(self):
        return sqlite3.connect(self.getDbPathOrUrl())

    def pushDataToDb(self, path: str) -> bool:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS area(
            id TEXT PRIMARY KEY
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS category(
            id TEXT PRIMARY KEY,
            quartile TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS area_category(
            area_id TEXT,
            category_id TEXT,
            PRIMARY KEY(area_id, category_id),
            FOREIGN KEY(area_id) REFERENCES area(id),
            FOREIGN KEY(category_id) REFERENCES category(id)
        )""")

        def upsert_area(a: str):
            cur.execute("INSERT OR IGNORE INTO area(id) VALUES(?)", (a,))
        def upsert_cat(c: str, q: Optional[str]):
            cur.execute("INSERT OR IGNORE INTO category(id, quartile) VALUES(?,?)", (c, q))
            if q is not None:
                cur.execute("UPDATE category SET quartile=? WHERE id=? AND (quartile IS NULL OR quartile='')", (q, c))
        def link(a: str, c: str):
            cur.execute("INSERT OR IGNORE INTO area_category(area_id, category_id) VALUES(?,?)", (a, c))

        if isinstance(data, list):
            for row in data:
                area = str(row.get("area") or row.get("Area") or "").strip()
                cat = str(row.get("category") or row.get("Category") or "").strip()
                q = row.get("quartile") or row.get("Quartile")
                q = str(q).strip() if q is not None else None
                if not area or not cat:
                    continue
                upsert_area(area)
                upsert_cat(cat, q)
                link(area, cat)
        elif isinstance(data, dict):
            areas = set([str(a).strip() for a in data.get("areas", []) if str(a).strip()])
            cats = data.get("categories", [])
            links = data.get("links", [])
            for a in areas:
                upsert_area(a)
            for c in cats:
                if isinstance(c, dict):
                    cid = str(c.get("id") or c.get("name") or "").strip()
                    q = c.get("quartile")
                else:
                    cid = str(c).strip()
                    q = None
                if cid:
                    upsert_cat(cid, (str(q).strip() if q else None))
            for l in links:
                a = str(l.get("area") or "").strip()
                c = str(l.get("category") or "").strip()
                if a and c:
                    upsert_area(a)
                    upsert_cat(c, None)
                    link(a, c)
        else:
            conn.close()
            raise ValueError("Unsupported JSON structure for scimago data.")

        conn.commit()
        conn.close()
        return True


class CategoryQueryHandler(QueryHandler):
    def _conn(self):
        return sqlite3.connect(self.getDbPathOrUrl())

    # ------- Basic entity lookup
    def getById(self, idd: str) -> pd.DataFrame:
        conn = self._conn()
        cur = conn.cursor()
        # try category
        cur.execute("SELECT id, quartile FROM category WHERE id=?", (idd,))
        row = cur.fetchone()
        if row:
            df = pd.DataFrame([{"kind":"Category","id":row[0], "quartile":row[1]}])
            conn.close()
            return df
        # try area
        cur.execute("SELECT id FROM area WHERE id=?", (idd,))
        row = cur.fetchone()
        conn.close()
        if row:
            return pd.DataFrame([{"kind":"Area","id":row[0]}])
        return pd.DataFrame([])

    # ------- All categories / areas
    def getAllCategories(self) -> pd.DataFrame:
        conn = self._conn()
        df = pd.read_sql_query("SELECT DISTINCT id as id, quartile FROM category", conn)
        conn.close()
        return df

    def getAllAreas(self) -> pd.DataFrame:
        conn = self._conn()
        df = pd.read_sql_query("SELECT DISTINCT id as id FROM area", conn)
        conn.close()
        return df

    # ------- Filters
    def getCategoriesWithQuartile(self, quartiles: Set[str]) -> pd.DataFrame:
        conn = self._conn()
        if not quartiles:
            q = "SELECT DISTINCT id as id, quartile FROM category"
            df = pd.read_sql_query(q, conn)
            conn.close()
            return df
        qmarks = ",".join("?" for _ in quartiles)
        q = f"SELECT DISTINCT id as id, quartile FROM category WHERE quartile IN ({qmarks})"
        df = pd.read_sql_query(q, conn, params=tuple(quartiles))
        conn.close()
        return df

    def getCategoriesAssignedToAreas(self, area_ids: Set[str]) -> pd.DataFrame:
        conn = self._conn()
        if not area_ids:
            q = """SELECT DISTINCT c.id as id, c.quartile
                   FROM category c JOIN area_category ac ON c.id=ac.category_id"""
            df = pd.read_sql_query(q, conn)
            conn.close()
            return df
        qmarks = ",".join("?" for _ in area_ids)
        q = f"""SELECT DISTINCT c.id as id, c.quartile
                FROM category c
                JOIN area_category ac ON c.id=ac.category_id
                WHERE ac.area_id IN ({qmarks})"""
        df = pd.read_sql_query(q, conn, params=tuple(area_ids))
        conn.close()
        return df

    def getAreasAssignedToCategories(self, category_ids: Set[str]) -> pd.DataFrame:
        conn = self._conn()
        if not category_ids:
            q = """SELECT DISTINCT a.id as id
                   FROM area a JOIN area_category ac ON a.id=ac.area_id"""
            df = pd.read_sql_query(q, conn)
            conn.close()
            return df
        qmarks = ",".join("?" for _ in category_ids)
        q = f"""SELECT DISTINCT a.id as id
                FROM area a JOIN area_category ac ON a.id=ac.area_id
                WHERE ac.category_id IN ({qmarks})"""
        df = pd.read_sql_query(q, conn, params=tuple(category_ids))
        conn.close()
        return df

# ============================================
# Graph side: Journals (Blazegraph / SPARQL)
# ============================================

class _SPARQL:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
    def update(self, query: str) -> None:
        r = requests.post(self.endpoint, data={"update": query})
        r.raise_for_status()
    def select(self, query: str) -> List[Dict[str,str]]:
        r = requests.get(self.endpoint, params={"query": query, "format":"application/sparql-results+json"})
        r.raise_for_status()
        return r.json()["results"]["bindings"]

def _b(s: str) -> str:
    # Boolean literal
    return "true" if _bool(s) else "false"

def _lit(s: str) -> str:
    s = (s or "").replace('\\', '\\\\').replace('"', '\\"')
    return f"\"{s}\""

def _iri(kind: str, ident: str) -> str:
    return f"<{EX}{kind}/{_slug(ident)}>"

class JournalUploadHandler(UploadHandler):
    """
    Reads DOAJ-like CSV and creates:
     - :Journal nodes with EX:id, title, publisher, languages*, licence, apc, seal
     - :Category nodes (by name) and :hasCategory edges (so FullQueryEngine can join with SQLite)
    CSV header names are made robust (several common variants supported).
    """
    JOURNAL_CLASS = f"<{EX}Journal>"
    CATEGORY_CLASS = f"<{EX}Category>"
    PROP_ID = f"<{EX}id>"
    PROP_TITLE = f"<{EX}title>"
    PROP_PUBLISHER = f"<{EX}publisher>"
    PROP_LANG = f"<{EX}language>"
    PROP_LIC = f"<{EX}licence>"
    PROP_APC = f"<{EX}apc>"
    PROP_SEAL = f"<{EX}seal>"
    PROP_HAS_CAT = f"<{EX}hasCategory>"

    def _sparql(self) -> _SPARQL:
        return _SPARQL(self.getDbPathOrUrl())

    def _stable_id(self, row: Dict[str,str]) -> str:
        eissn = _first_non_empty(row.get("Journal EISSN (online version)"),
                                 row.get("EISSN"), row.get("eissn"))
        issn = _first_non_empty(row.get("Journal ISSN (print version)"),
                                row.get("ISSN"), row.get("issn"))
        title = _first_non_empty(row.get("Journal title"), row.get("Title"), row.get("title"))
        return eissn or issn or title

    def _parse_categories(self, row: Dict[str,str]) -> List[str]:
        # Try several typical column names
        raw = _first_non_empty(row.get("Subjects"), row.get("Categories"), row.get("categories"))
        cats = _split_multi(raw)
        return cats

    def pushDataToDb(self, path: str) -> bool:
        endpoint = self.getDbPathOrUrl()
        if not endpoint:
            raise ValueError("SPARQL endpoint not set. Call setDbPathOrUrl first.")
        sparql = self._sparql()

        with open(path, "r", encoding="utf-8") as f:
            # autodetect delimiter
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)

            triples: List[str] = []

            for row in reader:
                sid = self._stable_id(row)
                if not sid:
                    continue
                jiri = _iri("journal", sid)
                title = _first_non_empty(row.get("Journal title"), row.get("Title"), row.get("title"))
                publisher = _first_non_empty(row.get("Publisher"), row.get("publisher"))
                languages = _split_multi(_first_non_empty(
                    row.get("Languages"), row.get("Language"), row.get("LanguagesLanguages in which the journal accepts manuscripts")
                ))
                licence = _first_non_empty(row.get("License"), row.get("Licence"), row.get("license"))
                apc = _bool(_first_non_empty(row.get("APC"), row.get("Has APC?"), row.get("apc")))
                seal = _bool(_first_non_empty(row.get("DOAJ Seal"), row.get("Seal"), row.get("doaj_seal")))

                triples.append(f"{jiri} a {self.JOURNAL_CLASS} .")
                triples.append(f"{jiri} {self.PROP_ID} {_lit(sid)} .")
                triples.append(f"{jiri} {self.PROP_TITLE} {_lit(title)} .")
                if publisher:
                    triples.append(f"{jiri} {self.PROP_PUBLISHER} {_lit(publisher)} .")
                if licence:
                    triples.append(f"{jiri} {self.PROP_LIC} {_lit(licence)} .")
                triples.append(f"{jiri} {self.PROP_APC} {'true' if apc else 'false'} .")
                triples.append(f"{jiri} {self.PROP_SEAL} {'true' if seal else 'false'} .")
                for lang in languages:
                    triples.append(f"{jiri} {self.PROP_LANG} {_lit(lang)} .")

                # Categories (create node by name and link)
                for c in self._parse_categories(row):
                    ciri = _iri("category", c)
                    triples.append(f"{ciri} a {self.CATEGORY_CLASS} .")
                    triples.append(f"{ciri} {self.PROP_ID} {_lit(c)} .")
                    triples.append(f"{jiri} {self.PROP_HAS_CAT} {ciri} .")

                # flush by chunks to avoid giant requests
                if len(triples) > 800:
                    sparql.update("INSERT DATA { " + "\n".join(triples) + " }")
                    triples = []

            if triples:
                sparql.update("INSERT DATA { " + "\n".join(triples) + " }")
        return True


class JournalQueryHandler(QueryHandler):
    JOURNAL_CLASS = f"<{EX}Journal>"
    CATEGORY_CLASS = f"<{EX}Category>"
    PROP_ID = f"<{EX}id>"
    PROP_TITLE = f"<{EX}title>"
    PROP_PUBLISHER = f"<{EX}publisher>"
    PROP_LANG = f"<{EX}language>"
    PROP_LIC = f"<{EX}licence>"
    PROP_APC = f"<{EX}apc>"
    PROP_SEAL = f"<{EX}seal>"
    PROP_HAS_CAT = f"<{EX}hasCategory>"

    def _sparql(self) -> _SPARQL:
        return _SPARQL(self.getDbPathOrUrl())

    def _rows_to_df(self, rows: List[Dict[str,dict]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame([])
        df = pd.DataFrame([{k: (v.get("value") if isinstance(v, dict) else v)
                            for k, v in r.items()} for r in rows])
        return df

    def getById(self, idd: str) -> pd.DataFrame:
        # Try journal / category / area in the graph; journals + categories exist here.
        Q = f"""
        PREFIX ex: <{EX}>
        SELECT ?kind ?id ?title ?publisher ?licence ?apc ?seal ?language ?category
        WHERE {{
          {{
            ?s a ex:Journal ;
               ex:id { _lit(idd) } .
            BIND("Journal" AS ?kind)
            OPTIONAL {{ ?s ex:id ?id }}
            OPTIONAL {{ ?s ex:title ?title }}
            OPTIONAL {{ ?s ex:publisher ?publisher }}
            OPTIONAL {{ ?s ex:licence ?licence }}
            OPTIONAL {{ ?s ex:apc ?apc }}
            OPTIONAL {{ ?s ex:seal ?seal }}
            OPTIONAL {{ ?s ex:language ?language }}
            OPTIONAL {{ ?s ex:hasCategory ?c . ?c ex:id ?category }}
          }}
          UNION
          {{
            ?s a ex:Category ;
               ex:id { _lit(idd) } .
            BIND("Category" AS ?kind)
            OPTIONAL {{ ?s ex:id ?id }}
          }}
        }}
        """
        rows = self._sparql().select(Q)
        return self._rows_to_df(rows)

    # ------- Journals-only helpers
    def _journals_df(self, extra_filter: str = "", params: Dict[str,str] = None) -> pd.DataFrame:
        Q = f"""
        PREFIX ex: <{EX}>
        SELECT ?id ?title ?publisher ?licence ?apc ?seal ?language ?category
        WHERE {{
          ?s a ex:Journal .
          OPTIONAL {{ ?s ex:id ?id }}
          OPTIONAL {{ ?s ex:title ?title }}
          OPTIONAL {{ ?s ex:publisher ?publisher }}
          OPTIONAL {{ ?s ex:licence ?licence }}
          OPTIONAL {{ ?s ex:apc ?apc }}
          OPTIONAL {{ ?s ex:seal ?seal }}
          OPTIONAL {{ ?s ex:language ?language }}
          OPTIONAL {{ ?s ex:hasCategory ?c . ?c ex:id ?category }}
          {extra_filter}
        }}
        """
        rows = self._sparql().select(Q)
        df = self._rows_to_df(rows)
        if df.empty:
            return df
        # aggregate languages/categories per journal id
        agg = (df.groupby(["id","title","publisher","licence","apc","seal"])
                 .agg({"language":lambda s: sorted(set([x for x in s.dropna().tolist()])),
                       "category":lambda s: sorted(set([x for x in s.dropna().tolist()]))})
                 .reset_index())
        # ensure list columns exist even if empty
        if "language" not in agg.columns:
            agg["language"] = [[]]*len(agg)
        if "category" not in agg.columns:
            agg["category"] = [[]]*len(agg)
        return agg

    def getAllJournals(self) -> pd.DataFrame:
        return self._journals_df()

    def getJournalsWithTitle(self, partialTitle: str) -> pd.DataFrame:
        flt = f'FILTER(CONTAINS(LCASE(?title), { _lit(partialTitle.lower()) }))'
        return self._journals_df(flt)

    def getJournalsPublishedBy(self, partialName: str) -> pd.DataFrame:
        flt = f'FILTER(CONTAINS(LCASE(?publisher), { _lit(partialName.lower()) }))'
        return self._journals_df(flt)

    def getJournalsWithLicense(self, licence: str) -> pd.DataFrame:
        flt = f'FILTER(?licence = { _lit(licence) })'
        return self._journals_df(flt)

    def getJournalsWithAPC(self) -> pd.DataFrame:
        flt = 'FILTER(?apc = true)'
        return self._journals_df(flt)

    def getJournalsWithDOAJSeal(self) -> pd.DataFrame:
        flt = 'FILTER(?seal = true)'
        return self._journals_df(flt)

# ============================================
# Engines
# ============================================

class BasicQueryEngine:
    def __init__(self) -> None:
        self.journalQuery: List[JournalQueryHandler] = []
        self.categoryQuery: List[CategoryQueryHandler] = []

    # ---- manage handlers
    def cleanJournalHandlers(self) -> bool:
        self.journalQuery = []
        return True
    def cleanCategoryHandlers(self) -> bool:
        self.categoryQuery = []
        return True
    def addJournalHandler(self, handler: JournalQueryHandler) -> bool:
        self.journalQuery.append(handler)
        return True
    def addCategoryHandler(self, handler: CategoryQueryHandler) -> bool:
        self.categoryQuery.append(handler)
        return True

    # ---- helpers: build model objects
    def _df_to_journals(self, df: pd.DataFrame) -> List[Journal]:
        if df is None or df.empty:
            return []
        out: List[Journal] = []
        for _, r in df.iterrows():
            cats = [Category(c) for c in (r.get("category") or [])]
            j = Journal(
                ids=[r["id"]],
                title=str(r.get("title") or ""),
                languages=list(r.get("language") or []),
                publisher=(None if pd.isna(r.get("publisher")) else str(r.get("publisher"))),
                seal=_bool(r.get("seal")),
                licence=str(r.get("licence") or ""),
                apc=_bool(r.get("apc")),
                _categories=cats,
                _areas=[],
            )
            out.append(j)
        return out

    # ---- entity by id (journal via graph; area/category via relational)
    def getEntityById(self, idd: str) -> Optional[IdentifiableEntity]:
        # try journals
        for jq in self.journalQuery:
            df = jq.getById(idd)
            if not df.empty and (df["kind"] == "Journal").any():
                # collapse to single journal row
                jdf = (df[df["kind"]=="Journal"]
                       .groupby(["id","title","publisher","licence","apc","seal"])
                       .agg({"language":lambda s: sorted(set([x for x in s.dropna().tolist()])),
                             "category":lambda s: sorted(set([x for x in s.dropna().tolist()]))})
                       .reset_index())
                js = self._df_to_journals(jdf)
                return js[0] if js else None
        # try relational
        for cq in self.categoryQuery:
            df = cq.getById(idd)
            if not df.empty:
                row = df.iloc[0].to_dict()
                if row.get("kind") == "Category":
                    return Category(row["id"], row.get("quartile"))
                if row.get("kind") == "Area":
                    return Area(row["id"])
        return None

    # ---- journal getters
    def getAllJournals(self) -> List[Journal]:
        frames = []
        for jq in self.journalQuery:
            frames.append(jq.getAllJournals())
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_journals(df)

    def getJournalsWithTitle(self, partialTitle: str) -> List[Journal]:
        frames = [jq.getJournalsWithTitle(partialTitle) for jq in self.journalQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_journals(df)

    def getJournalsPublishedBy(self, partialName: str) -> List[Journal]:
        frames = [jq.getJournalsPublishedBy(partialName) for jq in self.journalQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_journals(df)

    def getJournalsWithLicense(self, licence: str) -> List[Journal]:
        frames = [jq.getJournalsWithLicense(licence) for jq in self.journalQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_journals(df)

    def getJournalsWithAPC(self) -> List[Journal]:
        frames = [jq.getJournalsWithAPC() for jq in self.journalQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_journals(df)

    def getJournalsWithDOAJSeal(self) -> List[Journal]:
        frames = [jq.getJournalsWithDOAJSeal() for jq in self.journalQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_journals(df)

    # ---- categories / areas (relational)
    def _df_to_categories(self, df: pd.DataFrame) -> List[Category]:
        if df is None or df.empty:
            return []
        out = []
        for _, r in df.iterrows():
            out.append(Category(str(r["id"]), (None if pd.isna(r.get("quartile")) else str(r.get("quartile")))))
        return out
    def _df_to_areas(self, df: pd.DataFrame) -> List[Area]:
        if df is None or df.empty:
            return []
        return [Area(str(x)) for x in df["id"].tolist()]

    def getAllCategories(self) -> List[Category]:
        frames = [cq.getAllCategories() for cq in self.categoryQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_categories(df)

    def getAllAreas(self) -> List[Area]:
        frames = [cq.getAllAreas() for cq in self.categoryQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_areas(df)

    def getCategoriesWithQuartile(self, quartiles: Set[str]) -> List[Category]:
        frames = [cq.getCategoriesWithQuartile(quartiles) for cq in self.categoryQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_categories(df)

    def getCategoriesAssignedToAreas(self, area_ids: Set[str]) -> List[Category]:
        frames = [cq.getCategoriesAssignedToAreas(area_ids) for cq in self.categoryQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_categories(df)

    def getAreasAssignedToCategories(self, category_ids: Set[str]) -> List[Area]:
        frames = [cq.getAreasAssignedToCategories(category_ids) for cq in self.categoryQuery]
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_areas(df)

# --------------------------
# Full Query Engine (mashup)
# --------------------------

class FullQueryEngine(BasicQueryEngine):
    # Journals that have (any of) categories with given quartiles
    def getJournalsInCategoriesWithQuartile(
        self,
        category_ids: Set[str],
        quartiles: Set[str]
    ) -> List[Journal]:
        # 1) find categories allowed by quartile
        cats_q = set([c.getIds()[0] for c in self.getCategoriesWithQuartile(quartiles)])
        # if category filter provided, intersect
        if category_ids:
            cats_ok = sorted(cats_q.intersection(category_ids))
        else:
            cats_ok = sorted(cats_q)

        if not cats_ok:
            return []

        # 2) from graph, collect journals having any of these categories
        frames = []
        for jq in self.journalQuery:
            flt = "FILTER(?category IN (" + ",".join(_lit(c) for c in cats_ok) + "))"
            frames.append(jq._journals_df(flt))
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_journals(df)

    # Journals with a licence and in areas
    def getJournalsInAreasWithLicense(
        self,
        areas_ids: Set[str],
        licenses: Set[str]
    ) -> List[Journal]:
        # 1) categories that belong to target areas
        cats = set([c.getIds()[0] for c in self.getCategoriesAssignedToAreas(areas_ids)])
        if not cats:  # areas empty => all areas, cats already expanded
            return []

        # 2) from graph, journals with those categories
        frames = []
        cat_filter = "FILTER(?category IN (" + ",".join(_lit(c) for c in sorted(cats)) + "))"
        if licenses:
            lic_filter = "FILTER(?licence IN (" + ",".join(_lit(l) for l in sorted(licenses)) + "))"
        else:
            lic_filter = ""
        extra = "\n".join([cat_filter, lic_filter])
        for jq in self.journalQuery:
            frames.append(jq._journals_df(extra))
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_journals(df)

    # Diamond journals (no APC) AND in given areas AND categories with quartiles
    def getDiamondJournalsInAreasAndCategoriesWithQuartile(
        self,
        areas_ids: Set[str],
        category_ids: Set[str],
        quartiles: Set[str]
    ) -> List[Journal]:
        # areas -> categories
        cats_in_areas = set([c.getIds()[0] for c in self.getCategoriesAssignedToAreas(areas_ids)])
        # quartile filter
        cats_by_q = set([c.getIds()[0] for c in self.getCategoriesWithQuartile(quartiles)])
        cats = cats_in_areas.intersection(cats_by_q)
        if category_ids:
            cats = cats.intersection(set(category_ids))
        if not cats:
            return []

        frames = []
        extra = "FILTER(?apc = false)\n" + "FILTER(?category IN (" + ",".join(_lit(c) for c in sorted(cats)) + "))"
        for jq in self.journalQuery:
            frames.append(jq._journals_df(extra))
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        return self._df_to_journals(df)
