import csv
import hashlib
import re
import requests
from typing import List

# ===== Namespaces =====
EX = "http://example.org/"               
DCT = "http://purl.org/dc/terms/"
SCHEMA = "https://schema.org/"
XSD = "http://www.w3.org/2001/XMLSchema#"

# ===== Small helpers =====
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
    # languages are separated by ", " (comma + space)
    s = s or ""
    parts = [p.strip() for p in s.split(", ")] if s else []
    return [p for p in parts if p]



# ===== Data model classes (UML layer) =====
class IdentifiableEntity:
    def __init__(self, id_: str):
        self.id = id_

    def getIds(self):
        return [self.id]


class Journal(IdentifiableEntity):
    def __init__(self, id_: str, title: str, languages: list[str], 
                 publisher: str, seal: bool, licence: str, apc: bool):
        super().__init__(id_)
        self.title = title
        self.languages = languages
        self.publisher = publisher
        self.seal = seal
        self.licence = licence
        self.apc = apc
        self.categories = []   # list[Category]
        self.areas = []        # list[Area]

    # UML-required methods
    def getTitle(self): return self.title
    def getLanguages(self): return self.languages
    def getPublisher(self): return self.publisher or None
    def hasDOAJSeal(self): return self.seal
    def getLicence(self): return self.licence
    def hasAPC(self): return self.apc
    def getCategories(self): return self.categories
    def getAreas(self): return self.areas



# ===== Handler classes =====
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

        # Read CSV robustly (handles quoted commas in titles)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
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
                    chunks.append(f"  {s} dct:language \"{_escape(lang)}\" .")

            chunks.append("}")
            payload = "\n".join(chunks)

            resp = requests.post(
                endpoint,
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/sparql-update"},
                timeout=60,
            )
            if not resp.ok:
                raise RuntimeError(f"Blazegraph insert failed [{resp.status_code}]: {resp.text[:400]}")

        for r in rows:
            batch.append(r)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        flush(batch)
        return True


class QueryHandler(Handler):
    def getById(self, id_: str):
        raise NotImplementedError

class JournalQueryHandler(QueryHandler):
    pass  

