"""Low-level repositories encapsulating database access."""

from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import uuid
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalise_identifier(value: Optional[str]) -> str:
    if value is None:
        return ""
    return re.sub(r"[^0-9a-z]+", "", value.strip().lower())


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# SQLite repository for categories/areas
# ---------------------------------------------------------------------------

class SQLiteCategoryRepository:
    """Persist and query category/area information via SQLite."""

    def __init__(self, path: str):
        self.path = path

    # -- schema ----------------------------------------------------------------

    @staticmethod
    def _initialise_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS journal (
                id TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS journal_alias (
                alias TEXT PRIMARY KEY,
                journal_id TEXT NOT NULL,
                FOREIGN KEY (journal_id) REFERENCES journal(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS category (
                id TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS area (
                id TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS category_quartile (
                category_id TEXT NOT NULL,
                quartile TEXT NOT NULL,
                PRIMARY KEY (category_id, quartile),
                FOREIGN KEY (category_id) REFERENCES category(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS category_area (
                category_id TEXT NOT NULL,
                area_id TEXT NOT NULL,
                PRIMARY KEY (category_id, area_id),
                FOREIGN KEY (category_id) REFERENCES category(id) ON DELETE CASCADE,
                FOREIGN KEY (area_id) REFERENCES area(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS journal_category (
                journal_id TEXT NOT NULL,
                category_id TEXT NOT NULL,
                quartile TEXT,
                PRIMARY KEY (journal_id, category_id, quartile),
                FOREIGN KEY (journal_id) REFERENCES journal(id) ON DELETE CASCADE,
                FOREIGN KEY (category_id) REFERENCES category(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS journal_area (
                journal_id TEXT NOT NULL,
                area_id TEXT NOT NULL,
                PRIMARY KEY (journal_id, area_id),
                FOREIGN KEY (journal_id) REFERENCES journal(id) ON DELETE CASCADE,
                FOREIGN KEY (area_id) REFERENCES area(id) ON DELETE CASCADE
            );
            """
        )

    def _connect(self) -> sqlite3.Connection:
        _ensure_dir(self.path)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        self._initialise_schema(conn)
        return conn

    # -- loading ---------------------------------------------------------------

    def load_json(self, file_path: str) -> bool:
        if not self.path or not file_path or not os.path.isfile(file_path):
            return False

        with open(file_path, "r", encoding="utf-8") as handler:
            payload = json.load(handler)

        if not isinstance(payload, list):
            return False

        with self._connect() as conn:
            cursor = conn.cursor()
            for entry in payload:
                identifiers = [str(i).strip() for i in entry.get("identifiers", []) if str(i).strip()]
                if not identifiers:
                    identifiers = [str(uuid.uuid4())]
                canonical = identifiers[0]

                cursor.execute(
                    "INSERT OR IGNORE INTO journal(id) VALUES (?)",
                    (canonical,),
                )

                aliases = set(identifiers)
                aliases.update(alias.replace("-", "") for alias in identifiers if alias)

                for alias in aliases:
                    cursor.execute(
                        "INSERT OR IGNORE INTO journal_alias(alias, journal_id) VALUES (?, ?)",
                        (alias, canonical),
                    )

                categories = entry.get("categories", [])
                areas = [str(a).strip() for a in entry.get("areas", []) if str(a).strip()]

                for area_id in areas:
                    cursor.execute("INSERT OR IGNORE INTO area(id) VALUES (?)", (area_id,))

                for category_entry in categories:
                    category_id = str(category_entry.get("id", "")).strip()
                    quartile = str(category_entry.get("quartile", "")).strip()
                    if not category_id:
                        continue

                    cursor.execute("INSERT OR IGNORE INTO category(id) VALUES (?)", (category_id,))

                    if quartile:
                        cursor.execute(
                            "INSERT OR IGNORE INTO category_quartile(category_id, quartile) VALUES (?, ?)",
                            (category_id, quartile),
                        )

                    for area_id in areas:
                        cursor.execute(
                            "INSERT OR IGNORE INTO category_area(category_id, area_id) VALUES (?, ?)",
                            (category_id, area_id),
                        )

                    cursor.execute(
                        "INSERT OR IGNORE INTO journal_category(journal_id, category_id, quartile) VALUES (?, ?, ?)",
                        (canonical, category_id, quartile if quartile else None),
                    )

                for area_id in areas:
                    cursor.execute(
                        "INSERT OR IGNORE INTO journal_area(journal_id, area_id) VALUES (?, ?)",
                        (canonical, area_id),
                    )

            conn.commit()
        return True

    # -- helpers ---------------------------------------------------------------

    def _fetch_all_categories(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id AS category_id,
                    GROUP_CONCAT(DISTINCT cq.quartile) AS quartiles,
                    GROUP_CONCAT(DISTINCT ca.area_id) AS areas
                FROM category c
                LEFT JOIN category_quartile cq ON cq.category_id = c.id
                LEFT JOIN category_area ca ON ca.category_id = c.id
                GROUP BY c.id
                ORDER BY c.id
                """
            ).fetchall()

        data = []
        for row in rows:
            quartiles = sorted({item for item in (row["quartiles"] or "").split(",") if item})
            areas = sorted({item for item in (row["areas"] or "").split(",") if item})
            data.append({"id": row["category_id"], "quartiles": quartiles, "areas": areas})
        return pd.DataFrame(data)

    def _fetch_all_areas(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    a.id AS area_id,
                    GROUP_CONCAT(DISTINCT ca.category_id) AS categories
                FROM area a
                LEFT JOIN category_area ca ON ca.area_id = a.id
                GROUP BY a.id
                ORDER BY a.id
                """
            ).fetchall()

        data = []
        for row in rows:
            categories = sorted({item for item in (row["categories"] or "").split(",") if item})
            data.append({"id": row["area_id"], "categories": categories})
        return pd.DataFrame(data)

    # -- public querying -------------------------------------------------------

    def fetch_categories(self) -> pd.DataFrame:
        return self._fetch_all_categories()

    def fetch_areas(self) -> pd.DataFrame:
        return self._fetch_all_areas()

    def fetch_categories_with_quartiles(self, quartiles: Iterable[str]) -> pd.DataFrame:
        quartile_norm = {_normalise_identifier(q) for q in quartiles or [] if q}
        df = self._fetch_all_categories()
        if df.empty or not quartile_norm:
            return df
        mask = df["quartiles"].apply(
            lambda items: bool({_normalise_identifier(i) for i in items}.intersection(quartile_norm))
        )
        return df.loc[mask].reset_index(drop=True)

    def fetch_categories_assigned_to_areas(self, areas: Iterable[str]) -> pd.DataFrame:
        area_norm = {_normalise_identifier(a) for a in areas or [] if a}
        df = self._fetch_all_categories()
        if df.empty or not area_norm:
            return df
        mask = df["areas"].apply(
            lambda items: bool({_normalise_identifier(i) for i in items}.intersection(area_norm))
        )
        return df.loc[mask].reset_index(drop=True)

    def fetch_areas_assigned_to_categories(self, categories: Iterable[str]) -> pd.DataFrame:
        category_norm = {_normalise_identifier(c) for c in categories or [] if c}
        df = self._fetch_all_areas()
        if df.empty or not category_norm:
            return df
        mask = df["categories"].apply(
            lambda items: bool({_normalise_identifier(i) for i in items}.intersection(category_norm))
        )
        return df.loc[mask].reset_index(drop=True)

    def resolve_journal(self, identifier: str) -> Optional[str]:
        if not identifier:
            return None
        identifier_variants = [identifier, identifier.replace("-", "")]
        with self._connect() as conn:
            for variant in identifier_variants:
                row = conn.execute(
                    "SELECT journal_id FROM journal_alias WHERE alias = ? COLLATE NOCASE LIMIT 1",
                    (variant,),
                ).fetchone()
                if row:
                    return row["journal_id"]
        return None

    def fetch_journal_categories(self, journal_id: str) -> Dict[str, Set[str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT category_id, quartile
                FROM journal_category
                WHERE journal_id = ?
                """,
                (journal_id,),
            ).fetchall()

        result: Dict[str, Set[str]] = defaultdict(set)
        for row in rows:
            cid = row["category_id"]
            quartile = row["quartile"]
            if quartile:
                result[cid].add(quartile)
            else:
                result.setdefault(cid, set())
        return result

    def fetch_journal_areas(self, journal_id: str) -> Set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT area_id FROM journal_area WHERE journal_id = ?",
                (journal_id,),
            ).fetchall()
        return {row["area_id"] for row in rows}

    def fetch_entity_by_identifier(self, identifier: str) -> pd.DataFrame:
        if not identifier:
            return pd.DataFrame(
                columns=["type", "id", "quartiles", "areas", "categories"]
            )

        norm = _normalise_identifier(identifier)
        with self._connect() as conn:
            category_row = conn.execute(
                "SELECT id FROM category WHERE LOWER(REPLACE(id, '-', '')) = ? LIMIT 1",
                (norm,),
            ).fetchone()
            if category_row:
                all_categories = self._fetch_all_categories()
                match = all_categories.loc[all_categories["id"] == category_row["id"]]
                if not match.empty:
                    row = match.iloc[0]
                    return pd.DataFrame(
                        [
                            {
                                "type": "category",
                                "id": row["id"],
                                "quartiles": row["quartiles"],
                                "areas": row["areas"],
                            }
                        ]
                    )

            area_row = conn.execute(
                "SELECT id FROM area WHERE LOWER(REPLACE(id, '-', '')) = ? LIMIT 1",
                (norm,),
            ).fetchone()
            if area_row:
                all_areas = self._fetch_all_areas()
                match = all_areas.loc[all_areas["id"] == area_row["id"]]
                if not match.empty:
                    row = match.iloc[0]
                    return pd.DataFrame(
                        [
                            {
                                "type": "area",
                                "id": row["id"],
                                "categories": row["categories"],
                            }
                        ]
                    )

            journal_row = conn.execute(
                "SELECT journal_id FROM journal_alias WHERE LOWER(REPLACE(alias, '-', '')) = ? LIMIT 1",
                (norm,),
            ).fetchone()
            if journal_row:
                journal_id = journal_row["journal_id"]
                categories = self.fetch_journal_categories(journal_id)
                areas = sorted(self.fetch_journal_areas(journal_id))
                return pd.DataFrame(
                    [
                        {
                            "type": "journal",
                            "id": journal_id,
                            "categories": sorted(categories.keys()),
                            "areas": areas,
                        }
                    ]
                )

        return pd.DataFrame(
            columns=["type", "id", "quartiles", "areas", "categories"]
        )

    def export_all(self) -> Dict[str, dict]:
        categories: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: {"quartiles": set(), "areas": set()})
        areas: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: {"categories": set()})
        journal_categories: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        journal_areas: Dict[str, Set[str]] = defaultdict(set)
        alias_map: Dict[str, str] = {}

        with self._connect() as conn:
            for row in conn.execute("SELECT category_id, quartile FROM category_quartile"):
                if row["quartile"]:
                    categories[row["category_id"]]["quartiles"].add(row["quartile"])

            for row in conn.execute("SELECT category_id, area_id FROM category_area"):
                categories[row["category_id"]]["areas"].add(row["area_id"])
                areas[row["area_id"]]["categories"].add(row["category_id"])

            for row in conn.execute("SELECT journal_id, category_id, quartile FROM journal_category"):
                journal_categories[row["journal_id"]][row["category_id"]].update(
                    {row["quartile"]} if row["quartile"] else set()
                )

            for row in conn.execute("SELECT journal_id, area_id FROM journal_area"):
                journal_areas[row["journal_id"]].add(row["area_id"])

            for row in conn.execute("SELECT alias, journal_id FROM journal_alias"):
                alias_map[_normalise_identifier(row["alias"])] = row["journal_id"]

        export_categories = {
            cid: {
                "quartiles": set(data["quartiles"]),
                "areas": set(data["areas"]),
            }
            for cid, data in categories.items()
        }
        export_areas = {
            aid: {
                "categories": set(data["categories"]),
            }
            for aid, data in areas.items()
        }
        export_journal_categories = {
            jid: {cid: set(quartiles) for cid, quartiles in data.items()}
            for jid, data in journal_categories.items()
        }
        export_journal_areas = {
            jid: set(area_set)
            for jid, area_set in journal_areas.items()
        }

        return {
            "categories": export_categories,
            "areas": export_areas,
            "journal_categories": export_journal_categories,
            "journal_areas": export_journal_areas,
            "journal_alias": dict(alias_map),
        }


# ---------------------------------------------------------------------------
# SPARQL repository for journals (with offline fallback)
# ---------------------------------------------------------------------------

class _InMemoryJournalStore:
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
        self._frame = pd.DataFrame(columns=self.COLUMNS)
        self._index: Dict[str, str] = {}

    def add_records(self, records: List[Dict[str, object]]) -> None:
        if not records:
            return
        new_df = pd.DataFrame.from_records(records)
        if self._frame.empty:
            self._frame = new_df
        else:
            combined = pd.concat([self._frame, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset="id", keep="last")
            self._frame = combined.reset_index(drop=True)
        for rec in records:
            for identifier in rec.get("identifiers", ()):
                normalised = _normalise_identifier(identifier)
                if normalised:
                    self._index[normalised] = rec["id"]
                    self._index[_normalise_identifier(identifier.replace("-", ""))] = rec["id"]

    def all(self) -> pd.DataFrame:
        return self._frame.copy()

    def by_identifier(self, identifier: str) -> pd.DataFrame:
        if not identifier:
            return pd.DataFrame(columns=self.COLUMNS)
        target = self._index.get(_normalise_identifier(identifier))
        if not target:
            target = self._index.get(_normalise_identifier(identifier.replace("-", "")))
        if target:
            frame = self._frame.loc[self._frame["id"] == target]
            if not frame.empty:
                return frame.reset_index(drop=True)
        mask = self._frame["identifiers"].apply(
            lambda ids: any(_normalise_identifier(identifier) == _normalise_identifier(i) for i in ids)
        )
        result = self._frame.loc[mask]
        return result.reset_index(drop=True) if not result.empty else pd.DataFrame(columns=self.COLUMNS)

    def by_title(self, title_part: str) -> pd.DataFrame:
        frame = self.all()
        if not title_part:
            return frame
        mask = frame["title"].str.contains(str(title_part), case=False, na=False)
        return frame.loc[mask].reset_index(drop=True)

    def by_publisher(self, publisher_part: str) -> pd.DataFrame:
        frame = self.all()
        if not publisher_part:
            return frame
        mask = frame["publisher"].str.contains(str(publisher_part), case=False, na=False)
        return frame.loc[mask].reset_index(drop=True)

    def by_license(self, licenses: Iterable[str]) -> pd.DataFrame:
        frame = self.all()
        license_norm = {_normalise_identifier(l) for l in licenses or [] if l}
        if not license_norm:
            return frame
        mask = frame["license"].apply(
            lambda value: _normalise_identifier(value) in license_norm if isinstance(value, str) else False
        )
        return frame.loc[mask].reset_index(drop=True)

    def with_apc(self) -> pd.DataFrame:
        frame = self.all()
        return frame.loc[frame["apc"] == True].reset_index(drop=True)  # noqa: E712

    def with_doaj_seal(self) -> pd.DataFrame:
        frame = self.all()
        return frame.loc[frame["doaj_seal"] == True].reset_index(drop=True)  # noqa: E712


class SparqlJournalRepository:
    """Interact with a SPARQL endpoint; falls back to an in-memory store if unreachable."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._store = _JOURNAL_STORES.setdefault(endpoint, _InMemoryJournalStore())

    # -- loading ---------------------------------------------------------------

    def load_csv(self, file_path: str) -> bool:
        if not file_path or not os.path.isfile(file_path):
            return False
        df = pd.read_csv(file_path, dtype=str).fillna("")
        records: List[Dict[str, object]] = []
        for _, row in df.iterrows():
            title = row.get("Journal title", "").strip()
            issn_print = row.get("Journal ISSN (print version)", "").strip()
            issn_e = row.get("Journal EISSN (online version)", "").strip()
            languages_raw = row.get("Languages in which the journal accepts manuscripts", "")
            languages = tuple(
                lang.strip()
                for lang in str(languages_raw).split(",")
                if lang and lang.strip()
            )
            publisher = row.get("Publisher", "").strip()
            doaj_seal = str(row.get("DOAJ Seal", "")).strip().lower() == "yes"
            license_value = row.get("Journal license", "").strip()
            apc_value = str(row.get("APC", "")).strip().lower() == "yes"

            primary_id = (issn_e or issn_print or title or str(uuid.uuid4())).strip()
            identifiers = {primary_id}
            if issn_print:
                identifiers.add(issn_print)
            if issn_e:
                identifiers.add(issn_e)
            if title:
                identifiers.add(title)

            records.append(
                {
                    "id": primary_id,
                    "title": title,
                    "print_issn": issn_print,
                    "electronic_issn": issn_e,
                    "languages": tuple(sorted({lang for lang in languages if lang})),
                    "publisher": publisher,
                    "doaj_seal": doaj_seal,
                    "license": license_value,
                    "apc": apc_value,
                    "identifiers": tuple(sorted({identifier for identifier in identifiers if identifier})),
                }
            )

        self._store.add_records(records)

        if self.endpoint.startswith("http") and self._is_endpoint_available():
            try:
                self._push_records(records)
            except requests.RequestException:
                # Fall back silently – the in-memory representation keeps the data
                pass

        return True

    def _escape_literal(self, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _push_records(self, records: Sequence[Dict[str, object]], batch_size: int = 20) -> None:
        if not records:
            return

        base_uri = "http://example.org/journal/"

        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            statements = []
            for rec in batch:
                subject = f"<{base_uri}{rec['id']}>"
                statements.append(f"{subject} a <http://example.org/schema/Journal> ;")
                if rec["title"]:
                    statements.append(f'    <http://purl.org/dc/terms/title> {self._escape_literal(rec["title"])} ;')
                if rec["publisher"]:
                    statements.append(
                        f'    <http://purl.org/dc/terms/publisher> {self._escape_literal(rec["publisher"])} ;'
                    )
                if rec["license"]:
                    statements.append(
                        f'    <http://purl.org/dc/terms/license> {self._escape_literal(rec["license"])} ;'
                    )
                statements.append(
                    f'    <http://example.org/schema/hasAPC> {"true" if rec["apc"] else "false"} ;'
                )
                statements.append(
                    f'    <http://example.org/schema/hasDOAJSeal> {"true" if rec["doaj_seal"] else "false"} ;'
                )
                for identifier in rec["identifiers"]:
                    statements.append(
                        f'    <http://purl.org/dc/terms/identifier> {self._escape_literal(identifier)} ;'
                    )
                statements[-1] = statements[-1].rstrip(" ;")
                statements.append(".")

            update = "INSERT DATA { " + " ".join(statements) + " }"
            requests.post(
                self.endpoint,
                data=update.encode("utf-8"),
                headers={"Content-Type": "application/sparql-update"},
                timeout=10,
            )

    def _is_endpoint_available(self) -> bool:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((parsed.hostname, port), timeout=1):
                return True
        except OSError:
            return False

    # -- querying --------------------------------------------------------------

    def fetch_all(self) -> pd.DataFrame:
        return self._store.all()

    def fetch_by_id(self, identifier: str) -> pd.DataFrame:
        return self._store.by_identifier(identifier)

    def fetch_by_title(self, title_part: str) -> pd.DataFrame:
        return self._store.by_title(title_part)

    def fetch_by_publisher(self, publisher_part: str) -> pd.DataFrame:
        return self._store.by_publisher(publisher_part)

    def fetch_by_license(self, licenses: Iterable[str]) -> pd.DataFrame:
        return self._store.by_license(licenses)

    def fetch_with_apc(self) -> pd.DataFrame:
        return self._store.with_apc()

    def fetch_with_doaj_seal(self) -> pd.DataFrame:
        return self._store.with_doaj_seal()


_JOURNAL_STORES: Dict[str, _InMemoryJournalStore] = {}
