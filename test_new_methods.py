"""
Tests for the three new methods added to improve the project mark:

  1. CategoryQueryHandler.getCategoriesByName(cat_partial_name)
  2. JournalQueryHandler.getJournalsWithNoPublisher()
  3. FullQueryEngine.getJournalsWithNoPublisherInCategories(cat_partial_name)

HOW TO RUN:
  1. Make sure Blazegraph is running:
         java -server -Xmx1g -jar blazegraph.jar

  2. Put this file in the same folder as impl.py and test.py

  3. Run:
         python -m unittest test_new_methods

  4. All tests should print OK at the end.

IMPORTANT: adjust the paths and endpoint below (lines 35-38) to match your setup,
just like you did in the original test.py.
"""

import unittest
import pandas as pd

# ── Import your classes (same pattern as original test.py) ──────────────────
from impl import (
    CategoryUploadHandler,
    CategoryQueryHandler,
    JournalUploadHandler,
    JournalQueryHandler,
    FullQueryEngine,
    Journal,
    Category,
)

# ── Adjust these paths to match your local setup ────────────────────────────
CAT_DATA   = "data/scimago.json"          # path to the Scimago JSON file
JOU_DATA   = "data/doaj.csv"              # path to the DOAJ CSV file
REL_PATH   = "relational.db"             # path to your SQLite database
GRP_ENDP   = "http://127.0.0.1:9999/blazegraph/sparql"  # Blazegraph SPARQL endpoint
# ────────────────────────────────────────────────────────────────────────────


class TestNewMethods(unittest.TestCase):
    """
    We set up the databases ONCE for all tests (setUpClass), just like the
    original test.py does, to avoid re-uploading 322 000 triples every test.
    """

    @classmethod
    def setUpClass(cls):
        """
        Upload data and create query handlers once before any test runs.
        This mirrors the setup in the original test.py.
        """
        # -- Relational database (Scimago categories / areas) -----------------
        cat_upload = CategoryUploadHandler()
        cat_upload.setDbPathOrUrl(REL_PATH)
        cat_upload.pushDataToDb(CAT_DATA)

        # -- Graph database (DOAJ journals) -----------------------------------
        jou_upload = JournalUploadHandler()
        jou_upload.setDbPathOrUrl(GRP_ENDP)
        jou_upload.pushDataToDb(JOU_DATA)

        # -- Query handlers ---------------------------------------------------
        cls.cat_qh = CategoryQueryHandler()
        cls.cat_qh.setDbPathOrUrl(REL_PATH)

        cls.jou_qh = JournalQueryHandler()
        cls.jou_qh.setDbPathOrUrl(GRP_ENDP)

        # -- Full query engine ------------------------------------------------
        cls.fq = FullQueryEngine()
        cls.fq.addCategoryHandler(cls.cat_qh)
        cls.fq.addJournalHandler(cls.jou_qh)


    # ════════════════════════════════════════════════════════════════════════
    # 1. CategoryQueryHandler.getCategoriesByName
    # ════════════════════════════════════════════════════════════════════════

    def test_01_getCategoriesByName_returns_dataframe(self):
        """The method must always return a pandas DataFrame, never None or a list."""
        result = self.cat_qh.getCategoriesByName("Medicine")
        self.assertIsInstance(result, pd.DataFrame,
            "getCategoriesByName must return a DataFrame")

    def test_02_getCategoriesByName_partial_match(self):
        """Partial name search must find categories containing the substring."""
        result = self.cat_qh.getCategoriesByName("intel")
        # "Artificial Intelligence" contains "intel" → must appear in results
        self.assertFalse(result.empty,
            "getCategoriesByName('intel') should find at least one category "
            "(e.g. 'Artificial Intelligence')")
        # Check the id column contains something with "intel" (case-insensitive)
        ids_lower = result["id"].astype(str).str.lower()
        self.assertTrue(ids_lower.str.contains("intel").any(),
            "At least one returned category should contain 'intel' in its name")

    def test_03_getCategoriesByName_case_insensitive(self):
        """Search must be case-insensitive: 'MEDICINE' == 'medicine' == 'Medicine'."""
        lower_result  = self.cat_qh.getCategoriesByName("medicine")
        upper_result  = self.cat_qh.getCategoriesByName("MEDICINE")
        mixed_result  = self.cat_qh.getCategoriesByName("Medicine")
        self.assertEqual(
            len(lower_result), len(upper_result),
            "getCategoriesByName must be case-insensitive (lower vs upper)"
        )
        self.assertEqual(
            len(lower_result), len(mixed_result),
            "getCategoriesByName must be case-insensitive (lower vs mixed)"
        )

    def test_04_getCategoriesByName_no_duplicates(self):
        """Results must have no duplicate category ids."""
        result = self.cat_qh.getCategoriesByName("a")   # broad search → many results
        if not result.empty and "id" in result.columns:
            self.assertEqual(
                len(result["id"]), len(result["id"].unique()),
                "getCategoriesByName must return no duplicate category ids"
            )

    def test_05_getCategoriesByName_nonexistent_returns_empty(self):
        """Searching for a name that doesn't exist must return an empty DataFrame."""
        result = self.cat_qh.getCategoriesByName("zzz_nonexistent_category_xyz")
        self.assertIsInstance(result, pd.DataFrame,
            "Must return a DataFrame even when nothing matches")
        self.assertTrue(result.empty,
            "Should return an empty DataFrame for a name that doesn't exist")


    # ════════════════════════════════════════════════════════════════════════
    # 2. JournalQueryHandler.getJournalsWithNoPublisher
    # ════════════════════════════════════════════════════════════════════════

    def test_06_getJournalsWithNoPublisher_returns_dataframe(self):
        """The method must always return a pandas DataFrame."""
        result = self.jou_qh.getJournalsWithNoPublisher()
        self.assertIsInstance(result, pd.DataFrame,
            "getJournalsWithNoPublisher must return a DataFrame")

    def test_07_getJournalsWithNoPublisher_no_publisher_column(self):
        """Every row in the result must have an empty/missing publisher."""
        result = self.jou_qh.getJournalsWithNoPublisher()
        if result.empty:
            return  # valid: there may genuinely be no such journals in the dataset
        if "publisher" in result.columns:
            # Each value must be NaN, empty string, 'none', or 'nan'
            for val in result["publisher"]:
                self.assertTrue(
                    pd.isna(val) or str(val).strip().lower() in ("", "none", "nan"),
                    f"Row with publisher='{val}' should not appear in "
                    "getJournalsWithNoPublisher results"
                )

    def test_08_getJournalsWithNoPublisher_not_in_publishedBy(self):
        """
        Journals returned by getJournalsWithNoPublisher must NOT appear in
        getJournalsPublishedBy results (when publisher names match).
        Cross-check: the two methods must be consistent with each other.
        """
        no_pub   = self.jou_qh.getJournalsWithNoPublisher()
        has_pub  = self.jou_qh.getAllJournals()
        if no_pub.empty or has_pub.empty:
            return
        if "publisher" not in has_pub.columns:
            return
        # Journals that DO have a publisher
        pub_ids = set(
            has_pub.loc[
                ~(has_pub["publisher"].isna() |
                  has_pub["publisher"].astype(str).str.strip().str.lower().isin(["", "none", "nan"])),
                "id"
            ].astype(str)
        )
        no_pub_ids = set(no_pub["id"].astype(str))
        overlap = pub_ids & no_pub_ids
        self.assertEqual(len(overlap), 0,
            f"These journal ids appear in both 'has publisher' and 'no publisher' sets: {overlap}")


    # ════════════════════════════════════════════════════════════════════════
    # 3. FullQueryEngine.getJournalsWithNoPublisherInCategories
    # ════════════════════════════════════════════════════════════════════════

    def test_09_getJournalsWithNoPublisherInCategories_returns_list(self):
        """The method must return a list (not a DataFrame, not None)."""
        result = self.fq.getJournalsWithNoPublisherInCategories("Medicine")
        self.assertIsInstance(result, list,
            "getJournalsWithNoPublisherInCategories must return a list")

    def test_10_getJournalsWithNoPublisherInCategories_items_are_journals(self):
        """Every item in the returned list must be a Journal object."""
        result = self.fq.getJournalsWithNoPublisherInCategories("Medicine")
        for item in result:
            self.assertIsInstance(item, Journal,
                f"Expected Journal object, got {type(item)}")

    def test_11_getJournalsWithNoPublisherInCategories_no_publisher(self):
        """
        Every Journal in the result must have no publisher
        (getPublisher returns None, empty string, or 'None').
        """
        result = self.fq.getJournalsWithNoPublisherInCategories("Medicine")
        for journal in result:
            pub = journal.getPublisher()
            self.assertTrue(
                pub is None or str(pub).strip().lower() in ("", "none", "nan"),
                f"Journal '{journal.getTitle()}' has publisher='{pub}' "
                "but should have no publisher"
            )

    def test_12_getJournalsWithNoPublisherInCategories_nonexistent_returns_empty(self):
        """Searching with a category name that doesn't exist must return an empty list."""
        result = self.fq.getJournalsWithNoPublisherInCategories("zzz_nonexistent_xyz")
        self.assertIsInstance(result, list,
            "Must return a list even when nothing matches")
        self.assertEqual(len(result), 0,
            "Should return empty list for a category name that doesn't exist")

    def test_13_getJournalsWithNoPublisherInCategories_subset_of_no_publisher(self):
        """
        Results must be a subset of all no-publisher journals.
        A journal can only appear here if it also has no publisher overall.
        """
        # getJournalsWithNoPublisher lives on JournalQueryHandler, not FullQueryEngine.
        # So we call it on jou_qh and collect all no-publisher ISSNs into a flat set.
        all_no_pub_df = self.jou_qh.getJournalsWithNoPublisher()
        if all_no_pub_df.empty or 'id' not in all_no_pub_df.columns:
            return  # nothing to check against
        # id may be comma-separated ISSNs - split and flatten into one big set
        all_no_pub_ids = set()
        for cell in all_no_pub_df['id'].astype(str):
            for part in cell.split(','):
                all_no_pub_ids.add(part.strip().lower())

        result = self.fq.getJournalsWithNoPublisherInCategories("a")  # broad search
        for journal in result:
            ids = journal.getIds()
            if ids:
                # Normalise to lowercase for comparison (ISSNs are case-insensitive)
                journal_id = ids[0].strip().lower()
                self.assertIn(journal_id, all_no_pub_ids,
                    f"Journal {ids[0]} is in category results but not in "
                    "overall no-publisher set")


if __name__ == "__main__":
    unittest.main(verbosity=2)
