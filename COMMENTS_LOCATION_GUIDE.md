# Code Comments Location Guide

## Quick Navigation

Use Ctrl+F (Cmd+F) to find these sections in `impl.py`:

### 1. JOURNAL CLASS DOCUMENTATION
**Search for:** `class Journal(IdentifiableEntity):`
- **What:** Complete class-level documentation explaining the Journal data model
- **Explains:** How journals store DOAJ data, relationships to categories/areas, identifier strategy
- **Why:** Professor may ask about journal representation in semantic systems

### 2. JOURNAL PROPERTY ACCESSORS
**Search for:** `# --- Basic field accessors (DOAJ journal metadata) ---`
- **What:** Detailed comments for each getter method
- **Explains:** 
  - `getTitle()` - What title means
  - `getPublisher()` - Publisher information source
  - `getLicense()` - Common license types (CC BY, CC BY-NC, etc.)
  - `getAPC()` - Article Processing Charge meaning and implications
  - `getDOAJSeal()` - DOAJ certification criteria
  - `getLanguages()` - Manuscript language support
- **Why:** These are query methods professor may ask about

### 3. JOURNAL RELATIONSHIP ACCESSORS
**Search for:** `# --- Knowledge Graph relationship accessors ---`
- **What:** Explanations of category and area linking
- **Explains:**
  - `addCategory()` / `getCategories()` - Many-to-many journal-category relationships
  - `addArea()` / `getAreas()` - Many-to-many journal-area relationships
  - `getIds()` - Returns journal identifiers (new method added for fixes)
- **Why:** Shows how semantic relationships are represented

### 4. BUILD JOURNAL URI FUNCTION
**Search for:** `def _build_journal_uri(issn: str) -> URIRef:`
- **What:** Explanation of URI creation for RDF
- **Explains:**
  - Why URIs are needed (unique identifiers in RDF)
  - URI format: `http://example.org/periodical/{issn}`
  - How URIs enable linking in knowledge bases
- **Why:** Understanding RDF requires understanding URIs

### 5. BLAZEGRAPH CLIENT CLASS (COMPREHENSIVE)
**Search for:** `class _BlazegraphClient:`
- **What:** Complete explanation of what Blazegraph is and why it's used
- **Explains:**
  - What Blazegraph does (high-performance RDF database)
  - What RDF is (Resource Description Framework)
  - Why SPARQL is used (SQL for semantic data)
  - Two main operations: upload and query
  - Benefits of direct HTTP POST vs. alternatives

#### Sub-section: `upload_graph()` method
**Search for:** `def upload_graph(self, g: Graph) -> bool:`
- **What:** 5 step explanation of data upload process
- **Explains:**
  - Serialization to N-Triples format
  - SPARQL INSERT DATA statement creation
  - HTTP POST to Blazegraph endpoint
  - Success/failure handling
  - Fallback behavior
- **Why:** Understanding how data gets into the semantic database

#### Sub-section: `select()` method
**Search for:** `def select(self, query: str) -> List[Dict[str, Any]]:`
- **What:** SPARQL query execution explanation
- **Explains:**
  - SPARQL query structure and syntax
  - Example query for CC BY licenses
  - Result format (list of dictionaries)
  - Exception handling and fallback
  - Comprehensive guide to SPARQL for your professor
- **Why:** SPARQL queries are core to semantic database operations

### 6. JOURNAL UPLOAD HANDLER CLASS (COMPREHENSIVE)
**Search for:** `class JournalUploadHandler(UploadHandler):`
- **What:** Complete pipeline explanation from CSV to RDF
- **Explains:**
  - Data pipeline stages (Read → Parse → Convert → Upload → Cache)
  - RDF schema used (schema.org vocabulary)
  - RDF triple structure with examples
  - Why two-tier storage (Blazegraph + cache)
  - Flexibility in CSV format handling

#### Sub-section: `pushDataToDb()` method
**Search for:** `def pushDataToDb(self, file_path: str) -> bool:`
- **What:** Detailed step-by-step explanation of data processing
- **Explains:**
  - Path resolution for flexibility
  - Flexible column mapping (handles different CSV formats)
  - Data extraction and normalization
  - RDF triple construction
  - How APC and DOAJ Seal are stored as PropertyValues
  - Upload attempt and cache fallback
- **Why:** Shows robust data handling and semantic modeling

### 7. JOURNAL QUERY HANDLER CLASS (COMPREHENSIVE)
**Search for:** `class JournalQueryHandler(QueryHandler):`
- **What:** Architecture explanation for two-tier querying
- **Explains:**
  - Why primary use is cache (performance)
  - Why fallback to SPARQL (completeness)
  - Benefits of this architecture
  - Helper method purposes

#### Sub-section: `_aggregate_rows()` method
**Search for:** `def _aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:`
- **What:** Why and how SPARQL results are consolidated
- **Explains:**
  - Why multiple rows per journal occur (e.g., multiple languages)
  - Consolidation process with examples
  - Boolean conversion logic for APC and DOAJ Seal
  - Result transformation
- **Why:** Understanding result processing improves query literacy

#### Sub-section: `_select_df()` method
**Search for:** `def _select_df(self, where_filter: str = "", limit: Optional[int] = None) -> pd.DataFrame:`
- **What:** Two-tier query execution strategy in detail
- **Explains:**
  - Why cache is tried first
  - Regex pattern matching for text filtering
  - Filter parsing (extracting search terms)
  - SPARQL query fallback
  - Result limitation options
- **Why:** Core query execution method

#### Sub-section: Individual Query Methods
**Search for:** `def getById(self, id_value: str) -> pd.DataFrame:`
- **getById()** - Finding journals by ISSN or title
  - Search strategy explanation
  - Priority given to exact matches
  - Fallback to title matching

**Search for:** `def getAllJournals(self) -> pd.DataFrame:`
- **getAllJournals()** - Returns all 21,307+ journals
- Comments explain it retrieves the entire dataset

**Search for:** `def getJournalsWithTitle(self, text: str) -> pd.DataFrame:`
- **getJournalsWithTitle()** - Case-insensitive title search
- Example: Finding journals with "Nature" in title

**Search for:** `def getJournalsPublishedBy(self, text: str) -> pd.DataFrame:`
- **getJournalsPublishedBy()** - Publisher name search
- Example: Finding Springer-published journals

**Search for:** `def getJournalsWithLicense(self, licenses: Set[str]) -> pd.DataFrame:`
- **getJournalsWithLicense()** - Find by open access license
- Common licenses explained: CC BY, CC BY-NC, etc.

**Search for:** `def getJournalsWithAPC(self) -> pd.DataFrame:`
- **getJournalsWithAPC()** - Find journals charging article fees
- Explanation of Diamond OA (free journals)

**Search for:** `def getJournalsWithDOAJSeal(self) -> pd.DataFrame:`
- **getJournalsWithDOAJSeal()** - Find DOAJ-certified journals
- DOAJ Seal criteria explained

---

## 🎯 Comments by Topic

### To understand "What is a Journal in this system?"
→ Read: Journal class section (item #1)

### To understand "How does Blazegraph store data?"
→ Read: _build_journal_uri (#4) + upload_graph (#5)

### To understand "What is RDF and why use it?"
→ Read: _BlazegraphClient class introduction (#5)

### To understand "How do SPARQL queries work?"
→ Read: select() method (#5) + _select_df() (#7)

### To understand "How are categories and areas linked?"
→ Read: Relationship accessors (#3)

### To understand "Why does the system cache data?"
→ Read: JournalUploadHandler class (#6)

### To understand "How are journal searches performed?"
→ Read: JournalQueryHandler class (#7)

---

## 📊 Comment Density by Section

| Section | Comments Added | Purpose |
|---------|---------------|---------|
| Journal class | ~80 lines | Explain data model |
| Journal methods | ~120 lines | Document each accessor |
| _build_journal_uri | ~25 lines | Explain URI creation |
| _BlazegraphClient | ~180 lines | Complete Blazegraph guide |
| JournalUploadHandler | ~120 lines | Data pipeline explanation |
| JournalQueryHandler | ~280 lines | Query system documentation |
| **Total** | **~805 lines** | **Complete educational guide** |

---

## 🔍 How to Use These Comments

### For Self-Study:
1. Start with Journal class documentation
2. Understand RDF and URIs (_build_journal_uri)
3. Learn about Blazegraph (_BlazegraphClient)
4. Study data flow (JournalUploadHandler)
5. Understand queries (JournalQueryHandler)

### For Explaining to Professor:
1. **"What's a Journal?"** → Show Journal class
2. **"Why Blazegraph?"** → Show _BlazegraphClient introduction
3. **"How does data get in?"** → Show JournalUploadHandler
4. **"How do you query?"** → Show JournalQueryHandler
5. **"What if Blazegraph crashes?"** → Show fallback cache comments

### For Debugging:
- Each method has explained purpose
- Comments show expected inputs/outputs
- Examples provided for clarity
- Error handling explained

---

## ✨ Special Features of These Comments

✓ **Comprehensive** - Explain the "why" not just "what"
✓ **Educational** - Written to teach concepts, not just document code
✓ **Contextual** - Show how each part relates to the bigger picture
✓ **Searchable** - Use Ctrl+F to find specific topics
✓ **Example-based** - Many code examples provided
✓ **Professor-friendly** - Written at the level of academic explanation

---

## 📝 Next Steps

1. **Read through** the comments in order
2. **Understand** the overall architecture
3. **Ask questions** about unclear sections
4. **Explain to professor** using the provided frameworks
5. **Review COMMENTS_GUIDE.md** for additional context

Good luck with your presentation! 🚀
