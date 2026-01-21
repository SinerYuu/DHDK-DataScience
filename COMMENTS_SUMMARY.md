# Summary of Code Comments Added

## What Was Added

I've added **comprehensive educational comments** throughout `impl.py` to explain the Journal class and Blazegraph integration. This makes it easy for you to understand and explain to your professor.

---

## 📍 JOURNAL CLASS (Enhanced Comments)

### What it represents:
- A single academic journal from DOAJ (Directory of Open Access Journals)
- Stores metadata: title, publisher, license, fees, languages
- Can be linked to research Categories and Areas

### Methods documented:
- **getTitle()** - Returns journal name
- **getPublisher()** - Returns publishing organization
- **getLicense()** - Returns open access license type
- **getAPC()** - Check if authors are charged article processing fees
- **getDOAJSeal()** - Check if journal has DOAJ quality certification
- **getLanguages()** - Get supported manuscript languages
- **getCategories()** - Get linked research categories
- **getAreas()** - Get linked research areas
- **getIds()** - Get unique identifier (the ISSN)

---

## 🔗 BLAZEGRAPH INTEGRATION (Enhanced Comments)

### Three main components:

#### 1. **_build_journal_uri()**
- Creates unique URI identifiers for journals in the RDF database
- Format: `http://example.org/periodical/{ISSN}`
- Used as the subject in RDF triples

#### 2. **_BlazegraphClient class**
**What it does:**
- Connects to Blazegraph (semantic database)
- Uploads RDF triples
- Executes SPARQL queries

**Key methods:**
- `upload_graph()` - Insert journal data as RDF
- `select()` - Query using SPARQL language

**Comments explain:**
- What Blazegraph is (high-performance RDF database)
- What RDF is (semantic data format with triples)
- Why SPARQL is used (SQL-like queries for linked data)
- What happens if upload fails (graceful fallback)

#### 3. **JournalUploadHandler class**
**What it does:**
- Reads DOAJ CSV file with journal metadata
- Converts to RDF triples using schema.org vocabulary
- Uploads to Blazegraph
- Maintains local cache as fallback

**Comments explain:**
- Data pipeline (CSV → Parse → RDF → Blazegraph)
- Schema.org vocabulary used (Periodical class)
- RDF triple structure with examples
- Why caching matters
- How it handles missing data

#### 4. **JournalQueryHandler class**
**What it does:**
- Queries journals from Blazegraph or cache
- Two-tier strategy: try cache first, then SPARQL
- Aggregates multi-row results into single journal rows

**Comments explain:**
- Why two-tier approach (performance + reliability)
- How aggregation works (consolidating multiple rows)
- Each query method's purpose and examples

**Query methods documented:**
- `getById()` - Find by ISSN or title
- `getAllJournals()` - Get all journals
- `getJournalsWithTitle()` - Search by name
- `getJournalsPublishedBy()` - Search by publisher
- `getJournalsWithLicense()` - Find by open access license
- `getJournalsWithAPC()` - Find journals that charge fees
- `getJournalsWithDOAJSeal()` - Find certified journals

---

## 📚 Key Concepts Explained

### Open Access Terms
- **DOAJ** - Directory of Open Access Journals
- **Diamond OA** - Free for authors and readers (APC=false)
- **Gold OA** - Free for readers, may charge authors (APC=true)
- **DOAJ Seal** - Award for high-quality open access journals

### License Types (Most to Least Permissive)
1. **CC BY** - Any use with attribution
2. **CC BY-NC** - Non-commercial use only
3. **CC BY-SA** - Must share-alike
4. **CC BY-NC-SA** - Non-commercial + share-alike

### Technical Concepts
- **RDF** - Machine-readable semantic data format
- **SPARQL** - Query language for RDF (like SQL for graphs)
- **Triple** - Subject → Predicate → Object (RDF unit)
- **URI** - Unique identifier (like http://example.org/periodical/ISSN)
- **Ontology** - Formal specification of concepts and relationships
- **Schema** - Standard vocabulary (schema.org used here)

---

## 🎯 What You Can Now Explain to Professor

### Journal Class
✓ How journals are represented as objects  
✓ What metadata is stored for each journal  
✓ How relationships to categories/areas work  
✓ Why multiple identifiers are needed  

### Blazegraph System
✓ What problem RDF/Blazegraph solves  
✓ Why semantic databases are better than simple SQL  
✓ How data flows from CSV to Blazegraph  
✓ How queries are executed  
✓ What happens if Blazegraph fails  

### Data Quality
✓ How DOAJ ensures journal quality (DOAJ Seal)  
✓ Different open access models (Diamond, Gold)  
✓ Why license information matters  
✓ How APC (Article Processing Charges) work  

### System Design
✓ Why caching is important  
✓ Two-tier query strategy benefits  
✓ How the system is fault-tolerant  
✓ Flexibility in handling different data formats  

---

## 📖 Additional Resource

See **COMMENTS_GUIDE.md** for:
- Detailed explanations of all sections
- Examples of RDF triples and SPARQL queries
- Likely professor questions and answers
- Testing code snippets
- Key concepts and terminology

---

## ✅ Everything Still Works

- ✓ All syntax is valid
- ✓ All methods function correctly
- ✓ No performance changes
- ✓ Comments are educational and thorough
- ✓ Code is production-ready

You can now confidently explain:
1. **What** each component does
2. **Why** it was designed this way
3. **How** data flows through the system
4. **What happens** if things fail
