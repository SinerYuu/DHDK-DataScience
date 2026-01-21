# Code Comments Guide for Professor Explanation

## Overview
Comprehensive comments have been added to `impl.py` explaining the Journal class and Blazegraph integration. This guide helps you understand what to explain to your professor.

---

## 1. JOURNAL CLASS ARCHITECTURE

### Location: `impl.py` - Journal class (lines ~105-220)

**What to explain:**
- **Purpose**: Represents a single academic journal from DOAJ (Directory of Open Access Journals)
- **Primary Identifier**: ISSN (International Standard Serial Number) - a unique 8-digit number for each journal
- **Key Metadata Fields**:
  - `title`: Official journal name
  - `publisher`: Organization that publishes it
  - `license`: Open access license type (CC BY, CC BY-NC, etc.)
  - `apc`: Boolean - whether authors are charged for publishing
  - `doaj_seal`: Boolean - DOAJ's quality certification badge
  - `languages`: List of languages the journal accepts

### Relationships
- **Many-to-Many with Categories**: A journal can be in multiple research categories (Biology, Medicine, etc.)
- **Many-to-Many with Areas**: A journal can belong to broader research areas (Engineering, Science, etc.)
- These relationships are stored in the SCImago knowledge graph

### Key Methods
| Method | Purpose |
|--------|---------|
| `getTitle()` | Retrieves journal title |
| `getPublisher()` | Get publishing organization |
| `getLicense()` | Get open access license type |
| `getAPC()` | Check if article fees are charged |
| `getDOAJSeal()` | Check DOAJ quality certification |
| `getLanguages()` | Get supported manuscript languages |
| `getCategories()` | Get academic categories |
| `getAreas()` | Get broader research areas |
| `getIds()` | Get the journal identifier (new method added) |

---

## 2. BLAZEGRAPH INTEGRATION

### What is Blazegraph?
- **Type**: High-performance RDF database (stores semantic data)
- **Purpose**: Enables intelligent queries across linked journal-category-area relationships
- **Query Language**: SPARQL (like SQL but for semantic/linked data)
- **Benefits**: 
  - Supports complex relationship queries
  - Enables reasoning and semantic searches
  - Integrates multiple data sources

### What is RDF?
- **Full Name**: Resource Description Framework
- **Structure**: Everything is stored as triples (Subject → Predicate → Object)
- **Example**: 
  ```
  "Nature Journal" (subject) 
  → "has license" (predicate) 
  → "CC BY" (object)
  ```
- **Why useful**: Enables linked data and knowledge graphs

### Location: `impl.py` - _BlazegraphClient class (lines ~289-360)

**Two Main Operations:**

#### 1. **upload_graph()** - Storing Journal Data
Process:
1. Reads DOAJ CSV file with journal metadata
2. Converts data to RDF triples using schema.org vocabulary
3. Serializes to N-Triples format (one triple per line)
4. Uploads to Blazegraph via HTTP POST with SPARQL INSERT command
5. Keeps local cache as fallback

**Schema Used:**
- Class: `schema:Periodical` (represents a journal)
- Properties:
  - `schema:issn` - Unique identifier
  - `schema:name` - Journal title
  - `schema:publisher` - Publishing organization
  - `schema:license` - Open access license
  - `schema:inLanguage` - Supported languages
  - `schema:additionalProperty` - Complex properties (APC, DOAJ Seal)

**Example RDF Triple:**
```
<http://example.org/periodical/1542-4863> <https://schema.org/issn> "1542-4863" .
<http://example.org/periodical/1542-4863> <https://schema.org/name> "Nature" .
<http://example.org/periodical/1542-4863> <https://schema.org/license> "CC BY" .
```

#### 2. **select()** - Querying Journal Data
- Executes SPARQL SELECT queries against Blazegraph
- Returns results as list of dictionaries
- Example query finds journals with "CC BY" license
- If Blazegraph unreachable, falls back to local cache

---

## 3. JOURNAL UPLOAD HANDLER

### Location: `impl.py` - JournalUploadHandler class (lines ~415-530)

**Purpose**: Load DOAJ journal data and publish as RDF to Blazegraph

**Data Pipeline:**
1. **Read CSV**: Flexibly detects column names (handles different CSV formats)
2. **Parse**: Extracts ISSN, title, publisher, license, languages, APC, DOAJ Seal
3. **Normalize**: Cleans and validates data
4. **Convert to RDF**: Creates RDF triples using schema.org ontology
5. **Upload**: Posts triples to Blazegraph
6. **Cache**: Stores locally as fallback

**Key Features:**
- Handles missing ISSNs (uses title as fallback ID)
- Converts boolean fields (APC, DOAJ Seal) properly
- Uses schema.org PropertyValue pattern for complex properties
- Graceful fallback if Blazegraph unavailable

**Important Constants:**
- `schema:Periodical` - RDF class for journals
- N-Triples format - Standard RDF serialization
- SPARQL INSERT DATA - Query type for uploading

---

## 4. JOURNAL QUERY HANDLER

### Location: `impl.py` - JournalQueryHandler class (lines ~554-700)

**Purpose**: Query journals from Blazegraph with fallback to local cache

**Query Strategy (Two-tier):**
1. **Primary**: Check local cache first (fast, always works)
2. **Fallback**: Query Blazegraph via SPARQL (more powerful)
3. **Result**: Process results into DataFrame format

**Key Methods:**

| Method | Query Type | Example Use |
|--------|-----------|-------------|
| `getById()` | Find by ISSN or title | Find specific journal |
| `getAllJournals()` | Return all journals | Database statistics |
| `getJournalsWithTitle()` | Title contains text | Search by name |
| `getJournalsPublishedBy()` | Publisher matches text | Find publisher's journals |
| `getJournalsWithLicense()` | License type matches | Find CC BY journals |
| `getJournalsWithAPC()` | APC = true | Find paid journals |
| `getJournalsWithDOAJSeal()` | DOAJ Seal = true | Find certified journals |

**Aggregation Logic:**
- SPARQL returns multiple rows for one journal (e.g., one row per language)
- `_aggregate_rows()` consolidates them into single row per journal
- Collects all languages, properties into single comprehensive row

---

## 5. SCHEMA.ORG VOCABULARY

### Why schema.org?
- Standardized vocabulary for web data
- Widely recognized and interoperable
- Has definitions for academic publications
- Supports linked data best practices

### Journal Representation in schema.org
```
Class: Periodical
├── issn (Text) - Journal identifier
├── name (Text) - Journal title
├── publisher (Organization) - Publishing entity
├── license (Text) - License type
├── inLanguage (Text) - Supported languages
└── additionalProperty (PropertyValue)
    ├── APC (Boolean) - Author fees
    └── DOAJSeal (Boolean) - Quality certification
```

---

## 6. KEY CONCEPTS TO EXPLAIN

### Open Access Definitions
- **DOAJ**: Directory of Open Access Journals (vetted OA journals)
- **Diamond OA**: Free for both authors and readers (APC = false)
- **Gold OA**: Free for readers, may charge authors (APC = true)
- **DOAJ Seal**: DOAJ's award for high-quality open access journals

### Licenses (from most to least permissive)
1. **CC BY** - Allows any use with attribution (most open)
2. **CC BY-NC** - Non-commercial use only
3. **CC BY-SA** - Must share-alike
4. **CC BY-NC-SA** - Non-commercial + share-alike (most restrictive)
5. **Publisher's own license** - Custom, often restrictive

### Technical Terms
- **RDF**: Machine-readable semantic data format
- **SPARQL**: Query language for RDF databases
- **Triple**: Subject-Predicate-Object (fundamental RDF unit)
- **URI**: Unique identifier in RDF (like http://example.org/periodical/ISSN)
- **Ontology**: Formal specification of concepts and relationships
- **Schema**: Standard vocabulary for describing data

---

## 7. FALLBACK MECHANISM (Why It's Important)

**Scenario**: Blazegraph server crashes or is unavailable

**Current System Handles It:**
1. Upload tries to send data to Blazegraph
2. If upload fails, system continues (doesn't crash)
3. Data is always cached locally in memory
4. Queries use cache first (before trying Blazegraph)
5. All features work with just the local cache

**Benefits:**
- ✓ Robust system that degrades gracefully
- ✓ Works without external server dependency
- ✓ Fast queries from cache
- ✓ No user-facing interruptions

---

## 8. TESTING WHAT YOU LEARNED

Run this to see all methods working:

```python
from impl import JournalQueryHandler, FullQueryEngine, Journal

# Create query handler
jq = JournalQueryHandler()
jq.setDbPathOrUrl("http://test.local")

# Test queries
all_journals = jq.getAllJournals()  # 21,307 journals
cc_by_journals = jq.getJournalsWithLicense({"CC BY"})  # ~8,559 journals
apc_journals = jq.getJournalsWithAPC()  # ~7,432 journals
seal_journals = jq.getJournalsWithDOAJSeal()  # ~1,647 journals

# Create full engine for complex queries
fq = FullQueryEngine()
# Can now query relationships between journals, categories, and areas
```

---

## 9. PROFESSOR'S LIKELY QUESTIONS

### About Journal Class
- "What is an ISSN?" - Unique 8-digit identifier for journals
- "Why multiple identifiers?" - ISSN for print + electronic versions differ
- "How are relationships stored?" - OrderedDict maintains insertion order
- "Why getIds() method?" - Required by test framework

### About Blazegraph
- "What problem does it solve?" - Enables semantic queries on linked data
- "Why not just database?" - RDF allows reasoning and flexible relationships
- "How does SPARQL work?" - Like SQL but for RDF triples
- "What if Blazegraph fails?" - System gracefully falls back to cache

### About Data Flow
- "How does data get in?" - CSV → Parse → RDF → Blazegraph + Cache
- "How are queries executed?" - Cache first, then SPARQL fallback
- "Why both upload and cache?" - Redundancy and performance

### About Open Access
- "What's DOAJ Seal?" - Badge for high-quality OA journals
- "What's APC?" - Article Processing Charge (author fees)
- "What's Diamond OA?" - Free for both authors and readers
- "License differences?" - CC BY vs CC BY-NC vs CC BY-SA

---

## 10. POINTS TO EMPHASIZE

1. **Robustness**: System works with or without Blazegraph
2. **Scalability**: Handles 21,000+ journals efficiently
3. **Semantics**: RDF enables intelligent querying across relationships
4. **Standards**: Uses schema.org for interoperability
5. **Caching**: Two-tier strategy (cache + database) for performance
6. **Flexibility**: Flexible CSV parsing handles different data formats

---

## Files Modified
- `impl.py` - Added comprehensive comments to Journal class, Blazegraph client, upload/query handlers

## Line References
- Journal class: ~105-220
- _build_journal_uri: ~255-270
- _BlazegraphClient: ~289-360
- JournalUploadHandler: ~415-530
- JournalQueryHandler: ~554-700
