from impl import (
    CategoryUploadHandler, CategoryQueryHandler,
    JournalUploadHandler, JournalQueryHandler,
    FullQueryEngine
)

REL = "relational.db"
SPARQL = "http://127.0.0.1:9999/blazegraph/sparql"

# 1) load relational
cat_up = CategoryUploadHandler(); cat_up.setDbPathOrUrl(REL)
cat_up.pushDataToDb("data/scimago.json")

# 2) load graph
jou_up = JournalUploadHandler(); jou_up.setDbPathOrUrl(SPARQL)
jou_up.pushDataToDb("data/doaj.csv")

# 3) handlers
cat_q = CategoryQueryHandler(); cat_q.setDbPathOrUrl(REL)
jou_q = JournalQueryHandler();  jou_q.setDbPathOrUrl(SPARQL)

# 4) engine
qe = FullQueryEngine(); qe.addCategoryHandler(cat_q); qe.addJournalHandler(jou_q)

print("Journals:", [j.getIds()[0] for j in qe.getAllJournals()][:5])
print("Title contains 'ai':", [j.getIds()[0] for j in qe.getJournalsWithTitle("ai")])
print("License CC-BY:", [j.getIds()[0] for j in qe.getJournalsWithLicense("CC-BY")])
print("APC:", [j.getIds()[0] for j in qe.getJournalsWithAPC()])
print("Seal:", [j.getIds()[0] for j in qe.getJournalsWithDOAJSeal()])

print("By ID (journal):", type(qe.getEntityById("2532-8816")).__name__)
print("By ID (category):", type(qe.getEntityById("Artificial Intelligence")).__name__)

print("Q1 in {AI, Oncology}:", [j.getIds()[0] for j in
      qe.getJournalsInCategoriesWithQuartile({"Artificial Intelligence","Oncology"}, {"Q1"})])
