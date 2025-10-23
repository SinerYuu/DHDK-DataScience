# Supposing that all the classes developed for the project
# are contained in the file 'impl.py', then:

# 1) Importing all the classes for handling the relational database
from impl import CategoryUploadHandler, CategoryQueryHandler

# 2) Importing all the classes for handling graph database
from impl import JournalUploadHandler, JournalQueryHandler

# 3) Importing the class for dealing with mashup queries
from impl import FullQueryEngine

# Once all the classes are imported, first create the relational
# database using the related source data
rel_path = "relational.db"
cat = CategoryUploadHandler()
cat.setDbPathOrUrl(rel_path)
cat.pushDataToDb("data/scimago.json")
# Please remember that one could, in principle, push one or more files
# calling the method one or more times (even calling the method twice
# specifying the same file!)

# Then, create the graph database (remember first to run the
# Blazegraph instance) using the related source data
grp_endpoint = "http://127.0.0.1:9999/blazegraph/sparql"
jou = JournalUploadHandler()
jou.setDbPathOrUrl(grp_endpoint)
jou.pushDataToDb("data/doaj.csv")
# Please remember that one could, in principle, push one or more files
# calling the method one or more times (even calling the method twice
# specifying the same file!)

# In the next passage, create the query handlers for both
# the databases, using the related classes
cat_qh = CategoryQueryHandler()
cat_qh.setDbPathOrUrl(rel_path)

jou_qh = JournalQueryHandler()
jou_qh.setDbPathOrUrl(grp_endpoint)

# Finally, create a advanced mashup object for asking
# about data
que = FullQueryEngine()
que.addCategoryHandler(cat_qh)
que.addJournalHandler(jou_qh)

result_q1 = que.getAllJournals()
result_q2 = que.getJournalsInCategoriesWithQuartile({"Artificial Intelligence", "Oncology"}, {"Q1"})
result_q3 = que.getEntityById("1027-202X")
result_q4 = que.getEntityById("2615-1065")
# etc...

# === Utility function for neat printing ===
def show_entity(obj):
    if not obj:
        return "No entity found."
    cls = obj.__class__.__name__
    name = getattr(obj, "getName", lambda: "N/A")()
    details = []
    if hasattr(obj, "getLicense"):
        details.append(f"License: {obj.getLicense()}")
    if hasattr(obj, "getQuartile"):
        details.append(f"Quartile: {obj.getQuartile()}")
    return f"[{cls}] {name} ({', '.join(details) if details else 'No additional info'})"

# === 6. Display results ===
print("== All Journals ==")
for j in result_q1[:5]:   # show only first 5
    print("-", j.getName(), j.getLanguages())

print("\n== Q1 Quartile Journals ==")
for j in result_q2[:5]:
    print("-", j.getName(), j.getLicense())

print("\n== Single Category Query ==")
print(show_entity(result_q3))

print("\n== Single Journal Query ==")
print(show_entity(result_q4))


