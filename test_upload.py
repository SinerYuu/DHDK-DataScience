from impl import JournalUploadHandler
import requests

# Step 1 – upload the CSV
jou = JournalUploadHandler()
jou.setDbPathOrUrl("http://127.0.0.1:9999/blazegraph/sparql")
jou.pushDataToDb("data/doaj.csv")

# Step 2 – verify upload worked
q = """
PREFIX ex: <http://example.org/>
PREFIX dct: <http://purl.org/dc/terms/>
SELECT (COUNT(?j) AS ?n)
WHERE { ?j a ex:Journal . }
"""
resp = requests.post(
    "http://127.0.0.1:9999/blazegraph/sparql",
    data={"query": q},
    headers={"Accept": "application/sparql-results+json"},
    timeout=30
)
print(resp.json())
