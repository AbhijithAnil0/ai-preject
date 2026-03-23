from llm_parser import parse_query
from query_engine import QueryEngine

print("Video query system ready.")
print("Type a question about the video or type 'exit'.")

engine = QueryEngine("../events.json")

while True:

    query = input("\nAsk something about the video: ")

    if query.lower() == "exit":
        print("Exiting query system.")
        break

    filters = parse_query(query)

    print("\nParsed filters:", filters)

    results = engine.search(filters)

    print("\nMatches found:", len(results))

    for r in results[:5]:
        print(r)