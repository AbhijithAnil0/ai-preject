import json


class QueryEngine:

    def __init__(self, event_file="events.json"):

        with open(event_file, "r") as f:
            self.events = json.load(f)


    def search(self, filters):

        results = []

        for e in self.events:

            match = True

            if "object" in filters:
                if filters["object"] not in e["class"]:
                    match = False

            if "color" in filters:
                if e["color"] != filters["color"]:
                    match = False

            if "action" in filters:
                if e["action"] != filters["action"]:
                    match = False

            if "umbrella" in filters:
                if filters["umbrella"] and "person_U" not in e["class"]:
                    match = False

            if match:
                results.append(e)

        return results