import json

class EventStore:

    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)

    def get_all(self):
        return self.events

    def search(self, key, value):

        results = []

        for e in self.events:
            if key in e and e[key] == value:
                results.append(e)

        return results

    def save(self, filename="events.json"):

        with open(filename, "w") as f:
            json.dump(self.events, f, indent=4)