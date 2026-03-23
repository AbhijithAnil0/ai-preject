from events.event_store import EventStore


class EventEngine:

    def __init__(self):

        # event database
        self.event_store = EventStore()

        # prevent duplicate events
        self.last_event = {}


    def process(self, obj_id, obj_class, action, color, bbox, frame_id, route_event=None):

        event = None

        # ---------- EXPLICIT ROUTE EVENT ----------
        if route_event is not None:
            event = route_event


        # ---------- PERSON EVENTS ----------
        if event is None and obj_class in ["person", "person_U"]:

            if action == "run":
                event = "PERSON_RUNNING"

            elif action == "walk":
                event = "PERSON_WALKING"

            elif action == "stand":
                event = "PERSON_STANDING"


        # ---------- VEHICLE EVENTS ----------
        elif event is None and obj_class in ["car", "truck", "bus", "motorcycle", "bicycle"]:

            event = "VEHICLE_DETECTED"


        # ---------- STORE EVENT ----------
        if event:
            event_record = {
                "id": obj_id,
                "class": obj_class,
                "event": event,
                "action": action,
                "color": color,
                "bbox": bbox,
                "frame": frame_id
            }
            self.event_store.add_event(event_record)
            print(f"EVENT: {event} | ID {obj_id}")


    def get_events(self):

        return self.event_store.get_all()