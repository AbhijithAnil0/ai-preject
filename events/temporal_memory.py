class PersonMemory:

    def __init__(self):

        self.memory = {}

    def update(self, person_id, action, frame_id):

        if person_id not in self.memory:

            self.memory[person_id] = []

        self.memory[person_id].append({
            "frame": frame_id,
            "action": action
        })

    def get_history(self, person_id):

        return self.memory.get(person_id, [])