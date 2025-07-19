import json, pathlib
from engine.config import MEM_PATH


class DecayMemory(dict):
    def __init__(self, path=MEM_PATH, decay_rate=0.90):
        self.path = pathlib.Path(path)
        self.rate = decay_rate
        if self.path.exists():
            self.update(json.loads(self.path.read_text()))

    def remember(self, k, v, score=1.0):
        cur = self.get(k, {"v": v, "s": score})
        self[k] = {"v": v, "s": max(score, cur["s"])}

    def decay(self):
        for k in list(self):
            self[k]["s"] *= self.rate
            if self[k]["s"] < 0.05:
                del self[k]

    def save(self):
        self.path.write_text(json.dumps(self, indent=2))


mem = DecayMemory()
