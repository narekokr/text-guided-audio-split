from typing import List, Dict
from collections import defaultdict

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = defaultdict(lambda: {"history": [], "file": None})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions[session_id]["history"]

    def add_message(self, session_id: str, role: str, content: str):
        self.sessions[session_id]["history"].append({"role": role, "content": content})

    def set_file(self, session_id: str, file_path: str):
        self.sessions[session_id]["file"] = file_path

    def get_file(self, session_id: str):
        return self.sessions[session_id]["file"]

    def reset(self, session_id: str):
        self.sessions[session_id] = {"history": [], "file": None}

    def exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    def set_stem_file(self, session_id: str, stem: str, path: str):
        self.sessions[session_id]["stems"][stem] = path

    def get_stem_file(self, session_id: str, stem: str):
        return self.sessions[session_id]["stems"].get(stem)

session_manager = SessionManager()
