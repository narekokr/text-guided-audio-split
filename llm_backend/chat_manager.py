from typing import List, Dict
from collections import defaultdict

class SessionManager:
    def __init__(self):
        # session_id -> list of {"role": "user"/"assistant", "content": "..."}
        self.sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        self.sessions[session_id].append({"role": role, "content": content})

    def reset(self, session_id: str):
        self.sessions[session_id] = []

    def exists(self, session_id: str) -> bool:
        return session_id in self.sessions
