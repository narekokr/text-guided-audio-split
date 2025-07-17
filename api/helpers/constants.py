from enum import Enum

DEFAULT_VOLUMES = {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0}
VALID_STEMS = ["vocals", "drums", "bass", "other"]
SAMPLE_RATE = 44100
SESSION_TASK_REMIX = "remix"
SESSION_TASK_SEPARATION = "separation"

SILENCE_THRESHOLD = -40
MIN_SILENCE_LENGTH = 1000

SEPARATED_FILES_DIR = "separated"
DOWNLOADS_MOUNT_PATH = "/downloads"

class IntentType(Enum):
    SEPARATION = "separation"
    REMIX = "remix"
    CLARIFICATION = "clarification"