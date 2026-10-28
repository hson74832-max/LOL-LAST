import re


def normalize_label(label: str) -> str:
    """Accept the common space, dash and underscore class-name spellings."""
    return re.sub(r"[\s_-]+", " ", str(label).strip().lower())
