"""Text helpers shared by every part: word normalisation, singulars, the term pattern."""
import re
from functools import lru_cache

TERM_RE = re.compile(r"\b(\d{1,3})\s*[- ]?\s*(month|mo|year|yr)s?\b", re.I)
_WORD = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=65536)
def normalize(text: str) -> str:
    """Lower-case words separated by single spaces; apostrophes dropped ("investor's" -> "investors")."""
    t = (text or "").lower().replace("&", " and ").replace("’", "").replace("'", "")
    return " ".join(_WORD.findall(t))


def singular(tok: str) -> str:
    """Simple English singular: lobbies -> lobby, loans -> loan; Illinois, bonus, address stay."""
    if len(tok) <= 3 or not tok.endswith("s") or tok.endswith(("ss", "is", "us", "ous")):
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    return tok[:-1]
