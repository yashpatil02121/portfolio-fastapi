"""
Knowledge base loader + zero-cost retrieval over resume/yash-profile.json.

The JSON is already retrieval-shaped: every `chunk.text` is self-contained and
names its own subject, and every chunk carries `q` (example questions) and
`tags`. So we never need to send the whole resume to the model - we match the
user question locally and send only the 1-2 chunks that matter.
"""

import json
import math
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_PATH = os.path.join(BASE_DIR, "resume", "yash-profile.json")

# How many chunks may ever reach the model, and how weak a match we still trust.
MAX_CHUNKS = 2
MIN_SCORE = 6.0          # below this we treat the question as out of scope
SECOND_CHUNK_RATIO = 0.5  # 2nd chunk only if it is at least half as good as the 1st

_STOPWORDS = {
    "a", "about", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "been", "by", "can", "could", "did", "do", "does", "for", "from", "get",
    "give", "had", "has", "have", "he", "her", "him", "his", "how", "i", "in",
    "is", "it", "its", "know", "like", "list", "me", "much", "my", "of", "on",
    "or", "please", "s", "she", "so", "some", "tell", "that", "the", "their",
    "them", "there", "they", "this", "to", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "yash", "patil",  # the subject's own name carries no signal here
}

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def normalize(text: str) -> str:
    """Lowercase, collapse punctuation to spaces - used for phrase matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.]+", " ", text.lower())).strip()


def tokenize(text: str) -> set:
    return {
        t.strip(".")
        for t in _TOKEN_RE.findall(text.lower())
        if t.strip(".") and t.strip(".") not in _STOPWORDS and len(t.strip(".")) > 1
    }


class KnowledgeBase:
    def __init__(self, path: str = PROFILE_PATH):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.profile = data["profile"]
        self.chunks = data["chunks"]
        self.by_id = {c["id"]: c for c in self.chunks}

        # Pre-compute per-chunk token sets once at startup (not per request).
        self._index = []
        for c in self.chunks:
            tags = [normalize(t) for t in c.get("tags", [])]
            self._index.append(
                {
                    "id": c["id"],
                    "tag_phrases": [t for t in tags if " " in t],
                    "tag_tokens": tokenize(" ".join(c.get("tags", []))),
                    "q_tokens": tokenize(" ".join(c.get("q", []))),
                    "text_tokens": tokenize(c.get("text", "")),
                }
            )

        # IDF over chunk text so rare terms ("razorpay", "frappe") outweigh
        # common ones ("developer", "python").
        n = len(self._index)
        df = {}
        for entry in self._index:
            for tok in entry["text_tokens"] | entry["tag_tokens"] | entry["q_tokens"]:
                df[tok] = df.get(tok, 0) + 1
        self._idf = {tok: math.log(1 + n / count) for tok, count in df.items()}

        # Tokens that only a handful of chunks are tagged with ("celery",
        # "django", "razorpay"). Their presence means the question is about a
        # specific topic, not a one-line profile fact.
        tag_df = {}
        for entry in self._index:
            for tok in entry["tag_tokens"]:
                tag_df[tok] = tag_df.get(tok, 0) + 1
        self.specific_tokens = {t for t, c in tag_df.items() if c <= 3}
        self.tag_tokens = {e["id"]: e["tag_tokens"] for e in self._index}

        self.profile_card = self._build_profile_card()

    def _idf_overlap(self, q_tokens: set, doc_tokens: set) -> float:
        return sum(self._idf.get(t, 1.0) for t in q_tokens & doc_tokens)

    def _build_profile_card(self) -> str:
        """~60 tokens of always-true identity, cheap enough to send every time."""
        p = self.profile
        return (
            f"{p['full_name']} ({p['preferred_name']}) - {p['headline']}, "
            f"{p['years_experience']} years experience. {p['current_role']}. "
            f"Based in {p['location']}."
        )

    def search(self, query: str, max_chunks: int = MAX_CHUNKS):
        """Return [(chunk, score)] for the best-matching chunks, best first."""
        q_norm = normalize(query)
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scored = []
        for entry in self._index:
            score = 0.0
            # A multi-word tag appearing verbatim ("current job", "final year
            # project") is the strongest signal we have.
            for phrase in entry["tag_phrases"]:
                if phrase in q_norm:
                    score += 6.0
            score += 2.5 * self._idf_overlap(q_tokens, entry["q_tokens"])
            score += 2.0 * self._idf_overlap(q_tokens, entry["tag_tokens"])
            score += 1.0 * self._idf_overlap(q_tokens, entry["text_tokens"])
            if score > 0:
                scored.append((self.by_id[entry["id"]], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored or scored[0][1] < MIN_SCORE:
            return []

        best = scored[0][1]
        keep = [scored[0]]
        for chunk, score in scored[1:max_chunks]:
            if score >= best * SECOND_CHUNK_RATIO:
                keep.append((chunk, score))
        return keep


kb = KnowledgeBase()
