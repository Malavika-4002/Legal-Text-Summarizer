# src/summarizer/sentence_splitter.py
from typing import List
from src.sbd.inference import get_sentences

def split_into_sentences(text: str) -> List[str]:
    """
    Uses the trained SBD CRF+CNN pipeline to split text into sentences.
    Falls back to a simple newline+punctuation splitter if inference fails.
    """
    try:
        sents = get_sentences(text)
        if sents and len(sents) > 0:
            return sents
    except Exception:
        pass

    # Fallback (simple)
    import re
    splits = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s.strip() for s in splits if s.strip()]
