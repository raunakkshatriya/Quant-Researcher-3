"""Score headlines with FinBERT on the GPU, resumably.

FinBERT (ProsusAI/finbert) reads a headline and returns three probabilities: positive, negative, neutral.
Our score = P(positive) - P(negative), between -1 (bad news) and +1 (good news).
The model downloads once (~440 MB) into your Hugging Face cache. Scores are cached in data/news_scores.csv
so each article is only ever scored once.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "ProsusAI/finbert"
SCORES_FILE = Path(__file__).resolve().parents[2] / "data" / "news_scores.csv"


def finbert_scorer(device="cuda", batch_size=128, max_length=64):
    """Return a function texts -> numpy array of scores. Refuses to silently fall back to the CPU."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see your GPU (torch.cuda.is_available() is False). Run  python check_gpu.py  "
                           "and follow docs/ALPACA_SETUP.md section 'GPU'. (Use --cpu to run on the CPU anyway, slowly.)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    if device.startswith("cuda"):
        model = model.half()                       # half precision: ~2x faster, half the memory, same answers to ~3 decimals
    labels = {i: l.lower() for i, l in model.config.id2label.items()}
    pos = [i for i, l in labels.items() if l == "positive"][0]
    neg = [i for i, l in labels.items() if l == "negative"][0]

    @torch.inference_mode()
    def score(texts):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
            out.append(p[:, pos] - p[:, neg])
        return np.concatenate(out) if out else np.array([])
    return score


def score_articles(articles, scorer, cache_path=SCORES_FILE, chunk=4096, progress=True):
    """Score every unique article id once (resumable). Returns a Series: article id -> score."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    have = pd.read_csv(cache_path).drop_duplicates("id").set_index("id")["score"] if cache_path.exists() else pd.Series(dtype=float)
    todo = articles.drop_duplicates("id")
    todo = todo[~todo["id"].isin(have.index) & todo["headline"].str.len().gt(0)]
    todo = todo.assign(n=todo["headline"].str.len()).sort_values("n")          # similar lengths batch together = faster
    t0, done = time.time(), 0
    for i in range(0, len(todo), chunk):
        part = todo.iloc[i:i + chunk]
        s = scorer(part["headline"].tolist())
        pd.DataFrame({"id": part["id"].values, "score": s}).to_csv(cache_path, mode="a", header=not cache_path.exists(), index=False)
        done += len(part)
        if progress:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  scored {done:,} / {len(todo):,} headlines ({rate:,.0f}/s, about {(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min left)")
    return pd.read_csv(cache_path).drop_duplicates("id").set_index("id")["score"] if cache_path.exists() else pd.Series(dtype=float)
