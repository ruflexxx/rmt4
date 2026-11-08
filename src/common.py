"""Frozen protocol constants, data loading, prompts, label parsing. Do NOT edit after the freeze (see Appendix A)."""
import os, re, json, hashlib, random, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# ---------------- FROZEN SETTINGS ----------------
SEED = 42
N_TRANS = 100            # FLORES-200 devtest random subset (reduced from 500 after the freeze because of API quotas; exploratory)
N_RUNS = {"cls": 1, "trans": 1}   # reduced from cls 3 after the freeze because of API quotas; temperature 0, so one run per configuration
TEMPERATURE = 0.0
MAX_TOKENS = 1024
N_BOOT = 1000
LABELS = ["entertainment", "geography", "health", "politics", "science/technology", "sports", "travel"]
LANG_NAME = {"eng": "English", "kaz": "Kazakh"}

MODELS = {  # final list; the original freeze listed llama-3.3-70b and gemini-2.5-flash-lite, which providers removed (see report, Section 3, item 8)
    # rpm = requests per minute; set to your real account limit (operational setting, not a scientific parameter)
    "gpt-oss-120b":   dict(type="open-weight", base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY",   model_id="openai/gpt-oss-120b", rpm=20),
    "qwen3.8-27b":    dict(type="open-weight", base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY",   model_id="qwen/qwen3.8-27b",    rpm=30),
    "gemini-3.5-flash-lite": dict(type="commercial", base_url="https://generativelanguage.googleapis.com/v1beta/openai/", key_env="GEMINI_API_KEY", model_id="gemini-3.5-flash-lite", rpm=10),
}

TRANS_PROMPT = ("Translate the following text from {src} to {tgt}. Output only the translation, "
                "with no explanations and no quotation marks.\n\n{text}")
CLS_PROMPT = ("Classify the topic of the following sentence. Choose exactly one label from this list: "
              + ", ".join(LABELS) + ". Answer with the label only.\n\nSentence: {text}")

# ---------------- DATA ----------------
def _read(p):
    return pathlib.Path(p).read_text(encoding="utf-8").rstrip("\n").split("\n")

def load_flores(split="devtest"):
    d = DATA / "flores200_dataset" / split
    eng, kaz = _read(d / f"eng_Latn.{split}"), _read(d / f"kaz_Cyrl.{split}")
    assert len(eng) == len(kaz), "FLORES line counts differ"
    return eng, kaz

def load_sib(split="test"):
    from datasets import load_dataset
    e = load_dataset("Davlan/sib200", "eng_Latn", split=split)
    k = load_dataset("Davlan/sib200", "kaz_Cyrl", split=split)
    E = {r["index_id"]: r for r in e}; K = {r["index_id"]: r for r in k}
    ids = sorted(E)
    assert ids == sorted(K), "SIB-200 index_id mismatch"
    for i in ids:
        assert E[i]["category"] == K[i]["category"], f"label mismatch {i}"
    return [dict(id=i, label=E[i]["category"], eng=E[i]["text"], kaz=K[i]["text"]) for i in ids]

def trans_sample_ids(n_total=1012):
    return sorted(random.Random(SEED).sample(range(n_total), N_TRANS))

def sha256(path):
    h = hashlib.sha256(); h.update(pathlib.Path(path).read_bytes()); return h.hexdigest()

# ---------------- PARSING ----------------
def strip_think(s):
    return re.sub(r"<think>.*?</think>", "", s or "", flags=re.S).strip()

def parse_label(ans):
    a = re.sub(r"[^a-z/ ]", "", strip_think(ans).lower()).strip()
    if a in LABELS: return a
    hits = [l for l in LABELS if l in a]
    return hits[0] if len(hits) == 1 else "INVALID"
