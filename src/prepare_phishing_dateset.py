# prepare_phish_dataset_balanced.py
import os
import random
import re
from pathlib import Path
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# --------- CONFIG ---------
INPUT_CSV = "data/raw/CEAS_08.csv"       # change this if your CSV filename differs
OUTPUT_DIR = "data/prepared_dataset"
PHISH_TARGET = 750             # desired Malicious count
BENIGN_TARGET = 750            # desired Benign count
# --------------------------

def load_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return df

def ensure_columns(df):
    """Ensure expected columns exist; create blanks if missing."""
    for c in ["sender", "receiver", "date", "subject", "body", "label", "urls"]:
        if c not in df.columns:
            df[c] = ""
    return df

def normalize_label_to_text(label_str):
    """Map various encodings to 'Malicious' / 'Benign'; return None if unknown."""
    x = str(label_str).strip().lower()
    if x in {"1","phish","phishing","spam","true","malicious","attack","fraud","bad"}:
        return "Malicious"
    if x in {"0","ham","benign","false","legit","legitimate","good","safe"}:
        return "Benign"
    # allow exact-cased already-good values
    if label_str in {"Malicious","Benign"}:
        return label_str
    # unknown
    return None

def combined_text(row):
    subj = (row.get("subject") or "").strip()
    body = (row.get("body") or "").strip()
    return subj + "\n\n" + body

# tiny obfuscation to diversify upsampled Malicious examples (optional)
def obfuscate_text(text):
    zws = "\u200b"
    swaps = {"o":"0", "l":"1", "i":"1", "s":"$"}  # simple, safe swaps
    txt = text
    for k,v in swaps.items():
        if random.random() < 0.25:
            txt = re.sub(rf"(?i)\b{k}\b", v, txt)
    if random.random() < 0.6 and len(txt) > 0:
        idx = random.randint(0, len(txt)-1)
        txt = txt[:idx] + zws + txt[idx:]
    txt = re.sub(r"(?i)verify", "ver i fy", txt)
    txt = re.sub(r"(?i)password", "pass word", txt)
    return txt

def sample_balanced_text(df_mal, df_ben, target_mal, target_ben):
    mal_count, ben_count = len(df_mal), len(df_ben)

    # --- Malicious ---
    if mal_count >= target_mal:
        mal_out = df_mal.sample(n=target_mal, random_state=42).copy()
        mal_out["text"] = mal_out.apply(combined_text, axis=1)
    else:
        needed = target_mal - mal_count
        base = df_mal.copy()
        if mal_count == 0:
            raise ValueError("No Malicious rows available. Provide at least one 'Malicious' example.")
        rep = base.sample(n=needed, replace=True, random_state=42).copy()
        rep["text"] = rep.apply(combined_text, axis=1).apply(obfuscate_text)
        base["text"] = base.apply(combined_text, axis=1)
        mal_out = pd.concat([base, rep], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        mal_out = mal_out.iloc[:target_mal]

    # --- Benign ---
    if ben_count >= target_ben:
        ben_out = df_ben.sample(n=target_ben, random_state=42).copy()
        ben_out["text"] = ben_out.apply(combined_text, axis=1)
    else:
        needed = target_ben - ben_count
        base = df_ben.copy()
        if ben_count == 0:
            raise ValueError("No Benign rows available. Provide at least one 'Benign' example.")
        rep = base.sample(n=needed, replace=True, random_state=42).copy()
        base["text"] = base.apply(combined_text, axis=1)
        rep["text"]  = rep.apply(combined_text, axis=1)
        ben_out = pd.concat([base, rep], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        ben_out = ben_out.iloc[:target_ben]

    # Set string labels explicitly
    mal_out = mal_out.copy(); mal_out["label"] = "Malicious"
    ben_out = ben_out.copy(); ben_out["label"] = "Benign"

    # Combine, shuffle
    out = pd.concat([mal_out, ben_out], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # ensure urls numeric (0/1) if present but empty
    def has_url(t):
        return 1 if re.search(r"(?i)\b(hxxps?|https?|ftp)://\S+|\bwww\.\S+|\b[a-z0-9.-]+\.[a-z]{2,}\S*", str(t or "")) else 0
    if "urls" in out.columns:
        # If existing values look empty/non-numeric, recompute from body
        if not pd.to_numeric(out["urls"], errors="coerce").notna().all():
            out["urls"] = out["body"].apply(has_url)
    else:
        out["urls"] = out["body"].apply(has_url)

    return out

def main():
    print("Loading:", INPUT_CSV)
    if not Path(INPUT_CSV).exists():
        raise SystemExit(f"ERROR: Input file not found at {INPUT_CSV}")

    df = load_csv(INPUT_CSV)
    df = ensure_columns(df)

    # Normalize labels to 'Malicious'/'Benign' and drop unknowns
    df["label_text"] = df["label"].apply(normalize_label_to_text)
    unknowns = df["label_text"].isna().sum()
    if unknowns:
        print(f"Warning: {unknowns} rows have unknown labels and will be excluded.")
    df = df[df["label_text"].notna()].copy()
    df["label"] = df["label_text"]; df.drop(columns=["label_text"], inplace=True)

    # Pools
    df_mal = df[df["label"] == "Malicious"].copy()
    df_ben = df[df["label"] == "Benign"].copy()
    print("Available — Malicious:", len(df_mal), "Benign:", len(df_ben))

    # Build balanced 1500 dataset
    balanced = sample_balanced_text(df_mal, df_ben, PHISH_TARGET, BENIGN_TARGET)
    print("Balanced shape:", balanced.shape)
    print(balanced["label"].value_counts())

    # Save single CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "balanced_1500.csv")
    # Keep original columns plus 'text'
    cols = ["sender","receiver","date","subject","body","label","urls","text"]
    # make sure all exist
    for c in cols:
        if c not in balanced.columns:
            balanced[c] = "" if c != "urls" else 0
    balanced[cols].to_csv(out_path, index=False)
    print("Saved:", out_path)

    # Peek
    print("\nSample rows:\n", balanced.head(5)[["sender","subject","label","urls"]].to_string(index=False))

if __name__ == "__main__":
    main()
