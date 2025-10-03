# # #!/usr/bin/env python3
# # """
# # convert_emails.py

# # Convert a CSV with columns: email_text,label
# # to: sender,receiver,date,subject,body,label,urls

# # Usage:
# #   python convert_emails.py --in test_emails.csv --out test_emails_structured.csv
# # """

# # import argparse
# # import csv
# # import re
# # import pandas as pd
# # from typing import Tuple, Dict, List

# # # Detect URLs including obfuscated ones (hxxp/hxxps), normal http(s), ftp, www, or bare domains
# # URL_PATTERN = re.compile(
# #     r"(?i)\b(?:hxxps?|https?|ftp)://\S+|\bwww\.\S+|\b[a-z0-9.-]+\.[a-z]{2,}\S*"
# # )

# # def unfold_headers(lines: List[str]) -> List[str]:
# #     """
# #     Handle header line folding (continuation lines that start with whitespace).
# #     Returns a list of unfolded 'Key: value' lines.
# #     """
# #     unfolded = []
# #     for line in lines:
# #         if not unfolded:
# #             unfolded.append(line)
# #             continue
# #         if line.startswith((" ", "\t")):
# #             unfolded[-1] = unfolded[-1].rstrip("\r\n") + " " + line.strip()
# #         else:
# #             unfolded.append(line)
# #     return unfolded

# # def parse_email_text(raw: str) -> Tuple[Dict[str, str], str]:
# #     """
# #     Parse raw email text into headers dict and body string.
# #     We look for a blank line separating headers from body.
# #     """
# #     raw = raw.replace("\r\n", "\n").replace("\r", "\n")
# #     parts = raw.split("\n\n", 1)
# #     header_block = parts[0]
# #     body = parts[1] if len(parts) > 1 else ""

# #     header_lines = [l for l in header_block.split("\n") if l.strip() != ""]
# #     header_lines = unfold_headers(header_lines)

# #     headers = {}
# #     for line in header_lines:
# #         if ":" in line:
# #             k, v = line.split(":", 1)
# #             headers[k.strip().lower()] = v.strip()
# #     return headers, body.strip()

# # def has_url(text: str) -> int:
# #     return 1 if URL_PATTERN.search(text or "") else 0

# # def convert_row(email_text: str, label: str) -> Dict[str, str]:
# #     headers, body = parse_email_text(email_text or "")

# #     subject = headers.get("subject", "")
# #     sender  = headers.get("from", "")
# #     to      = headers.get("to", "")
# #     # date intentionally left blank per your request (keep column but no value)
# #     date    = ""

# #     # If subject/body were embedded only in body (no headers), try a light fallback:
# #     if not subject and email_text:
# #         # Try to find a "Subject:" line anywhere
# #         m = re.search(r"(?im)^subject:\s*(.+)$", email_text)
# #         if m:
# #             subject = m.group(1).strip()

# #     urls_flag = has_url(body)

# #     # Keep label exactly as provided (e.g., 'Malicious'/'Benign'); change here if you want 1/0.
# #     return {
# #         "sender": sender,
# #         "receiver": to,
# #         "date": date,
# #         "subject": subject,
# #         "body": body,
# #         "label": label,
# #         "urls": urls_flag,
# #     }

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--in",  dest="in_path",  required=True, help="Input CSV (email_text,label)")
# #     ap.add_argument("--out", dest="out_path", required=True, help="Output CSV (sender,receiver,date,subject,body,label,urls)")
# #     args = ap.parse_args()

# #     df = pd.read_csv(args.in_path, dtype=str, keep_default_na=False)
# #     # Basic validation
# #     if "email_text" not in df.columns or "label" not in df.columns:
# #         raise SystemExit("Input CSV must have columns: email_text,label")

# #     rows = []
# #     for _, r in df.iterrows():
# #         rows.append(convert_row(r.get("email_text", ""), r.get("label", "")))

# #     out_df = pd.DataFrame(rows, columns=["sender","receiver","date","subject","body","label","urls"])
# #     # Ensure CSV quoted properly
# #     out_df.to_csv(args.out_path, index=False, quoting=csv.QUOTE_MINIMAL)
# #     print(f"Converted {len(out_df)} rows → {args.out_path}")

# # if __name__ == "__main__":
# #     main()



#!/usr/bin/env python3
# normalize_for_training.py
import os, re, csv
from pathlib import Path
import pandas as pd

INPUT_CSV  = "../data/testing/final_test.csv"            # <-- set to your file
OUTPUT_DIR = "../prepared_dataset"
OUTPUT_CSV = "../data/testing/test_normalized.csv"

# --- helpers ---
ZW_RE = re.compile(r"[\u200B-\u200D\uFEFF]")  # zero-width chars
WS_RE = re.compile(r"[ \t]+")
BLANKS_RE = re.compile(r"\n{3,}")            # 3+ blank lines -> 2
URL_RE = re.compile(r"(?i)\b(hxxps?|https?|ftp)://\S+|\bwww\.\S+|\b[a-z0-9.-]+\.[a-z]{2,}\S*")

def clean_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = ZW_RE.sub("", s)                          # remove zero-width
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')  # unify punctuation
    s = WS_RE.sub(" ", s)                         # collapse inline spaces/tabs
    s = BLANKS_RE.sub("\n\n", s)                  # collapse excessive blank lines
    return s.strip()

def deobfuscate_for_eval(s: str) -> str:
    # keep original text for 'text'; build normalized 'text_norm' for consistent eval if desired
    t = s
    t = re.sub(r"(?i)\bhxxps\b", "https", t)
    t = re.sub(r"(?i)\bhxxp\b", "http",  t)
    t = t.replace("micros0ft", "microsoft")
    return t

def has_url(s: str) -> int:
    return 1 if URL_RE.search(s or "") else 0

def normalize_label(lbl: str) -> str:
    x = str(lbl).strip().lower()
    if x in {"malicious","phish","phishing","spam","1","true"}:
        return "Malicious"
    if x in {"benign","ham","0","false","legit","legitimate"}:
        return "Benign"
    # default: keep as-is if already good
    if lbl in {"Malicious","Benign"}:
        return lbl
    raise ValueError(f"Unknown label value: {lbl}")

def main():
    in_path = Path(INPUT_CSV)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)

    # Ensure required columns exist
    for c in ["sender","receiver","date","subject","body","label","urls"]:
        if c not in df.columns:
            df[c] = ""

    # Normalize columns
    df["sender"]   = df["sender"].apply(clean_text)
    df["receiver"] = df["receiver"].apply(clean_text)
    df["date"]     = df["date"].apply(lambda s: s.strip())  # you said date not needed; we leave as-is/blank
    df["subject"]  = df["subject"].apply(clean_text)
    df["body"]     = df["body"].apply(clean_text)
    df["label"]    = df["label"].apply(normalize_label)

    # Recompute urls if needed (or keep provided if valid)
    if not pd.to_numeric(df["urls"], errors="coerce").notna().all():
        df["urls"] = df["body"].apply(has_url)
    else:
        # fix any obvious inconsistencies
        recalc = df["body"].apply(has_url)
        df.loc[recalc != pd.to_numeric(df["urls"], errors="coerce").fillna(0).astype(int), "urls"] = recalc

    # Build model-input fields
    df["text"]      = df.apply(lambda r: (r["subject"] + "\n\n" + r["body"]).strip(), axis=1)
    df["text_norm"] = df["text"].apply(deobfuscate_for_eval)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / OUTPUT_CSV
    cols = ["sender","receiver","date","subject","body","label","urls","text","text_norm"]
    df[cols].to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved normalized test file → {out_path.resolve()}")

if __name__ == "__main__":
    main()

# import pandas as pd

# # Load your test CSV that has text_norm
# df = pd.read_csv("prepared_dataset/test_normalized.csv")

# # Drop text_norm and keep only text
# if "text_norm" in df.columns:
#     df = df.drop(columns=["text_norm"])

# # Save aligned test file
# df.to_csv("prepared_dataset/test_aligned.csv", index=False)
# print("Saved prepared_dataset/test_aligned.csv with schema:", df.columns.tolist())