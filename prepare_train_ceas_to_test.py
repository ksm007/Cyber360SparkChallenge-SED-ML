#!/usr/bin/env python3
import argparse
import pandas as pd
import re

def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = re.sub(r"hxxps?", "http", s, flags=re.IGNORECASE)
    s = s.replace("[.]", ".").replace("(.)", ".")
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_label(val) -> str:
    if pd.isna(val):
        raise ValueError("Found NaN label.")
    s = str(val).strip().lower()
    if s in {"1", "malicious", "phish", "phishing", "spam", "true", "yes"}:
        return "Malicious"
    if s in {"0", "benign", "ham", "false", "no"}:
        return "Benign"
    # try numeric cast
    try:
        iv = int(float(s))
        return "Malicious" if iv == 1 else "Benign"
    except Exception:
        raise ValueError(f"Unrecognized label value: {val!r}")

def assemble_text(row: pd.Series) -> str:
    subject = basic_clean(row.get("subject", ""))
    body = basic_clean(row.get("body", ""))
    sender = str(row.get("sender", "") or "")
    receiver = str(row.get("receiver", "") or "")
    date = str(row.get("date", "") or "")
    urls = row.get("urls", "")
    urls_str = f" [URLS:{urls}]" if pd.notna(urls) and str(urls).strip() != "" else ""

    combined = (
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"From: {sender}\nTo: {receiver}\nDate: {date}{urls_str}"
    ).strip()
    return combined

def main():
    ap = argparse.ArgumentParser(description="Convert wide phishing CSV to email_text,label")
    ap.add_argument("--in_csv", required=True, help="Input CSV: sender,receiver,date,subject,body,label,urls")
    ap.add_argument("--out_csv", required=True, help="Output CSV: email_text,label")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # sanity checks
    need = {"subject", "body", "label"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Input must contain at least {need}, got {set(df.columns)}")

    # build unified text + label
    out = pd.DataFrame()
    out["email_text"] = df.apply(assemble_text, axis=1)
    out["label"] = df["label"].apply(normalize_label)

    # drop empty rows if any
    out = out[out["email_text"].str.strip() != ""].reset_index(drop=True)

    out.to_csv(args.out_csv, index=False)
    print(f"✅ Wrote {len(out)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
