#!/usr/bin/env python3
"""
convert_emails.py

Convert a CSV with columns: email_text,label
to: sender,receiver,date,subject,body,label,urls

Usage:
  python convert_emails.py --in test_emails.csv --out test_emails_structured.csv
"""

import argparse
import csv
import re
import pandas as pd
from typing import Tuple, Dict, List

# Detect URLs including obfuscated ones (hxxp/hxxps), normal http(s), ftp, www, or bare domains
URL_PATTERN = re.compile(
    r"(?i)\b(?:hxxps?|https?|ftp)://\S+|\bwww\.\S+|\b[a-z0-9.-]+\.[a-z]{2,}\S*"
)

def unfold_headers(lines: List[str]) -> List[str]:
    """
    Handle header line folding (continuation lines that start with whitespace).
    Returns a list of unfolded 'Key: value' lines.
    """
    unfolded = []
    for line in lines:
        if not unfolded:
            unfolded.append(line)
            continue
        if line.startswith((" ", "\t")):
            unfolded[-1] = unfolded[-1].rstrip("\r\n") + " " + line.strip()
        else:
            unfolded.append(line)
    return unfolded

def parse_email_text(raw: str) -> Tuple[Dict[str, str], str]:
    """
    Parse raw email text into headers dict and body string.
    We look for a blank line separating headers from body.
    """
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    parts = raw.split("\n\n", 1)
    header_block = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    header_lines = [l for l in header_block.split("\n") if l.strip() != ""]
    header_lines = unfold_headers(header_lines)

    headers = {}
    for line in header_lines:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return headers, body.strip()

def has_url(text: str) -> int:
    return 1 if URL_PATTERN.search(text or "") else 0

def convert_row(email_text: str, label: str) -> Dict[str, str]:
    headers, body = parse_email_text(email_text or "")

    subject = headers.get("subject", "")
    sender  = headers.get("from", "")
    to      = headers.get("to", "")
    # date intentionally left blank per your request (keep column but no value)
    date    = ""

    # If subject/body were embedded only in body (no headers), try a light fallback:
    if not subject and email_text:
        # Try to find a "Subject:" line anywhere
        m = re.search(r"(?im)^subject:\s*(.+)$", email_text)
        if m:
            subject = m.group(1).strip()

    urls_flag = has_url(body)

    # Keep label exactly as provided (e.g., 'Malicious'/'Benign'); change here if you want 1/0.
    return {
        "sender": sender,
        "receiver": to,
        "date": date,
        "subject": subject,
        "body": body,
        "label": label,
        "urls": urls_flag,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  required=True, help="Input CSV (email_text,label)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV (sender,receiver,date,subject,body,label,urls)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path, dtype=str, keep_default_na=False)
    # Basic validation
    if "email_text" not in df.columns or "label" not in df.columns:
        raise SystemExit("Input CSV must have columns: email_text,label")

    rows = []
    for _, r in df.iterrows():
        rows.append(convert_row(r.get("email_text", ""), r.get("label", "")))

    out_df = pd.DataFrame(rows, columns=["sender","receiver","date","subject","body","label","urls"])
    # Ensure CSV quoted properly
    out_df.to_csv(args.out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Converted {len(out_df)} rows → {args.out_path}")

if __name__ == "__main__":
    main()
