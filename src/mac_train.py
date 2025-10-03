# train_mac_phish.py  (patched for transformers>=4.46)
import os, sys, re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed

def clean_text(s):
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = re.sub(r"hxxps?", "http", s, flags=re.IGNORECASE)
    s = s.replace("[.]", ".").replace("(.)", ".")
    return re.sub(r"\s+", " ", s).strip()

def norm_label(v):
    s = str(v).strip().lower()
    if s in {"malicious","phish","phishing","spam","1","true","yes"}: return 1
    if s in {"benign","ham","0","false","no"}: return 0
    try: return 1 if int(float(s))==1 else 0
    except: raise ValueError(f"Unrecognized label: {v!r}")

def load_unified(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"email_text","label"}.issubset(df.columns):
        raise SystemExit("CSV must have columns: email_text,label")
    df["text"] = df["email_text"].map(clean_text)
    df["label"] = df["label"].map(norm_label).astype(int)
    df = df[df["text"].str.strip()!=""].reset_index(drop=True)
    return df[["text","label"]]

class TextDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels): self.enc, self.labels = enc, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k,v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item

def metrics_fn(pred):
    logits, labels = pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_mac_phish.py unified.csv [out_dir] [hf_model]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_dir  = sys.argv[2] if len(sys.argv)>2 else "phish-bert"
    hf_model = sys.argv[3] if len(sys.argv)>3 else "distilbert-base-uncased"

    os.makedirs(out_dir, exist_ok=True)
    set_seed(42)

    print("PyTorch:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    df = load_unified(csv_path)
    Xtr, Xte, ytr, yte = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, stratify=df["label"].tolist(), random_state=42
    )

    tok = AutoTokenizer.from_pretrained(hf_model)
    tr_enc = tok(Xtr, truncation=True, padding=True, max_length=512)
    te_enc = tok(Xte, truncation=True, padding=True, max_length=512)
    tr_ds, te_ds = TextDS(tr_enc, ytr), TextDS(te_enc, yte)

    model = AutoModelForSequenceClassification.from_pretrained(hf_model, num_labels=2)

    # NOTE: evaluation_strategy -> eval_strategy in recent Transformers
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        eval_strategy="epoch",      # <— this is the new name
        save_strategy="no",
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tr_ds,
        eval_dataset=te_ds,
        tokenizer=tok,
        compute_metrics=metrics_fn,
    )

    trainer.train()

    print("\nEval metrics:")
    m = trainer.evaluate(te_ds)
    for k, v in m.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    preds = trainer.predict(te_ds).predictions.argmax(-1)
    print("\nClassification report:")
    print(classification_report(yte, preds, target_names=["Benign(0)","Malicious(1)"], digits=4))

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print("\nSaved model + tokenizer to:", out_dir)

if __name__ == "__main__":
    main()
