# train_phishing_bert.py
import argparse
import os
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------
# Helpers
# ---------------------------
PHISH_POSITIVE_STRINGS = {"malicious", "phish", "phishing", "spam", "1", "true", "yes"}

def normalize_label(val):
    """Map label to 0 (Benign) / 1 (Malicious)."""
    if pd.isna(val):
        raise ValueError("Found NaN label.")
    s = str(val).strip().lower()
    if s in {"benign", "ham", "0", "false", "no"}:
        return 0
    if s in PHISH_POSITIVE_STRINGS:
        return 1
    # Fallback: try int cast (e.g., 0/1)
    try:
        iv = int(float(s))
        if iv in (0, 1):
            return iv
    except Exception:
        pass
    raise ValueError(f"Unrecognized label value: {val!r}")

def basic_email_text_clean(text: str) -> str:
    """Light normalization often seen in phishing datasets."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    t = text
    # common obfuscations
    t = re.sub(r"hxxps?", "http", t, flags=re.IGNORECASE)
    t = t.replace("[.]", ".").replace("(.)", ".")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def autodetect_text_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["email_text", "text", "content"]:
        if cand in df.columns:
            return cand
    return None

def build_text_from_wide_schema(row: pd.Series) -> str:
    subject = basic_email_text_clean(row.get("subject", ""))
    body = basic_email_text_clean(row.get("body", ""))
    sender = str(row.get("sender", "") or "")
    receiver = str(row.get("receiver", "") or "")
    date = str(row.get("date", "") or "")
    urls = row.get("urls", "")
    urls_str = f" [URLS:{urls}]" if pd.notna(urls) and str(urls).strip() != "" else ""
    combined = (
        f"Subject: {subject}\n\n{body}\n\n"
        f"From: {sender}\nTo: {receiver}\nDate: {date}{urls_str}"
    )
    return combined.strip()

def prepare_dataframe(csv_path: str, text_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Standardize column names (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Decide how to construct text
    if text_col is None:
        text_col = autodetect_text_column(df)

    if text_col and text_col in df.columns:
        df["text"] = df[text_col].astype(str).apply(basic_email_text_clean)
    else:
        # Expect wide schema: subject + body present
        required = {"subject", "body"}
        if not required.issubset(df.columns):
            raise ValueError(
                "Could not find a unified text column or subject/body columns. "
                "Provide --text_col or ensure columns include subject and body."
            )
        df["text"] = df.apply(build_text_from_wide_schema, axis=1)

    # Normalize labels
    df["label"] = df["label"].apply(normalize_label).astype(int)

    # Drop rows with empty text after cleaning
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows after cleaning.")
    return df[["text", "label"]]

# ---------------------------
# Dataset
# ---------------------------
class TextClsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# ---------------------------
# Weighted loss via custom Trainer (optional for imbalance)
# ---------------------------
@dataclass
class WeightedTrainer(Trainer):
    class_weights: Optional[torch.Tensor] = None
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    proba_pos = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, proba_pos) if len(np.unique(labels)) == 2 else np.nan,
        "auprc": average_precision_score(labels, proba_pos),
    }
    return metrics

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train/Test BERT for phishing email classification.")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV.")
    parser.add_argument("--text_col", default=None, help="Name of unified text column if present (e.g., email_text).")
    parser.add_argument("--model", default="bert-base-uncased", help="HF model checkpoint (e.g., distilbert-base-uncased).")
    parser.add_argument("--output_dir", default="./phish-bert", help="Where to save model & outputs.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_classes", action="store_true", help="Use class-weighted loss if data is imbalanced.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio.")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading & preparing data…")
    df = prepare_dataframe(args.csv, args.text_col)
    print(df.head())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=args.test_size,
        stratify=df["label"].tolist(),
        random_state=args.seed,
    )

    print("Tokenizing…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=args.max_len)
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=args.max_len)

    train_ds = TextClsDataset(train_enc, y_train)
    test_ds = TextClsDataset(test_enc, y_test)

    print("Loading model…")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    # Optional class weights
    class_weights_t = None
    if args.weight_classes:
        class_counts = np.bincount(y_train)
        # inverse frequency weights
        weights = len(y_train) / (2.0 * class_counts + 1e-6)
        class_weights_t = torch.tensor(weights, dtype=torch.float32)
        print(f"Using class weights: {weights}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
    )

    print("Training…")
    trainer_cls = WeightedTrainer if args.weight_classes else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        **({"class_weights": class_weights_t} if args.weight_classes else {}),
    )

    trainer.train()

    print("\nEvaluating on test set…")
    eval_metrics = trainer.evaluate(eval_dataset=test_ds)
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    # Detailed report
    preds_logits = trainer.predict(test_ds).predictions
    y_pred = preds_logits.argmax(axis=-1)
    y_score = torch.softmax(torch.tensor(preds_logits), dim=-1)[:, 1].numpy()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign(0)", "Malicious(1)"], digits=4))

    # Save model + tokenizer
    print(f"\nSaving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save predictions CSV (test set)
    out_df = pd.DataFrame({
        "text": X_test,
        "label": y_test,
        "pred_label": y_pred,
        "pred_score_malicious": y_score,
    })
    out_csv = os.path.join(args.output_dir, "test_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved test predictions to: {out_csv}")

    print("\nDone.")

if __name__ == "__main__":
    main()
