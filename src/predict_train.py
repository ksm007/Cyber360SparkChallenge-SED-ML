#!/usr/bin/env python3
# train_mac_compat.py — DistilBERT phishing fine-tune with transformers-version compatibility (macOS-friendly)

import os, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)
import evaluate

# ---------------- CONFIG ----------------
TRAIN_CSV = "data/prepared_dataset/balanced_1500.csv"
TEST_CSV  = "data/prepared_dataset/test_aligned.csv"   # set to None if not using a test set
MODEL_NAME = "distilbert-base-uncased"            # or "bert-base-uncased"
MAX_LEN = 256
OUTPUT_DIR = "models/distilbert-phish-mac"
EPOCHS = 3
LR = 2e-5
TRAIN_BS = 16
EVAL_BS = 32
SEED = 42
# ----------------------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)

label2id = {"Benign":0, "Malicious":1}
id2label = {v:k for k,v in label2id.items()}

def load_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"text","label"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must have columns {need}. Got {df.columns.tolist()}")
    def norm(lbl):
        s = str(lbl).strip().lower()
        if s in {"malicious","phish","phishing","spam","1","true"}: return "Malicious"
        if s in {"benign","ham","0","false","legit","legitimate"}: return "Benign"
        return lbl
    df["label"] = df["label"].map(norm)
    bad = set(df["label"].unique()) - set(label2id.keys())
    if bad:
        raise ValueError(f"Unexpected labels: {bad}. Expected {set(label2id.keys())}")
    return df

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

def compute_metrics_builder():
    metric_acc = evaluate.load("accuracy")
    metric_f1  = evaluate.load("f1")
    metric_prec= evaluate.load("precision")
    metric_rec = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = probs.argmax(axis=1)

        out = {}
        out.update(metric_acc.compute(predictions=preds, references=labels))
        out.update(metric_f1.compute(predictions=preds, references=labels, average="macro"))
        out["f1_pos"] = metric_f1.compute(predictions=preds, references=labels, average="binary", pos_label=1)["f1"]
        out["precision_pos"] = metric_prec.compute(predictions=preds, references=labels, average="binary", pos_label=1)["precision"]
        out["recall_pos"] = metric_rec.compute(predictions=preds, references=labels, average="binary", pos_label=1)["recall"]

        p, r, _ = precision_recall_curve(labels, probs[:,1])
        out["auprc"] = auc(r, p)
        return out
    return compute_metrics

def build_training_args():
    """
    Try modern TrainingArguments; if TypeError (older transformers), fall back to minimal args.
    """
    try:
        # Modern API
        return TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_pos",
            greater_is_better=True,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=TRAIN_BS,
            per_device_eval_batch_size=EVAL_BS,
            learning_rate=LR,
            weight_decay=0.01,
            logging_steps=50,
            seed=SEED,
            report_to=None,   # safe across versions
            fp16=False,       # macOS MPS/CPU -> keep False
            gradient_accumulation_steps=1,
        )
    except TypeError:
        # Older API fallback (no eval/save strategies, no best model, no early stopping)
        print("[Compat] Falling back to older TrainingArguments (no evaluation_strategy/save_strategy).")
        return TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=TRAIN_BS,
            per_device_eval_batch_size=EVAL_BS,
            learning_rate=LR,
            weight_decay=0.01,
            logging_steps=50,
            seed=SEED,
            fp16=False,
            # Older versions may not accept report_to at all; omit.
        )

def main():
    print("torch:", torch.__version__, "| MPS available:", torch.backends.mps.is_available())

    # 1) Load data
    df = load_csv(TRAIN_CSV)
    df["y"] = df["label"].map(label2id)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["y"])

    # 2) Datasets
    train_ds = Dataset.from_pandas(train_df[["text","y"]].rename(columns={"y":"labels"}))
    val_ds   = Dataset.from_pandas(val_df[["text","y"]].rename(columns={"y":"labels"}))
    ds = DatasetDict(train=train_ds, validation=val_ds)

    # 3) Tokenizer & tokenized
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenized = ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
    )

    # 5) Training setup
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = compute_metrics_builder()
    args = build_training_args()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6) Train (older transformers will still train fine; just no per-epoch autosave/best-model)
    trainer.train()

    # 7) Eval on val
    eval_metrics = trainer.evaluate()
    print("\nValidation metrics:", json.dumps(eval_metrics, indent=2))

    # 8) Save model + tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    # 9) Optional test
    if TEST_CSV and os.path.exists(TEST_CSV):
        test_df = load_csv(TEST_CSV)
        test_df["y"] = test_df["label"].map(label2id)
        test_ds = Dataset.from_pandas(test_df[["text","y"]].rename(columns={"y":"labels"}))
        test_tok = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])

        preds = trainer.predict(test_tok)
        logits = preds.predictions
        y_true = test_df["y"].to_numpy()
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:,1]
        y_pred = (probs >= 0.5).astype(int)

        print("\n=== TEST REPORT (thr=0.5) ===")
        print(classification_report(y_true, y_pred, target_names=["Benign","Malicious"], digits=3))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
        p, r, _ = precision_recall_curve(y_true, probs)
        print("AUPRC:", auc(r, p))
    else:
        print("\nNo TEST_CSV provided; training finished.")

if __name__ == "__main__":
    main()
