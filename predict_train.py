#!/usr/bin/env python3
"""
Fine-tune DistilBERT for phishing vs benign on CSV with columns:
sender,receiver,date,subject,body,label,urls,text

- Uses an 80/20 train/val split from the balanced_1500.csv
- Evaluates on an optional external test set (same schema)
- Saves the best model to ./models/distilbert-phish-v1
"""

import os, re, json, numpy as np, pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, auc
)
import evaluate
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
)

# --------- CONFIG ---------
TRAIN_CSV = "prepared_dataset/balanced_1500.csv"
TEST_CSV  = "prepared_dataset/test_aligned.csv"   # optional; set to None if you don't have it
MODEL_NAME = "distilbert-base-uncased"            # small & fast; you can switch to "bert-base-uncased"
MAX_LEN = 256
OUTPUT_DIR = "models/distilbert-phish-v1"
EPOCHS = 3
LR = 2e-5
TRAIN_BS = 16
EVAL_BS = 32
SEED = 42
# --------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)

label2id = {"Benign": 0, "Malicious": 1}
id2label = {v: k for k, v in label2id.items()}

def load_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # sanity checks
    need = {"text","label"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must have columns {need}")
    # normalize labels
    df["label"] = df["label"].map(lambda x: "Malicious" if str(x).strip().lower() in
                                  {"malicious","phish","phishing","spam","1","true"} else
                                  ("Benign" if str(x).strip().lower() in
                                   {"benign","ham","0","false","legit","legitimate"} else x))
    if set(df["label"].unique()) - set(label2id.keys()):
        raise ValueError(f"Labels must be in {set(label2id.keys())}. Got: {set(df['label'].unique())}")
    return df

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

def compute_metrics_builder():
    metric_acc = evaluate.load("accuracy")
    metric_f1  = evaluate.load("f1")
    metric_prec = evaluate.load("precision")
    metric_rec  = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = probs.argmax(axis=1)

        out = {}
        out.update(metric_acc.compute(predictions=preds, references=labels))
        # macro F1 helpful; also focus on Malicious (positive class=1)
        out.update(metric_f1.compute(predictions=preds, references=labels, average="macro"))
        out["f1_pos"] = evaluate.load("f1").compute(
            predictions=preds, references=labels, average="binary", pos_label=1
        )["f1"]
        out["precision_pos"] = metric_prec.compute(predictions=preds, references=labels, average="binary", pos_label=1)["precision"]
        out["recall_pos"] = metric_rec.compute(predictions=preds, references=labels, average="binary", pos_label=1)["recall"]

        # AUPRC (for the positive class)
        from sklearn.metrics import precision_recall_curve, auc
        prec, rec, _ = precision_recall_curve(labels, probs[:,1])
        out["auprc"] = auc(rec, prec)
        return out

    return compute_metrics

def main():
    # 1) Load training CSV (balanced 1500)
    df = load_csv(TRAIN_CSV)
    df["y"] = df["label"].map(label2id)

    # Stratified 80/20 split for internal validation
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["y"]
    )

    # 2) Build HF Datasets
    train_ds = Dataset.from_pandas(train_df[["text","y"]].rename(columns={"y":"labels"}))
    val_ds   = Dataset.from_pandas(val_df[["text","y"]].rename(columns={"y":"labels"}))
    ds = DatasetDict(train=train_ds, validation=val_ds)

    # 3) Tokenizer & tokenized datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenized = ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
    )

    # 5) Training setup
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = compute_metrics_builder()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_pos",  # focus on Malicious F1
        greater_is_better=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        seed=SEED,
        report_to="none",
        fp16=torch.cuda.is_available(),    # mixed precision on GPU
        gradient_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 6) Train
    trainer.train()

    # 7) Evaluate on val
    eval_metrics = trainer.evaluate()
    print("\nValidation metrics:", json.dumps(eval_metrics, indent=2))

    # 8) Save model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    # 9) Optional: evaluate on external test set
    if TEST_CSV and os.path.exists(TEST_CSV):
        test_df = load_csv(TEST_CSV)
        test_df["y"] = test_df["label"].map(label2id)
        test_ds = Dataset.from_pandas(test_df[["text","y"]].rename(columns={"y":"labels"}))
        test_tok = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])

        # raw predictions
        preds = trainer.predict(test_tok)
        logits = preds.predictions
        y_true = np.array(test_df["y"].tolist())
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:,1]
        y_pred = probs >= 0.5

        print("\n=== TEST REPORT (threshold=0.5) ===")
        print(classification_report(y_true, y_pred, target_names=["Benign","Malicious"], digits=3))
        print("Confusion matrix [rows=true, cols=pred]:")
        print(confusion_matrix(y_true, y_pred))

        prec, rec, _ = precision_recall_curve(y_true, probs)
        print("AUPRC:", auc(rec, prec))

        # Optional: choose a better threshold (maximize F1 on val metrics is already done via training;
        # here we could pick threshold by maximizing F1 on val and apply to test)
    else:
        print("\nNo external TEST_CSV provided; training complete.")

if __name__ == "__main__":
    main()
