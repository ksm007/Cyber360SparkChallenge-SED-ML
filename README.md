# Cyber360SparkChallenge-SED-ML



ASU staff have reported a new wave of targeted phishing (social engineering) emails disguised as urgent university alerts. Create a student-centered awareness campaign to help peers spot and report phishing attempts.

Recommended: Fine-tune a small transformer (BERT/DeBERTa/DistilBERT-class) for social engineering + phishing vs. benign on emails or other commonly used text mediums. You may train on a public dataset you select (e.g., from Hugging Face), craft your own, or mix and match as you please.

Deliverable format:
- Model == AI solution: software, ML, etc…
            ▪ Trained model file (.bin, .gguf, .etc)
            ▪ URL Github with code used to train and evaluate the model (commented out)
            ▪ URL to Hugging Face & model card if you host the weights.(as needed, depending on model size)
- Plus: presentation deck explaining features, training choices, and pitfalls (obfuscation, hard negatives, calibration, etc...)



Track tools:
- Datasets with phishing data points
- BERT or DBERT (foundational ML models to build upon, accessible by PyTorch)
- A deck explaining features, training choices, and pitfalls (obfuscation, hard negatives, calibration).



These \*BERT\* type models are extremely tiny, they produce a label with a repeatable confidence score, and most importantly they can be run on antique hardware at very fast speeds (~50-400ms depending on CPU, 1-10ms on GPU). Because of this, they are also accessible to train (or fine-tune when building off a base) on CPU.


Here is an example of what inferencing these models will look like. This comes from a deployed huggingface model that you can potentially use as a base if you so choose:

```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cybersectony/phishing-email-detection-distilbert_v2.1")
model = AutoModelForSequenceClassification.from_pretrained("cybersectony/phishing-email-detection-distilbert_v2.1")

def predict_email(email_text):
    # Preprocess and tokenize
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get probabilities for each class
    probs = predictions[0].tolist()

    # Create labels dictionary
    labels = {
        "legitimate_email": probs[0],
        "phishing_url": probs[1],
        "legitimate_url": probs[2],
        "phishing_url_alt": probs[3]
    }

    # Determine the most likely classification
    max_label = max(labels.items(), key=lambda x: x[1])

    return {
        "prediction": max_label[0],
        "confidence": max_label[1],
        "all_probabilities": labels
    }

# Example usage
email = """
Dear User,
Your account security needs immediate attention. Please verify your credentials.
Click here: http://suspicious-link.com
"""

result = predict_email(email)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nAll probabilities:")
for label, prob in result['all_probabilities'].items():
    print(f"{label}: {prob:.2%}")
```
The above model and reference inference script comes from the following [FOSS source on HuggingFace](https://huggingface.co/cybersectony/phishing-email-detection-distilbert_v2.1)



