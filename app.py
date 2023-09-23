from flask import Flask, render_template, request, redirect, url_for
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load BERT model
MODEL_PATH = "./models/exam_ease_bert_model_v2"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/results", methods=["POST"])
def results():
    text = request.form["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        sentiment = "Positive" if preds[0] == 0 else "Negative" if preds[0] == 1 else "Neutral"

    return render_template("results.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
