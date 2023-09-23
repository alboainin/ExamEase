import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = './models/exam_ease_bert_model'

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Ensure the model is in evaluation mode
model.eval()

# Decide on the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    # Tokenize input and get output from model
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=1).item()

    if predicted_class == 1:
        return "Positive sentiment detected."
    else:
        return "Negative sentiment detected."

if __name__ == "__main__":
    print("Welcome to ExamEase!")
    while True:
        text = input("Please enter your text or type 'exit' to quit: ")
        if text.lower() == 'exit':
            print("Thank you for using ExamEase!")
            break
        sentiment = predict_sentiment(text)
        print(sentiment)
