import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import os

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(42)
if device == torch.device("cuda"):
    torch.cuda.manual_seed_all(42)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load data
with open("data/positive_examples.txt", 'r') as f:
    positive_text = f.readlines()

with open("data/negative_examples.txt", 'r') as f:
    negative_text = f.readlines()

with open("data/neutral_examples.txt", 'r') as f:  # Load the neutral examples
    neutral_text = f.readlines()

# Tokenize and encode the text
# Tokenize and encode the text
max_seq_length = 256
positive_encodings = tokenizer(positive_text, truncation=True, padding='max_length', max_length=max_seq_length)
negative_encodings = tokenizer(negative_text, truncation=True, padding='max_length', max_length=max_seq_length)
neutral_encodings = tokenizer(neutral_text, truncation=True, padding='max_length', max_length=max_seq_length)

# Labels: 0 for stressed, 1 for relaxed, 2 for neutral
positive_labels = torch.zeros(len(positive_text), dtype=torch.long)
negative_labels = torch.ones(len(negative_text), dtype=torch.long)
neutral_labels = torch.full((len(neutral_text),), 2, dtype=torch.long)  # Labels for neutral examples

# Combine inputs and labels
# Combine inputs and labels
input_ids = torch.cat([
    torch.tensor(positive_encodings['input_ids']), 
    torch.tensor(negative_encodings['input_ids']), 
    torch.tensor(neutral_encodings['input_ids'])
])
attention_masks = torch.cat([
    torch.tensor(positive_encodings['attention_mask']), 
    torch.tensor(negative_encodings['attention_mask']), 
    torch.tensor(neutral_encodings['attention_mask'])
])
labels = torch.cat([positive_labels, negative_labels, neutral_labels])

# Create DataLoader
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=16)

# Initialize BERT model for sequence classification with 3 classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
print("Starting training...")
for epoch in range(3):  # You can change the number of epochs
    model.train()
    total_loss = 0
    print(f"Epoch {epoch + 1} | Starting training...")
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # Print every 10 steps to show progress
        if step % 10 == 0:
            print(f"Epoch {epoch + 1} | Step {step} | Current Loss: {loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} | Average Training Loss: {avg_train_loss:.4f}")

    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    print(f"Epoch {epoch + 1} | Starting evaluation...")
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device))
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(true_labels, predictions)
    print(f"Epoch {epoch + 1} | Validation Accuracy: {acc:.4f}")


# Save model
if not os.path.exists('./models'):
    os.makedirs('./models')
model.save_pretrained('./models/exam_ease_bert_model_v2')  # Saving the model as v2
tokenizer.save_pretrained('./models/exam_ease_bert_model_v2')
