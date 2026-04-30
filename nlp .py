!pip install -q transformers datasets sentencepiece accelerate scikit-learn evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import evaluate
# Loading the SNIPS dataset (you can swap 'snips_built_in_intents' for your preferred dataset)
dataset = load_dataset("snips_built_in_intents")

print("Dataset Dictionary Structure:")
print(dataset)

# Inspect a single example
print("\nSample Training Example:")
print(dataset['train'][0])

# Extract unique intents to understand our classification space
num_labels = len(dataset['train'].features['label'].names)
label_names = dataset['train'].features['label'].names
print(f"\nNumber of Intent Classes: {num_labels}")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Padding and truncation ensure all sequences are the same length
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

# Apply tokenization to the entire dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Format the dataset to return PyTorch tensors
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")
train_test_split_dataset = tokenized_datasets['train'].train_test_split(test_size=0.2)

# Rename the splits for clarity
train_dataset = train_test_split_dataset['train']
eval_dataset = train_test_split_dataset['test']

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(eval_dataset)}")
# Extract raw text and labels for the baseline
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
test_texts = dataset['train']['text'][:500] # Using a subset for quick testing
test_labels = dataset['train']['label'][:500]

# Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, train_labels)

# Evaluate baseline
nb_predictions = nb_model.predict(X_test)
print("Baseline Naive Bayes Accuracy:", accuracy_score(test_labels, nb_predictions))
# Load pre-trained BERT with the correct number of labels
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./intent_results",
    eval_strategy="epoch",  # <--- This is the fixed line!
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Metric function for the Trainer
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],
    compute_metrics=compute_metrics,
)

# Start fine-tuning
trainer.train()
# Get predictions on the evaluation set
predictions_output = trainer.predict(tokenized_datasets["train"]) # Use test set in practice
predictions = np.argmax(predictions_output.predictions, axis=1)
true_labels = predictions_output.label_ids

# Print a detailed classification report (Precision, Recall, F1-Score)
print(classification_report(true_labels, predictions, target_names=label_names))
from transformers import pipeline

# Create a classification pipeline using our fine-tuned model and tokenizer
intent_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Test some custom sentences
test_sentences = [
    "Add this track to my workout playlist.",
    "What is the weather going to be like tomorrow in Tokyo?",
    "Book a table for two at a Thai restaurant."
]

for sentence in test_sentences:
    result = intent_classifier(sentence)[0]
    # Map the predicted label ID back to the readable intent name
    label_id = int(result['label'].split('_')[-1]) if 'LABEL' in result['label'] else int(result['label'])
    intent_name = label_names[label_id] if 'LABEL' in result['label'] else result['label']

    print(f"User: '{sentence}'")
    print(f"Predicted Intent: {intent_name} (Confidence: {result['score']:.4f})\n")
import gradio as gr
from transformers import pipeline
import torch

# 1. Force the pipeline to rebuild on the correct hardware (GPU if available)
device = 0 if torch.cuda.is_available() else -1
intent_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

# 2. Define the prediction function with a safety net
def predict_intent(user_text):
    try:
        # Ignore empty submissions
        if not user_text.strip():
            return "Please type a message first."

        # Get prediction
        result = intent_classifier(user_text)[0]

        # Safely map the label back to the readable name
        if 'LABEL' in result['label']:
            label_id = int(result['label'].split('_')[-1])
            intent_name = label_names[label_id]
        else:
            try:
                label_id = int(result['label'])
                intent_name = label_names[label_id]
            except ValueError:
                intent_name = result['label'] # If it's already a text string

        return f"🎯 {intent_name}\n📊 Confidence: {result['score']:.2%}"

    except Exception as e:
        # If it breaks, tell us EXACTLY why on the screen
        return f"⚠️ Error: {str(e)}"

# 3. Build and launch the UI with debug mode ON
demo = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(lines=2, placeholder="Type a command here... e.g., 'Book a flight to Paris'"),
    outputs=gr.Textbox(label="Model Prediction"),
    title="🤖 Chatbot Intent Detection Prototype",
    height=500, # Increased height
    width=800   # Increased width
)

# debug=True forces Colab to print the exact error message in the cell output if it crashes again!
demo.launch(share=True, debug=True)
