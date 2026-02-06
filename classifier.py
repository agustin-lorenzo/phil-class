import torch
import pandas as pd
from datasets import Dataset
from transformers import EarlyStoppingCallback
from sklearn import preprocessing
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from huggingface_hub import notebook_login
import numpy as np
import matplotlib.pyplot as plt
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("data/data.csv")

# Encode labels
label_encoder = preprocessing.LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['labels'])

# Load model and tokenizer
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                            num_labels=4,
                                                            id2label=id2label,
                                                            label2id=label2id)
model.to(device)

# Create train/test splits
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
def tokenize(examples):
    return tokenizer(examples['text'], max_length=512, truncation=True)
dataset = dataset.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create F1 metric for evaluation
metric = evaluate.load('f1')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

# Train model
training_args = TrainingArguments(
    output_dir='checkpoints',
    eval_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=20,
    learning_rate=2e-5,
    optim='stable_adamw',
    metric_for_best_model='f1',
    greater_is_better=True,
    load_best_model_at_end=True,
    weight_decay=0.15,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
trainer.save_model('model')
tokenizer.save_pretrained('model')

# Analyze results with confusion matrix
predictions = trainer.predict(dataset['test'])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

label_names = ["Existentialism", "Nihilism", "Stoicism", "Utilitarianism"]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot().figure_.savefig('confusion_matrix.png')
