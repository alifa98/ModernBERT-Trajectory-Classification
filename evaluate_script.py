import numpy as np

from transformers import (
    PreTrainedTokenizerFast,
    ModernBertForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate

model_dir = "./modernbert-ho-traj-classifier"
model = ModernBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = PreTrainedTokenizerFast.from_pretrained("./ho-sequence-tokenizer")

dataset = load_dataset("csv", data_files="/local/data1/shared_data/higher_order_trajectory/rome/ho_rome_res8.csv")

splits = dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = splits["test"]

def filter_short_sequences(example):
    return len(example["higher_order_trajectory"].split()) > 10

dataset = dataset.filter(filter_short_sequences)

dataset = dataset.class_encode_column("taxi_id")

# 3) Tokenize the dataset
def tokenize_function(example):
    tokens = tokenizer(
        example["higher_order_trajectory"],
        truncation=True,
        padding=True,
        max_length=512
    )
    # 'taxi_id' is now guaranteed integer-labeled
    tokens["labels"] = example["taxi_id"]
    return tokens

dataset = dataset.map(tokenize_function)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """
    eval_pred is (logits, labels). We'll compute several metrics here:
    Accuracy, Precision, Recall, F1.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"]
    
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"]
    
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

results = trainer.evaluate(eval_dataset=dataset)
print("Evaluation Results:", results)
