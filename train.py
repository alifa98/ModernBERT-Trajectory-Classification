from transformers import PreTrainedTokenizerFast
from transformers import ModernBertConfig, ModernBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_dataset

# 1) Load your tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("./ho-sequence-tokenizer")

# 2) Load the dataset and filter
# dataset = load_dataset("csv", data_files="/local/data1/shared_data/higher_order_trajectory/rome/ho_rome_res8.csv")
dataset = load_dataset("csv", data_files="cut_rome_testi.csv")

def filter_short_sequences(example):
    return len(example["higher_order_trajectory"].split()) > 10

dataset = dataset.filter(filter_short_sequences)

# 3) Encode the 'taxi_id' column as class labels
dataset = dataset.class_encode_column("taxi_id")

# 4) Tokenize
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

# 5) Figure out how many unique classes
num_labels = len(set(dataset["train"]["taxi_id"]))

# 6) Build config and model
config = ModernBertConfig(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    hidden_size=128,
    num_hidden_layers=3,
    num_attention_heads=2,
    intermediate_size=256,
    max_position_embeddings=512,
    num_labels=num_labels
)

model = ModernBertForSequenceClassification(config)

# 7) Split data
dataset = dataset["train"].train_test_split(test_size=0.2)

# 8) Trainer args
training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=50,
    per_device_eval_batch_size=50,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=100
)

# 9) Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 10) Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("./modernbert-ho-traj-classifier")
