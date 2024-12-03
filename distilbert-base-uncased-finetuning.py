# %%
#%pip install transformers
#%pip install torch
#%pip install datasets

# %% [markdown]
# ## Readme
# Uncomment code below, to install requirements, then restart the kernel, comment out cell below and run all cells. 

# %%
#%pip install -r requirements.txt

# %%
import torch

device = "cpu"
# setup optimal acceleration device 
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Metal Performance Shaders on macOS
elif torch.cuda.is_available():
    device = torch.device("cuda")  # to check if cuda is an option https://www.restack.io/p/gpu-computing-answer-is-my-gpu-cuda-enabled-cat-ai

print(f"device is : {device}")

# %%
from transformers import AutoModelForMaskedLM, AutoTokenizer

#finetuning on IMDb
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")
imdb_dataset

model_checkpoint = 'distilbert/distilbert-base-uncased'

# Replace AutoModelForMaskedLM with the correct class for your task, e.g., AutoModelForSequenceClassification
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)



# %%
imdb_dataset["train"][0]

# %%
sample = imdb_dataset["unsupervised"].shuffle(seed=97).select(range(3))

# %%
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets

# %%
chunk_size = tokenizer.model_max_length // 2
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) if isinstance(examples[k][0], list) else examples[k] for k in examples.keys()}
    # Compute the total length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new "labels" column that is a copy of "input_ids"
    result["labels"] = result["input_ids"].copy()
    return result


# %%
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets

# %%
#deviate from the tutoral at this point, and look at documention instead 

# %%
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

tokenizer.pad_token = "[PAD]" #tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="movie_model",
    save_strategy="epoch",
    evaluation_strategy="epoch", 
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
    per_device_train_batch_size=32,  # lower this if memory error
)

trainer = Trainer(
    model=model.to(device),  # Move model to the specified device
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()


# %%
output_dir = "finetuned_models"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# %%
from transformers import pipeline

mask_filler_base = pipeline(
    "fill-mask", model=model_checkpoint, device="mps"
)

mask_filler_finetuned = pipeline(
    "fill-mask", model="finetuned_models/", device="mps"
)

# %%
text = "This is a great [MASK]"

#origional model
for pred in mask_filler_base(text):
    print(f"Origional >>> {pred['sequence']}")

print("\n")

#new IMDb finetuned model 
for pred in mask_filler_finetuned(text):
    print(f"Updated model >>> {pred['sequence']}")



