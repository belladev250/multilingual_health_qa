# %% [markdown]
# # 03. Training, Fine-Tuning & Inference Pipeline
# ### Final Project: Multilingual Health Question Answering in Low-Resource African Languages
# 
# This notebook handles loading sequence-to-sequence models (like `mT5`), wrapping them
# with PEFT LoRA adapters (EXP-07), training them, evaluating, and generating submissions.
# 
# ---

# %%
import os
import sys
import pandas as pd
import numpy as np
import torch

# Align paths to make imports seamless
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.model import load_model_and_tokenizer
from src.training import train_model
from src.inference import generate_answer, generate_submission

# %% [markdown]
# ## 1. Setting up Model & PEFT LoRA Configuration (EXP-07)
# We load the model and optionally apply parameter-efficient adapters.
# This trains ~1-2% of total parameters, optimizing GPU usage and reducing overfitting.

# %%
MODEL_NAME = "hf-internal-testing/tiny-random-t5" 

model, tokenizer = load_model_and_tokenizer(
    model_name=MODEL_NAME,
    use_peft=True,  # Enable PEFT LoRA!
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# %% [markdown]
# ## 2. Preparing Datasets for Training
# We load the processed datasets. If not present, we generate mock data.

# %%
DATA_DIR = "../data/raw"
train_path = os.path.join(DATA_DIR, "train.csv")

if not os.path.exists(train_path):
    from src.data_preprocessing import generate_mock_datasets
    generate_mock_datasets(DATA_DIR)

df_all = pd.read_csv(train_path)

# Train-Val split
np_random = pd.Series(range(len(df_all))).sample(frac=1.0, random_state=42)
train_indices = np_random.head(int(len(df_all) * 0.8)).index
val_indices = np_random.tail(len(df_all) - len(train_indices)).index

train_df = df_all.loc[train_indices].copy()
val_df = df_all.loc[val_indices].copy()

print(f"Data split successfully. Train size: {len(train_df)} | Val size: {len(val_df)}")

# %% [markdown]
# ## 3. Fine-Tuning execution (EXP-01 - EXP-08)
# We execute standard seq2seq training using the `train_model` engine.

# %%
OUTPUT_DIR = "../experiments/outputs_checkpoint"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# For speed during presentation/demo, we use small limits
train_result, eval_metrics = train_model(
    model=model,
    tokenizer=tokenizer,
    train_df=train_df.head(10), # Smoke-test slice
    val_df=val_df.head(3),
    output_dir=OUTPUT_DIR,
    epochs=1,
    lr=2e-4,
    batch_size=2,
    scheduler="cosine",
    use_prefix=True
)

# %% [markdown]
# ## 4. Custom Generation and Decoding Strategies (EXP-10)
# We generate a healthcare answer for a given question using custom generation decoders
# such as **Contrastive Search**, **Nucleus (Top-p) Sampling**, and **Beam Search**.

# %%
test_q = "Je, ni dalili gani za hatari wakati wa ujauzito?"
test_lang = "Kiswahili"

for dec_strat in ["greedy", "beam_search", "nucleus_sampling", "contrastive_search"]:
    ans = generate_answer(
        model=model,
        tokenizer=tokenizer,
        question=test_q,
        language=test_lang,
        use_prefix=True,
        decoding_strategy=dec_strat,
        max_length=32
    )
    print(f"\nDecoding Strategy: {dec_strat.upper()}")
    print(f"Generated Answer: '{ans}'")

# %% [markdown]
# ## 5. Submitting Predictions to Zindi
# Generating predictions for the test set and formatting the output CSV file
# according to the competition instructions.

# %%
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).head(5)
submission_path = "../data/processed/zindi_submission.csv"
os.makedirs("../data/processed", exist_ok=True)

sub_df = generate_submission(
    model=model,
    tokenizer=tokenizer,
    test_df=test_df,
    output_csv_path=submission_path,
    use_prefix=True,
    decoding_strategy="contrastive_search",
    max_length=32
)

print("\nSample Predictions from Submission DataFrame:")
sub_df.head()
