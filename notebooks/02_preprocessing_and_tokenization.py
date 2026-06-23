# %% [markdown]
# # 02. Preprocessing & Tokenization Pipeline
# ### Final Project: Multilingual Health Question Answering in Low-Resource African Languages
# 
# This notebook demonstrates our data cleaning strategies, language-specific prompt prefixing,
# and evaluates tokenizer fracturing and vocabulary indexing for low-resource languages.
# 
# ---

# %%
# Google Colab Setup (Runs automatically if in Google Colab environment)
import os
import sys

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Detected Google Colab environment. Setting up workspace...")
    if not os.path.exists("multilingual_health_qa"):
        os.system("git clone https://github.com/belladev250/multilingual_health_qa.git")
    os.chdir("multilingual_health_qa")
    os.system("pip install -r requirements.txt")
    print("Colab workspace configured successfully!")

import pandas as pd
from transformers import AutoTokenizer

# Align paths to make imports seamless from both notebooks/ and repository root
repo_root = os.path.abspath(os.getcwd())
if not os.path.exists(os.path.join(repo_root, "src")):
    repo_root = os.path.abspath(os.path.join(repo_root, ".."))
sys.path.append(repo_root)

from src.data_preprocessing import clean_text, apply_language_prefix, simulate_back_translation

# %% [markdown]
# ## 1. Text Normalization and Cleaning (EXP-02)
# We test cleaning noise, stripping HTML markers, and applying Unicode NFKC normalization.

# %%
raw_sample = "<p>Je, ni  dalili  gani za hatari wakati wa ujauzito?</p>"
cleaned = clean_text(raw_sample)

print("Raw Sample:")
print(raw_sample)
print("\nCleaned Sample (Standardized whitespace & unicode merged):")
print(cleaned)

# %% [markdown]
# ## 2. Language-Specific Prompt Prefixing (EXP-03)
# To guide the multilingual decoder (like `mT5`), we append source language tags to the questions.

# %%
question = "Je, ni dalili gani za hatari?"
prefix_sample = apply_language_prefix(question, "Kiswahili")
print("Standard Prompt Format:")
print(prefix_sample)

# %% [markdown]
# ## 3. Back-Translation Data Augmentation Simulation (EXP-04)
# Simulating back-translation-based data expansions by modifying synonms to double data samples.

# %%
orig_text = "Nifanye nini kumlinda mtoto mchanga dhidi ya malaria?"
augmented_text = simulate_back_translation(orig_text, "Kiswahili")

print("Original Text:")
print(orig_text)
print("\nAugmented Text:")
print(augmented_text)

# %% [markdown]
# ## 4. Evaluating Tokenizer Fracturing (Sub-word Over-fragmentation)
# We load the pre-trained `google/mt5-small` tokenizer. We observe how low-resource languages
# are fractured compared to high-resource languages, demonstrating why max sequence lengths must be handled carefully.

# %%
model_name = "hf-internal-testing/tiny-random-t5" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample phrases
kiswahili_phrase = "Je, ni dalili gani za hatari?"
english_phrase = "What are the danger signs?"

# Tokenize and view tokens
k_tokens = tokenizer.tokenize(kiswahili_phrase)
e_tokens = tokenizer.tokenize(english_phrase)

print(f"Kiswahili text: '{kiswahili_phrase}'")
print(f"Kiswahili sub-word tokens ({len(k_tokens)}): {k_tokens}")
print(f"Kiswahili tokens IDs: {tokenizer.encode(kiswahili_phrase)}")

print(f"\nEnglish text: '{english_phrase}'")
print(f"English sub-word tokens ({len(e_tokens)}): {e_tokens}")
print(f"English tokens IDs: {tokenizer.encode(english_phrase)}")

# %% [markdown]
# ## 5. Building PyTorch Dataset Objects
# Loading a sample DataFrame and building our PyTorch Dataset objects ready for PyTorch dataloaders.

# %%
from src.dataset import MultilingualQADataset

df_mock = pd.DataFrame([
    {"Question": "Je, ni dalili gani za hatari?", "Answer": "Dalili za hatari ni pamoja na kuvuja damu.", "Language": "Kiswahili"},
    {"Question": "Obubonero obw'akabi kuki?", "Answer": "Obubonero obw'akabi mwe muli okutonnya omusaayi.", "Language": "Luganda"}
])

dataset = MultilingualQADataset(
    df=df_mock,
    tokenizer=tokenizer,
    max_source_length=32,
    max_target_length=32,
    use_prefix=True
)

print(f"Dataset created successfully. Size: {len(dataset)}")
sample_batch = dataset[0]
print("Keys in batch sample:", sample_batch.keys())
print("Tokens shape:", sample_batch["input_ids"].shape)
print("Labels shape:", sample_batch["labels"].shape)
