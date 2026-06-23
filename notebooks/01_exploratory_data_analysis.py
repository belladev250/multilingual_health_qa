# %% [markdown]
# # 01. Exploratory Data Analysis (EDA)
# ### Final Project: Multilingual Health Question Answering in Low-Resource African Languages
# 
# This notebook explores the structural, linguistic, and statistical traits of the health QA dataset.
# We focus on four target languages: **Kiswahili**, **Luganda**, **Akan**, and **Amharic**.
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
import numpy as np
import matplotlib.pyplot as plt

# Align paths to make imports seamless from both notebooks/ and repository root
repo_root = os.path.abspath(os.getcwd())
if not os.path.exists(os.path.join(repo_root, "src")):
    repo_root = os.path.abspath(os.path.join(repo_root, ".."))
sys.path.append(repo_root)

from src.data_preprocessing import clean_text

# %% [markdown]
# ## 1. Loading the Dataset
# We check if the dataset files are located in `data/raw/` and load them.

# %%
DATA_DIR = os.path.join(repo_root, "data/raw")
train_path = os.path.join(DATA_DIR, "train.csv")

if not os.path.exists(train_path):
    print("Train dataset not found. Generating mock datasets first...")
    from src.data_preprocessing import generate_mock_datasets
    generate_mock_datasets(DATA_DIR)

df = pd.read_csv(train_path)
print(f"Dataset loaded. Total records: {len(df)}")

# Standardize columns to Question, Answer, Language for simplicity in notebook
if "subset" in df.columns:
    df = df.rename(columns={"input": "Question", "output": "Answer", "subset": "Language"})

df.head()

# %% [markdown]
# ## 2. Language Distribution Analysis
# Analyzing the proportion of questions per language in the corpus.

# %%
lang_counts = df["Language"].value_counts()
print("Records per Language:")
print(lang_counts)

plt.figure(figsize=(8, 5))
colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#EC4899', '#8B5CF6', '#3B82F6', '#6366F1']
lang_counts.plot(kind='bar', color=colors[:len(lang_counts)], edgecolor='black', linewidth=0.7)
plt.title('Distribution of Health QA Records by Language/Subset', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Language/Subset', fontsize=12, fontweight='bold')
plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, fontweight='semibold', ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
os.makedirs(os.path.join(repo_root, "experiments/plots"), exist_ok=True)
plt.savefig(os.path.join(repo_root, "experiments/plots/language_distribution.png"), dpi=300)
plt.show()

# %% [markdown]
# ## 3. Sequence Length Statistics (Words & Characters)
# Highly agglutinative languages (like Luganda and Kiswahili) have different word-length ratios.
# Analyzing sentence structures of both Questions and Answers.

# %%
df["q_word_len"] = df["Question"].apply(lambda x: len(str(x).split()))
df["a_word_len"] = df["Answer"].apply(lambda x: len(str(x).split()))

df["q_char_len"] = df["Question"].apply(lambda x: len(str(x)))
df["a_char_len"] = df["Answer"].apply(lambda x: len(str(x)))

print("--- Question Length Stats (Words) ---")
print(df["q_word_len"].describe())

print("\n--- Answer Length Stats (Words) ---")
print(df["a_word_len"].describe())

# %% [markdown]
# ## 4. Plotting Sequence Length Distributions
# Visualizing word-count densities to guide max-sequence-length truncation parameters during model loading.

# %%
plt.figure(figsize=(12, 5))

# Plot Questions Density
plt.subplot(1, 2, 1)
plt.hist(df["q_word_len"], bins=15, color='#4F46E5', alpha=0.85, edgecolor='black', linewidth=0.5)
plt.title('Question Word Length Density', fontsize=12, fontweight='bold')
plt.xlabel('Word Count', fontsize=10, fontweight='semibold')
log_y_q = True if len(df) > 100 else False
plt.ylabel('Frequency', fontsize=10, fontweight='semibold')
if log_y_q:
    plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.4)

# Plot Answers Density
plt.subplot(1, 2, 2)
plt.hist(df["a_word_len"], bins=15, color='#10B981', alpha=0.85, edgecolor='black', linewidth=0.5)
plt.title('Answer Word Length Density', fontsize=12, fontweight='bold')
plt.xlabel('Word Count', fontsize=10, fontweight='semibold')
log_y_a = True if len(df) > 100 else False
if log_y_a:
    plt.yscale('log')
plt.ylabel('Frequency', fontsize=10, fontweight='semibold')
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(repo_root, "experiments/plots/sequence_lengths.png"), dpi=300)
plt.show()

# %% [markdown]
# ## 5. Orthography Check: Ge'ez and Latin Script Differences
# Exploring the character configurations to highlight vocabulary coverage constraints in standard tokenizers.

# %%
print("Evaluating scripts and distinct character properties:")
for lang in df["Language"].unique():
    subset = df[df["Language"] == lang].head(1)
    q_sample = subset["Question"].values[0]
    print(f"\nLanguage/Subset: {lang}")
    print(f"Sample Question: {q_sample}")
    
    non_ascii = [char for char in q_sample if ord(char) > 127]
    print(f"Non-ASCII characters detected ({len(non_ascii)}): {' '.join(non_ascii[:10])}")
