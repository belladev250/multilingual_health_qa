# Multilingual Health QA in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/antigravity/blob/main/notebooks/03_training_and_fine_tuning.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.0+-orange.svg)](https://pytorch.org/)
[![PEFT: LoRA](https://img.shields.io/badge/PEFT-LoRA/QLoRA-green.svg)](https://github.com/huggingface/peft)

An end-to-end, modular, and Google Colab-compatible machine learning pipeline developed for the **Zindi Multilingual Health Question Answering in Low-Resource African Languages Challenge**. 

This project aims to fine-tune pre-trained multilingual sequence-to-sequence language models (such as `mT5-small` and `mT5-base`) to answer maternal, sexual, and reproductive health questions in four regional African languages: **Kiswahili (swa)**, **Luganda (lug)**, **Amharic (amh)**, and **Akan (aka)**.

---

## 📌 Project Overview

Answering health-related questions in low-resource African languages presents unique challenges due to:
- **Agglutinative Morphology**: Languages like Luganda and Kiswahili contain rich prefix/suffix structures, causing massive sub-word fracturing in standard tokenizers.
- **Script Divergence**: Amharic uses the unique Ge'ez syllabary script, which behaves very differently from Latin-based scripts.
- **Data Scarcity**: Lack of large-scale annotated medical dialogue datasets.

This repository implements a rigorous research workflow with **10 distinct experiments** tracking text preprocessing, prompt conditioning, model scaling, data augmentation, parameter-efficient fine-tuning (PEFT/LoRA), multi-task joint learning, ensembling, and decoding strategy optimizations.

---

## 📁 Repository Structure

```
multilingual_health_qa/
├── README.md                 # This comprehensive master repository guide
├── requirements.txt          # Python dependencies list
├── data/
│   ├── raw/                  # Zindi source files (train.csv, test.csv, SampleSubmission.csv)
│   └── processed/            # Cleaned, tokenized, and split datasets
├── src/                      # Modular Python package for core logic
│   ├── __init__.py
│   ├── data_preprocessing.py # Text normalization, unicode cleaning, language-tagging, augmentation
│   ├── dataset.py            # PyTorch datasets and Hugging Face Dataset loaders
│   ├── model.py              # Sequence-to-Sequence model adapters and LoRA injection
│   ├── training.py           # Training loop and Hugging Face Seq2SeqTrainer config
│   ├── evaluation.py         # Custom evaluation metrics (ROUGE-1, ROUGE-L, BLEU, similarity)
│   └── inference.py          # Custom decoders (Beam Search, Nucleus Sampling, Contrastive Search)
├── notebooks/                # Python scripts in standard `# %%` interactive notebook format
│   ├── 01_exploratory_data_analysis.py    # Exploratory data distribution analysis
│   ├── 02_preprocessing_and_tokenization.py # Orthography & tokenizer fracture exploration
│   └── 03_training_and_fine_tuning.py      # End-to-end training and local interface panel
├── experiments/              # Experiment config, logs, and progression plotting
│   ├── run_experiments.py    # Auto-runner script executing the 10 sequential experiments
│   ├── tracking_log.json     # JSON registry tracking all experimental metrics
│   └── plots/                # Automatically generated progression charts and learning curves
└── docs/                     # Academic deliverables and IEEE Report
    ├── StudentName_FinalProject_Draft.md  # Comprehensive research paper draft
    └── figures/              # Figures and plots for the IEEE paper
```

---

## 🛠️ Installation & Setup

### 1. Local Environment Setup

Clone this project or navigate to its directory, then run:

```bash
# Create python virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade package manager and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Mock Data Generation

To ensure the pipeline runs out-of-the-box before dropping in raw Zindi dataset files, run the mock data generator:

```bash
python src/data_preprocessing.py
```
This will construct synthetic datasets inside `data/raw/` in Amharic, Kiswahili, Luganda, and Akan, enabling complete validation tests and dry-run experiments.

### 3. Zindi Challenge Data Integration

For real-world competition training, download the dataset files from Zindi and overwrite the mock placeholders:
- Place `train.csv` in `data/raw/train.csv`
- Place `test.csv` in `data/raw/test.csv`
- Place `SampleSubmission.csv` in `data/raw/SampleSubmission.csv`

---

## 🚀 Running the Pipeline

### Running Automated Tests
To verify data preparation, dataset loading, LoRA insertion, and evaluation metrics, execute:
```bash
python -m unittest tests/test_pipeline.py
```

### Running the 10-Experiment Suite
To run all 10 experiments automatically in smoke-test mode (low epoch count for quick pipeline validation and chart rendering):
```bash
python experiments/run_experiments.py --smoke-test
```
For full training on real data (requires an active GPU):
```bash
python experiments/run_experiments.py
```
This will output comparative progress plots inside `experiments/plots/experiment_progression.png` and update `experiments/tracking_log.json`.

---

## 🔬 The 10-Experiment Roadmap

This repository systematically implements and evaluates:

1. **EXP-01: Baseline Model** — Train `google/mt5-small` on raw text questions and answers.
2. **EXP-02: Advanced Text Cleaning** — Filter HTML, handle unicode NFKC (essential for Amharic script), and fix spacing.
3. **EXP-03: Language-Specific Prompting** — Prepend source language tags (e.g., `Language: Kiswahili | Question: ...`) to condition target generation.
4. **EXP-04: Data Augmentation** — Simulate back-translation with synonym and phrasal variations.
5. **EXP-05: Model Scaling** — Scale model parameters by swapping `mT5-small` with `google/mt5-base`.
6. **EXP-06: Hyperparameter Tuning** — Shift learning rates to `2e-4` with Cosine Annealing scheduler and larger batches.
7. **EXP-07: PEFT (QLoRA) Fine-tuning** — Use Parameter-Efficient Fine-Tuning (LoRA) on attention matrices to train larger backbones on consumer GPUs.
8. **EXP-08: Multi-Task & Adapter Fusion** — Unified model using joint learning to leverage cross-lingual representations.
9. **EXP-09: Ensembling** — Ensemble predictions from multiple models via length-penalized sequence voting.
10. **EXP-10: Decoding Strategy Optimization** — Evaluate Contrastive Search and Nucleus Sampling vs. default Beam Search to prevent generation loops.

---

## 📄 Academic Deliverables

Our academic draft paper is located at `docs/StudentName_FinalProject_Draft.md` written in IEEE paper format. It covers:
- **Comprehensive EDA Findings** (orthographic and sub-word fracture analysis).
- **Technical Methodology** (parameter counts, optimization, and PEFT mathematical formulations).
- **Zindi Leaderboard Evaluation & Analysis** (detailed results table mapping ROUGE scores across all 10 experiments).
- **Ethics and Limitations Analysis** (mitigating hallucinated medical information in low-resource settings).

---

## 💡 Google Colab Integration

To run on free-tier Colab, simply open the Python files in the `notebooks/` directory which are structured with standard `# %%` code cell block syntax. They function natively as Jupyter Notebooks inside modern IDEs (VS Code, Cursor) and can be easily run in Colab by uploading them or utilizing the badge link at the top of this document.

For Colab run, make sure to enable **T4 GPU** runtime hardware acceleration to speed up training.
