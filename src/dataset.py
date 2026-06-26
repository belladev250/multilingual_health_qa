import torch
from torch.utils.data import Dataset
from src.data_preprocessing import clean_text, apply_language_prefix

class MultilingualQADataset(Dataset):
    """
    Custom PyTorch Dataset for Multilingual Sequence-to-Sequence (Seq2Seq)
    Question Answering. Supports training and test evaluation formats.
    """
    def __init__(self, df, tokenizer, max_source_length=128, max_target_length=128, 
                 use_prefix=False, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_prefix = use_prefix
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and clean question
        question = clean_text(row.get("Question", row.get("input", "")))
        language = row.get("Language", row.get("subset", ""))
        
        # Apply language-specific prefix formatting (EXP-03)
        if self.use_prefix and language:
            source_text = apply_language_prefix(question, language)
        else:
            source_text = question
            
        # Tokenize source sequence
        source_inputs = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": source_inputs["input_ids"].squeeze(0),
            "attention_mask": source_inputs["attention_mask"].squeeze(0)
        }
        
        if not self.is_test:
            # Load and clean answer
            answer = clean_text(row.get("Answer", row.get("output", "")))
            
            # Tokenize target sequence
            target_inputs = self.tokenizer(
                text_target=answer,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
                
            labels = target_inputs["input_ids"].squeeze(0)
            
            # Replace padding token id with -100 so Hugging Face PyTorch loss ignores it
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            item["labels"] = labels
            
        return item
