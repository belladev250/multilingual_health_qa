import os
import unittest
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.data_preprocessing import clean_text, apply_language_prefix, generate_mock_datasets
from src.dataset import MultilingualQADataset
from src.evaluation import compute_nlp_metrics
from src.inference import generate_answer, generate_submission

class TestMultilingualQAPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Paths
        cls.test_dir = "/Users/pixeleyeblue/.gemini/antigravity/scratch/multilingual_health_qa"
        cls.data_dir = os.path.join(cls.test_dir, "data/raw")
        cls.output_dir = os.path.join(cls.test_dir, "tests/output")
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Generate mock datasets if they don't exist
        generate_mock_datasets(cls.data_dir)
        
        # Use a super-tiny random model for testing to avoid huge model downloads
        cls.model_name = "hf-internal-testing/tiny-random-t5"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = AutoModelForSeq2SeqLM.from_pretrained(cls.model_name)

    def test_01_text_cleaning(self):
        html_text = "<p>Je, ni dalili gani za hatari?</p>"
        cleaned = clean_text(html_text)
        self.assertEqual(cleaned, "Je, ni dalili gani za hatari?")
        
        spacing_text = "Habari   ya   mama   na   mtoto"
        cleaned_spaces = clean_text(spacing_text)
        self.assertEqual(cleaned_spaces, "Habari ya mama na mtoto")

    def test_02_prefix_formatting(self):
        formatted = apply_language_prefix("Je, ni dalili gani?", "Kiswahili")
        self.assertIn("Language: Kiswahili", formatted)
        self.assertIn("Question: Je, ni dalili gani?", formatted)

    def test_03_dataset_and_loader(self):
        train_csv = os.path.join(self.data_dir, "train.csv")
        df = pd.read_csv(train_csv).head(5)
        
        dataset = MultilingualQADataset(
            df=df,
            tokenizer=self.tokenizer,
            max_source_length=32,
            max_target_length=32,
            use_prefix=True
        )
        
        self.assertEqual(len(dataset), 5)
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertEqual(item["input_ids"].shape, torch.Size([32]))
        self.assertEqual(item["labels"].shape, torch.Size([32]))

    def test_04_evaluation_metrics(self):
        preds = ["Je, ni dalili gani?", "Fever is dangerous"]
        refs = ["Je, ni dalili gani za hatari?", "High fever is extremely dangerous"]
        
        metrics = compute_nlp_metrics(preds, refs)
        self.assertIn("rouge1", metrics)
        self.assertIn("rougeL", metrics)
        self.assertIn("bleu", metrics)
        self.assertIn("semantic_similarity", metrics)
        self.assertGreater(metrics["rouge1"], 0.0)

    def test_05_inference_and_submission(self):
        test_csv = os.path.join(self.data_dir, "test.csv")
        test_df = pd.read_csv(test_csv).head(3)
        
        output_csv = os.path.join(self.output_dir, "test_sub.csv")
        
        # Run generation
        sub_df = generate_submission(
            model=self.model,
            tokenizer=self.tokenizer,
            test_df=test_df,
            output_csv_path=output_csv,
            use_prefix=True,
            decoding_strategy="greedy",
            max_length=16
        )
        
        self.assertTrue(os.path.exists(output_csv))
        self.assertEqual(len(sub_df), 3)
        self.assertListEqual(list(sub_df.columns), ["ID", "Answer"])

if __name__ == "__main__":
    unittest.main()
