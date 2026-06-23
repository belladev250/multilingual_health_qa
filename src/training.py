import os
import torch
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, IntervalStrategy
from src.dataset import MultilingualQADataset
from src.evaluation import compute_nlp_metrics

def train_model(model, tokenizer, train_df, val_df, output_dir: str, 
                epochs: int = 3, lr: float = 5e-4, batch_size: int = 4, 
                scheduler: str = "linear", use_prefix: bool = False,
                max_source_len: int = 128, max_target_len: int = 128):
    """
    Fine-tunes a Sequence-to-Sequence model using Hugging Face Seq2SeqTrainer.
    Supports standard training, learning rate schedulers, and prefix prompting.
    """
    print(f"Initializing datasets | Use prefix: {use_prefix}")
    train_dataset = MultilingualQADataset(
        train_df, tokenizer, 
        max_source_length=max_source_len, 
        max_target_length=max_target_len, 
        use_prefix=use_prefix, 
        is_test=False
    )
    val_dataset = MultilingualQADataset(
        val_df, tokenizer, 
        max_source_length=max_source_len, 
        max_target_length=max_target_len, 
        use_prefix=use_prefix, 
        is_test=False
    )

    # Initialize data collator for batching
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    # Define compute metrics function inside the trainer closure
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Decode predictions and references
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in labels as we can't decode it
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean white spaces
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Compute NLP metrics (ROUGE, BLEU, Semantic Similarity)
        return compute_nlp_metrics(decoded_preds, decoded_labels)

    # Training configuration arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        lr_scheduler_type=scheduler,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=max_target_len,
        fp16=torch.cuda.is_available(), # Leverage GPU half-precision if available
        dataloader_num_workers=0,
        report_to="none" # Turn off external logging integrations for reproducibility
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting fine-tuning...")
    train_result = trainer.train()
    print("Training finished successfully.")

    # Run evaluation
    print("Evaluating model...")
    eval_metrics = trainer.evaluate()
    print(f"Evaluation metrics: {eval_metrics}")

    # Save final model state
    model_save_path = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(model_save_path)
    print(f"Model saved successfully to: {model_save_path}")

    return train_result, eval_metrics
