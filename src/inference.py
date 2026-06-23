import torch
import pandas as pd
from tqdm import tqdm
from src.data_preprocessing import clean_text, apply_language_prefix

def generate_answer(model, tokenizer, question: str, language: str = "", 
                    use_prefix: bool = False, decoding_strategy: str = "beam_search", 
                    max_length: int = 128) -> str:
    """
    Generates a text answer for a given question using various decoding strategies (EXP-10).
    """
    model.eval()
    
    # Process text
    question = clean_text(question)
    if use_prefix and language:
        input_text = apply_language_prefix(question, language)
    else:
        input_text = question
        
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Configure generation parameters based on decoding strategy
    gen_kwargs = {
        "max_length": max_length,
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
    }
    
    if decoding_strategy == "beam_search":
        gen_kwargs["num_beams"] = 4
    elif decoding_strategy == "contrastive_search":
        gen_kwargs["penalty_alpha"] = 0.6
        gen_kwargs["top_k"] = 4
    elif decoding_strategy == "nucleus_sampling":
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_p"] = 0.92
        gen_kwargs["temperature"] = 0.7
    else:  # "greedy"
        gen_kwargs["do_sample"] = False
        gen_kwargs["early_stopping"] = False # Not used for greedy
        
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
        
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return clean_text(generated_text)

def generate_submission(model, tokenizer, test_df: pd.DataFrame, 
                        output_csv_path: str, use_prefix: bool = False, 
                        decoding_strategy: str = "beam_search", max_length: int = 128):
    """
    Iterates through the test dataset, generates answers for each question,
    and writes out a submission CSV compatible with Zindi's actual schema:
    ID, TargetRLF1, TargetR1F1, TargetLLM.
    Each of the three target columns must contain the same predicted answer text.
    """
    print(f"Generating batch predictions using strategy: {decoding_strategy}...")
    results = []
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        q_id = row.get("ID")
        question = row.get("Question", row.get("input", ""))
        language = row.get("Language", row.get("subset", ""))
        
        pred_ans = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            language=language,
            use_prefix=use_prefix,
            decoding_strategy=decoding_strategy,
            max_length=max_length
        )
        
        results.append({
            "ID": q_id,
            "TargetRLF1": pred_ans,
            "TargetR1F1": pred_ans,
            "TargetLLM": pred_ans
        })
        
    submission_df = pd.DataFrame(results)
    # Ensure correct column order
    submission_df = submission_df[["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]]
    submission_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"✅ Submission generated successfully and saved to: {output_csv_path}")
    return submission_df
