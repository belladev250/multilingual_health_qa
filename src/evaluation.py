import numpy as np
from typing import List

try:
    import evaluate
    ROUGE_EVALUATOR = evaluate.load("rouge")
    BLEU_EVALUATOR = evaluate.load("bleu")
    HAS_EVALUATE = True
except Exception:
    HAS_EVALUATE = False

def compute_nlp_metrics(predictions: List[str], references: List[str]) -> dict:
    """
    Computes ROUGE-1, ROUGE-L, BLEU and simulated Semantic Similarity scores.
    """
    if not predictions or not references or len(predictions) != len(references):
        return {"rouge1": 0.0, "rougeL": 0.0, "bleu": 0.0, "semantic_similarity": 0.0}

    # Handle blank values safely
    predictions = [p if isinstance(p, str) and p.strip() else "empty" for p in predictions]
    references = [r if isinstance(r, str) and r.strip() else "empty" for r in references]

    results = {}
    
    if HAS_EVALUATE:
        try:
            # Use Hugging Face evaluate library
            rouge_res = ROUGE_EVALUATOR.compute(predictions=predictions, references=references)
            results["rouge1"] = float(rouge_res.get("rouge1", 0.0))
            results["rougeL"] = float(rouge_res.get("rougeL", 0.0))
            
            # BLEU expects references as list-of-lists
            bleu_refs = [[ref] for ref in references]
            bleu_res = BLEU_EVALUATOR.compute(predictions=predictions, references=bleu_refs)
            results["bleu"] = float(bleu_res.get("bleu", 0.0))
        except Exception as e:
            print(f"Hugging Face evaluation library encountered error: {e}. Falling back to manual.")
            results = compute_manual_metrics(predictions, references)
    else:
        results = compute_manual_metrics(predictions, references)

    # Compute simulated AfroLM BERTScore / Semantic Similarity (as mentioned in challenge details)
    # This evaluates content overlap and semantic similarity using TF-IDF or Jaccard
    semantic_scores = []
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if not pred_words or not ref_words:
            semantic_scores.append(0.0)
            continue
        intersection = pred_words.intersection(ref_words)
        jaccard = len(intersection) / len(pred_words.union(ref_words))
        
        # Scale to match BertScore range (0.3 - 1.0 typically)
        bertscore_proxy = 0.3 + 0.7 * jaccard
        semantic_scores.append(bertscore_proxy)
        
    results["semantic_similarity"] = float(np.mean(semantic_scores))
    
    return results

def compute_manual_metrics(predictions: List[str], references: List[str]) -> dict:
    """
    Fallback method to calculate basic ROUGE and BLEU scores using string overlap
    when Hugging Face libraries are not accessible.
    """
    rouge1_scores = []
    rougeL_scores = []
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        p_tokens = pred.lower().split()
        r_tokens = ref.lower().split()
        
        if not p_tokens or not r_tokens:
            rouge1_scores.append(0.0)
            rougeL_scores.append(0.0)
            bleu_scores.append(0.0)
            continue
            
        # ROUGE-1 (Unigram overlap)
        matches = sum(1 for tok in p_tokens if tok in r_tokens)
        rec = matches / len(r_tokens)
        prec = matches / len(p_tokens)
        f1_r1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        rouge1_scores.append(f1_r1)
        
        # ROUGE-L (Longest Common Subsequence)
        lcs_len = longest_common_subsequence(p_tokens, r_tokens)
        rec_l = lcs_len / len(r_tokens)
        prec_l = lcs_len / len(p_tokens)
        f1_rl = (2 * prec_l * rec_l) / (prec_l + rec_l) if (prec_l + rec_l) > 0 else 0.0
        rougeL_scores.append(f1_rl)
        
        # BLEU-1 (Simple precision proxy)
        bleu_scores.append(prec)
        
    return {
        "rouge1": float(np.mean(rouge1_scores)),
        "rougeL": float(np.mean(rougeL_scores)),
        "bleu": float(np.mean(bleu_scores))
    }

def longest_common_subsequence(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
