import requests
import pandas as pd
from rouge import Rouge

# Q6: ROUGE score calculation
print("Q6: Calculating ROUGE scores between LLM and original answers...")

# Load the results from gpt4o-mini evaluations
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
results_url = url_prefix + 'rag_evaluation/data/results-gpt4o-mini.csv'
df_results = pd.read_csv(results_url)

print(f"Loaded {len(df_results)} results")

# Initialize ROUGE scorer
rouge_scorer = Rouge()

# First, let's look at the specific example at index 10
print("\n=== Example at index 10 ===")
r = df_results.iloc[10]
print(f"Document ID: {r['document']}")
print(f"LLM Answer: {r['answer_llm'][:100]}...")
print(f"Original Answer: {r['answer_orig'][:100]}...")

# Calculate ROUGE scores for the 10th document
scores = rouge_scorer.get_scores(r.answer_llm, r.answer_orig)[0]
print(f"\nROUGE scores for index 10:")
print(f"ROUGE-1: {scores['rouge-1']}")
print(f"ROUGE-2: {scores['rouge-2']}")
print(f"ROUGE-L: {scores['rouge-l']}")
print(f"ROUGE-1 F1 score: {scores['rouge-1']['f']:.4f}")

# Now calculate ROUGE scores for all pairs
print(f"\n=== Calculating ROUGE for all {len(df_results)} pairs ===")
rouge_1_f1_scores = []
rouge_2_f1_scores = []
rouge_l_f1_scores = []

for i in range(len(df_results)):
    try:
        r = df_results.iloc[i]
        scores = rouge_scorer.get_scores(r.answer_llm, r.answer_orig)[0]
        
        rouge_1_f1_scores.append(scores['rouge-1']['f'])
        rouge_2_f1_scores.append(scores['rouge-2']['f'])
        rouge_l_f1_scores.append(scores['rouge-l']['f'])
        
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(df_results)} pairs")
            
    except Exception as e:
        print(f"Error processing pair {i}: {e}")
        # Add zero scores for failed cases
        rouge_1_f1_scores.append(0.0)
        rouge_2_f1_scores.append(0.0)
        rouge_l_f1_scores.append(0.0)

# Calculate averages
avg_rouge_1_f1 = sum(rouge_1_f1_scores) / len(rouge_1_f1_scores)
avg_rouge_2_f1 = sum(rouge_2_f1_scores) / len(rouge_2_f1_scores)
avg_rouge_l_f1 = sum(rouge_l_f1_scores) / len(rouge_l_f1_scores)

print(f"\n=== Results ===")
print(f"Total pairs processed: {len(rouge_1_f1_scores)}")
print(f"Average ROUGE-1 F1: {avg_rouge_1_f1:.4f}")
print(f"Average ROUGE-1 F1 (rounded): {avg_rouge_1_f1:.2f}")
print(f"Average ROUGE-2 F1: {avg_rouge_2_f1:.4f}")
print(f"Average ROUGE-L F1: {avg_rouge_l_f1:.4f}")

# Show some statistics for ROUGE-1 F1
import numpy as np
print(f"\nROUGE-1 F1 Statistics:")
print(f"Min: {np.min(rouge_1_f1_scores):.4f}")
print(f"Max: {np.max(rouge_1_f1_scores):.4f}")
print(f"Median: {np.median(rouge_1_f1_scores):.4f}")
print(f"Std: {np.std(rouge_1_f1_scores):.4f}")

# Show first 10 ROUGE-1 F1 scores
print(f"\nFirst 10 ROUGE-1 F1 scores:")
for i in range(min(10, len(rouge_1_f1_scores))):
    print(f"Pair {i+1}: {rouge_1_f1_scores[i]:.4f}")

# Verify the 10th document score
print(f"\nVerification - Index 10 ROUGE-1 F1: {rouge_1_f1_scores[10]:.4f}") 