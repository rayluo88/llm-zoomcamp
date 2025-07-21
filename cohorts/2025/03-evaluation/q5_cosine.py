import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# Q5: Cosine similarity calculation
print("Q5: Calculating cosine similarity between LLM and original answers...")

# Load the results from gpt4o-mini evaluations
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
results_url = url_prefix + 'rag_evaluation/data/results-gpt4o-mini.csv'
df_results = pd.read_csv(results_url)

print(f"Loaded {len(df_results)} results")
print("Columns:", list(df_results.columns))
print("\nFirst few rows:")
print(df_results.head())

# Define cosine similarity functions
def normalize(u):
    norm = np.sqrt(u.dot(u))
    return u / norm

def cosine(u, v):
    u = normalize(u)
    v = normalize(v)
    return u.dot(v)

# Alternative simplified version
def cosine_simplified(u, v):
    u_norm = np.sqrt(u.dot(u))
    v_norm = np.sqrt(v.dot(v))
    return u.dot(v) / (u_norm * v_norm)

# Create the pipeline for embeddings
pipeline = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)

# Fit the vectorizer on all text data
print("\nFitting pipeline on all text data...")
all_text = df_results.answer_llm + ' ' + df_results.answer_orig + ' ' + df_results.question
pipeline.fit(all_text)

# Transform LLM answers and original answers to get embeddings
print("Creating embeddings...")
v_llm = pipeline.transform(df_results.answer_llm)
v_orig = pipeline.transform(df_results.answer_orig)

print(f"LLM embeddings shape: {v_llm.shape}")
print(f"Original embeddings shape: {v_orig.shape}")

# Calculate cosine similarity for each pair
print("Calculating cosine similarities...")
cosine_similarities = []

for i in range(len(df_results)):
    # Get embeddings for this pair
    llm_embedding = v_llm[i]  # Already dense after SVD
    orig_embedding = v_orig[i]  # Already dense after SVD
    
    # Calculate cosine similarity
    similarity = cosine_simplified(llm_embedding, orig_embedding)
    cosine_similarities.append(similarity)
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(df_results)} pairs")

# Calculate average cosine similarity
average_cosine = np.mean(cosine_similarities)

print(f"\nTotal pairs processed: {len(cosine_similarities)}")
print(f"Average cosine similarity: {average_cosine:.4f}")
print(f"Average cosine similarity (rounded): {average_cosine:.2f}")

# Show some statistics
print(f"\nStatistics:")
print(f"Min cosine similarity: {np.min(cosine_similarities):.4f}")
print(f"Max cosine similarity: {np.max(cosine_similarities):.4f}")
print(f"Median cosine similarity: {np.median(cosine_similarities):.4f}")
print(f"Std cosine similarity: {np.std(cosine_similarities):.4f}")

# Show first few similarities for verification
print(f"\nFirst 10 cosine similarities:")
for i in range(min(10, len(cosine_similarities))):
    print(f"Pair {i+1}: {cosine_similarities[i]:.4f}") 