import requests
import pandas as pd
from tqdm.auto import tqdm
import openai
import os
import time

# Load data
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
ground_truth = df_ground_truth.to_dict(orient='records')

# Evaluation functions
def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt = cnt + 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)
    
    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

# Q4: Qdrant with OpenAI embeddings (alternative to Jina)
print("Q4: Evaluating Qdrant with OpenAI embeddings...")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize Qdrant client (in-memory)
client = QdrantClient(":memory:")

# OpenAI client setup
openai_client = openai.OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding using OpenAI API with rate limiting"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        time.sleep(1)  # Rate limiting
        return None

# Create collection
collection_name = "documents"
vector_size = 1536  # OpenAI text-embedding-3-small dimension
print(f"Vector dimension: {vector_size}")

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

# Prepare texts and create embeddings
print("Creating embeddings with OpenAI model...")
texts_qdrant = []
embeddings = []

for i, doc in enumerate(documents):
    text = doc['question'] + ' ' + doc['text']
    texts_qdrant.append(text)
    
    if i % 10 == 0:
        print(f"Processing document {i+1}/{len(documents)}")
    
    embedding = get_embedding(text)
    if embedding:
        embeddings.append(embedding)
    else:
        # Skip failed embeddings
        embeddings.append([0.0] * vector_size)
    
    # Rate limiting
    time.sleep(0.05)  # 20 requests per second

print(f"Created {len(embeddings)} embeddings")

# Index documents in Qdrant
points = []
for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
    point = PointStruct(
        id=i,
        vector=embedding,
        payload={
            "id": doc["id"],
            "course": doc["course"],
            "question": doc["question"],
            "text": doc["text"],
            "section": doc["section"]
        }
    )
    points.append(point)

print("Indexing documents in Qdrant...")
client.upsert(
    collection_name=collection_name,
    points=points
)

def qdrant_search_function(q):
    # Create query embedding
    query_embedding = get_embedding(q['question'])
    if not query_embedding:
        return []
    
    # Search in Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        query_filter={
            "must": [
                {
                    "key": "course",
                    "match": {"value": q['course']}
                }
            ]
        },
        limit=5
    )
    
    # Convert results to expected format
    results = []
    for hit in search_result:
        results.append({
            "id": hit.payload["id"],
            "course": hit.payload["course"],
            "question": hit.payload["question"],
            "text": hit.payload["text"],
            "section": hit.payload["section"]
        })
    
    return results

# Evaluate the Qdrant approach
print("Evaluating Qdrant search...")
qdrant_results = evaluate(ground_truth, qdrant_search_function)
print(f"Hit rate: {qdrant_results['hit_rate']:.2f}")
print(f"MRR: {qdrant_results['mrr']:.2f}") 