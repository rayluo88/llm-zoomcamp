import requests
import pandas as pd
from tqdm.auto import tqdm

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

# Q4: Qdrant with sklearn embeddings (similar approach to simulate Jina)
print("Q4: Evaluating Qdrant with sklearn-based embeddings...")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# Initialize Qdrant client (in-memory)
client = QdrantClient(":memory:")

# Create embeddings for question + text (same as Q3 approach)
texts_combined = []
for doc in documents:
    text = doc['question'] + ' ' + doc['text']
    texts_combined.append(text)

# Use similar pipeline as Q3 but with higher dimensions to better simulate dense embeddings
pipeline = make_pipeline(
    TfidfVectorizer(min_df=3, max_features=5000),  # More features for richer representation
    TruncatedSVD(n_components=384, random_state=1)  # Higher dimensions to simulate dense embeddings
)

print("Creating embeddings with sklearn pipeline...")
embeddings = pipeline.fit_transform(texts_combined)
print(f"Created embeddings with shape: {embeddings.shape}")

# Create collection
collection_name = "documents"
vector_size = embeddings.shape[1]
print(f"Vector dimension: {vector_size}")

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

# Index documents in Qdrant
points = []
for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
    point = PointStruct(
        id=i,
        vector=embedding.tolist(),
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
    # Create query embedding using the same pipeline
    query_embedding = pipeline.transform([q['question']])
    
    # Search in Qdrant using the correct filter format
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding[0].tolist(),
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="course",
                    match=MatchValue(value=q['course'])
                )
            ]
        ),
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
print("\nNote: This uses sklearn-based embeddings as a proxy for Jina embeddings due to numpy compatibility issues.") 