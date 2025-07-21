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

# Q4: Qdrant with Jina embeddings
print("Q4: Evaluating Qdrant with Jina embeddings...")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Initialize Qdrant client (in-memory)
client = QdrantClient(":memory:")

# Initialize the Jina embedding model
model_handle = "jinaai/jina-embeddings-v2-small-en"
model = SentenceTransformer(model_handle)

# Create collection
collection_name = "documents"
vector_size = model.get_sentence_embedding_dimension()
print(f"Vector dimension: {vector_size}")

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

# Prepare texts and create embeddings
texts_qdrant = []
for doc in documents:
    text = doc['question'] + ' ' + doc['text']
    texts_qdrant.append(text)

print("Creating embeddings with Jina model...")
embeddings = model.encode(texts_qdrant)
print(f"Created embeddings with shape: {embeddings.shape}")

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
    # Create query embedding
    query_embedding = model.encode([q['question']])
    
    # Search in Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding[0].tolist(),
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