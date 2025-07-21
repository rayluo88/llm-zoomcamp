import requests
import pandas as pd

url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
ground_truth = df_ground_truth.to_dict(orient='records')

from tqdm.auto import tqdm

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


# Q1: Minsearch with boosting parameters
import minsearch

# Create minsearch index with boosting parameters
boost = {'question': 1.5, 'section': 0.1}
index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

# Index all documents
index.fit(documents)

def minsearch_function(q):
    query = q['question']
    results = index.search(
        query=query, 
        filter_dict={"course": q['course']}, 
        boost_dict=boost,
        num_results=5
    )
    return results

# Evaluate the minsearch approach
print("Evaluating minsearch with boosting parameters...")
results = evaluate(ground_truth, minsearch_function)
print(f"Hit rate: {results['hit_rate']:.2f}")
print(f"MRR: {results['mrr']:.2f}")

# Vector Search Implementation
from minsearch import VectorSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

print("\nCreating embeddings for vector search...")

# Create embeddings for the "question" field
texts = []

for doc in documents:
    t = doc['question']
    texts.append(t)

pipeline = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)
X = pipeline.fit_transform(texts)

print(f"Created embeddings with shape: {X.shape}")

# Q2: Vector search for question
print("\nQ2: Evaluating vector search...")

# Create VectorSearch index
vindex = VectorSearch(keyword_fields={'course'})
vindex.fit(X, documents)

def vector_search_function(q):
    # Transform the query using the same pipeline
    query_vector = pipeline.transform([q['question']])
    
    # Perform vector search
    results = vindex.search(
        query_vector=query_vector[0], 
        filter_dict={"course": q['course']}, 
        num_results=5
    )
    return results

# Evaluate the vector search approach
print("Evaluating vector search...")
vector_results = evaluate(ground_truth, vector_search_function)
print(f"Hit rate: {vector_results['hit_rate']:.2f}")
print(f"MRR: {vector_results['mrr']:.2f}")

# Q3: Vector search for question and answer combined
print("\nQ3: Evaluating vector search with question + text combined...")

# Create texts combining question and text (answer)
texts_combined = []

for doc in documents:
    t = doc['question'] + ' ' + doc['text']
    texts_combined.append(t)

# Use the same pipeline parameters
pipeline_combined = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)
X_combined = pipeline_combined.fit_transform(texts_combined)

print(f"Created combined embeddings with shape: {X_combined.shape}")

# Create VectorSearch index with combined embeddings
vindex_combined = VectorSearch(keyword_fields={'course'})
vindex_combined.fit(X_combined, documents)

def vector_search_combined_function(q):
    # Transform the query using the same pipeline
    query_vector = pipeline_combined.transform([q['question']])
    
    # Perform vector search
    results = vindex_combined.search(
        query_vector=query_vector[0], 
        filter_dict={"course": q['course']}, 
        num_results=5
    )
    return results

# Evaluate the combined vector search approach
print("Evaluating combined vector search...")
vector_combined_results = evaluate(ground_truth, vector_search_combined_function)
print(f"Hit rate: {vector_combined_results['hit_rate']:.2f}")
print(f"MRR: {vector_combined_results['mrr']:.2f}")

# Q4: Qdrant with Jina embeddings
print("\nQ4: Evaluating Qdrant with Jina embeddings...")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

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

