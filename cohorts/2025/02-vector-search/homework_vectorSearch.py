from fastembed import TextEmbedding
import numpy as np
import requests

# Initialize the embedding model
model = TextEmbedding("jinaai/jina-embeddings-v2-small-en")

# Define the query
query = "I just discovered the course. Can I join now?"

# Generate embeddings
embeddings = list(model.embed([query]))

# Convert to numpy array
embedding_array = np.array(embeddings[0])

# Print array info
print(f"Array shape: {embedding_array.shape}")
print(f"Array size: {embedding_array.size}")

# Find the minimal value
min_value = np.min(embedding_array)
print(f"Minimal value in the array: {min_value}")

print("\n" + "="*50)
print("VECTOR NORMALIZATION VERIFICATION")
print("="*50)

# Check if the vector is normalized (length should be 1.0)
q = embedding_array
vector_norm = np.linalg.norm(q)
print(f"Vector norm (length): {vector_norm}")
print(f"Is vector normalized? {np.isclose(vector_norm, 1.0)}")

# Compute dot product with itself (should be 1.0 for normalized vectors)
dot_product_self = q.dot(q)
print(f"Dot product with itself: {dot_product_self}")
print(f"Is dot product ~1.0? {np.isclose(dot_product_self, 1.0)}")

print(f"\nNote: Since the vector is normalized, dot product = cosine similarity")
print(f"Cosine similarity of query with itself: {dot_product_self}")

print("\n" + "="*50)
print("COSINE SIMILARITY WITH ANOTHER VECTOR")
print("="*50)

# Define the document
doc = 'Can I still join the course after the start date?'
print(f"Document: {doc}")

# Generate embeddings for the document
doc_embeddings = list(model.embed([doc]))
doc_vector = np.array(doc_embeddings[0])

# Verify document vector is also normalized
doc_norm = np.linalg.norm(doc_vector)
print(f"\nDocument vector norm: {doc_norm}")
print(f"Is document vector normalized? {np.isclose(doc_norm, 1.0)}")

# Calculate cosine similarity between query and document
cosine_similarity = q.dot(doc_vector)
print(f"\nCosine similarity between query and document: {cosine_similarity}")

# Round to 1 decimal place to match the given options
rounded_similarity = round(cosine_similarity, 1)
print(f"Rounded cosine similarity: {rounded_similarity}")

# Check which option it matches
options = [0.3, 0.5, 0.7, 0.9]
print(f"\nGiven options: {options}")
print(f"Closest option: {min(options, key=lambda x: abs(x - cosine_similarity))}")

print("\n" + "="*50)
print("Q3: RANKING BY COSINE SIMILARITY")
print("="*50)

# Define the documents
documents = [
    {'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
     'section': 'General course-related questions',
     'question': 'Course - Can I still join the course after the start date?',
     'course': 'data-engineering-zoomcamp'},
    {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
     'section': 'General course-related questions',
     'question': 'Course - Can I follow the course after it finishes?',
     'course': 'data-engineering-zoomcamp'},
    {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first \"Office Hours\" live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon't forget to register in DataTalks.Club's Slack and join the channel.",
     'section': 'General course-related questions',
     'question': 'Course - When will the course start?',
     'course': 'data-engineering-zoomcamp'},
    {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
     'section': 'General course-related questions',
     'question': 'Course - What can I do before the course starts?',
     'course': 'data-engineering-zoomcamp'},
    {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
     'section': 'General course-related questions',
     'question': 'How can we contribute to the course?',
     'course': 'data-engineering-zoomcamp'}
]

print(f"Number of documents: {len(documents)}")

# Extract text fields from documents
document_texts = [doc['text'] for doc in documents]

# Generate embeddings for all documents
print("Generating embeddings for all documents...")
all_doc_embeddings = list(model.embed(document_texts))

# Convert to numpy matrix V where each row is a document embedding
V = np.array(all_doc_embeddings)
print(f"Document embeddings matrix shape: {V.shape}")

# Compute cosine similarities using matrix multiplication
cosine_similarities = V.dot(q)
print(f"Cosine similarities shape: {cosine_similarities.shape}")

print("\nCosine similarities for each document:")
for i, similarity in enumerate(cosine_similarities):
    print(f"Document {i}: {similarity:.6f}")
    print(f"  Question: {documents[i]['question']}")
    print(f"  Text preview: {documents[i]['text'][:100]}...")
    print()

# Find the document with highest similarity
max_similarity_index = np.argmax(cosine_similarities)
max_similarity_value = cosine_similarities[max_similarity_index]

print(f"Document with highest similarity:")
print(f"Index: {max_similarity_index}")
print(f"Similarity: {max_similarity_value:.6f}")
print(f"Question: {documents[max_similarity_index]['question']}")

print(f"\nQ3 Answer: {max_similarity_index}")

print("\n" + "="*50)
print("Q4: RANKING BY COSINE SIMILARITY - VERSION TWO")
print("="*50)

# Create concatenated text field (question + text)
full_texts = []
for doc in documents:
    full_text = doc['question'] + ' ' + doc['text']
    full_texts.append(full_text)

print("Sample concatenated texts:")
for i, full_text in enumerate(full_texts):
    print(f"Document {i}:")
    print(f"  Full text preview: {full_text[:150]}...")
    print()

# Generate embeddings for concatenated texts
print("Generating embeddings for concatenated texts...")
full_text_embeddings = list(model.embed(full_texts))

# Convert to numpy matrix
V_full = np.array(full_text_embeddings)
print(f"Full text embeddings matrix shape: {V_full.shape}")

# Compute cosine similarities
cosine_similarities_full = V_full.dot(q)
print(f"Full text cosine similarities shape: {cosine_similarities_full.shape}")

print("\nCosine similarities for each document (with concatenated text):")
for i, similarity in enumerate(cosine_similarities_full):
    print(f"Document {i}: {similarity:.6f}")
    print(f"  Question: {documents[i]['question']}")
    print()

# Find the document with highest similarity
max_similarity_index_full = np.argmax(cosine_similarities_full)
max_similarity_value_full = cosine_similarities_full[max_similarity_index_full]

print(f"Document with highest similarity (full text):")
print(f"Index: {max_similarity_index_full}")
print(f"Similarity: {max_similarity_value_full:.6f}")
print(f"Question: {documents[max_similarity_index_full]['question']}")

print(f"\nQ4 Answer: {max_similarity_index_full}")

print("\n" + "="*50)
print("COMPARISON BETWEEN Q3 AND Q4")
print("="*50)

print(f"Q3 (text only) - Highest similarity: Document {max_similarity_index} (similarity: {max_similarity_value:.6f})")
print(f"Q4 (question + text) - Highest similarity: Document {max_similarity_index_full} (similarity: {max_similarity_value_full:.6f})")

if max_similarity_index != max_similarity_index_full:
    print(f"\n✅ DIFFERENT RESULTS!")
    print(f"Q3: Document {max_similarity_index} - {documents[max_similarity_index]['question']}")
    print(f"Q4: Document {max_similarity_index_full} - {documents[max_similarity_index_full]['question']}")
    print(f"\nWhy different?")
    print(f"- Q3 only used the 'text' field for embedding")
    print(f"- Q4 used 'question + text' for embedding")
    print(f"- The concatenated version includes the question, which may be more semantically")
    print(f"  similar to the query, affecting the overall similarity score")
    print(f"- Question texts often contain key terms that match user queries better than answers")
else:
    print(f"\n⚠️ SAME RESULTS!")
    print(f"Both Q3 and Q4 resulted in Document {max_similarity_index}")
    print(f"The question field didn't change the ranking in this case")

print("\n" + "="*50)
print("Q5: SELECTING THE EMBEDDING MODEL")
print("="*50)

# List available models
print("Available embedding models in FastEmbed:")
try:
    supported_models = TextEmbedding.list_supported_models()
    
    # Filter and show models with their dimensions
    model_dimensions = {}
    print("\nModel dimensions found:")
    
    for model_info in supported_models:
        model_name = model_info['model']
        # Try to get dimension info from model description/size
        if 'dim' in model_info:
            dim = model_info['dim']
        elif 'size' in model_info:
            dim = model_info['size']
        else:
            dim = "Unknown"
        
        model_dimensions[model_name] = dim
        print(f"{model_name}: {dim} dimensions")
        
        # Show first 10 models to avoid too much output
        if len(model_dimensions) >= 10:
            print("... (showing first 10 models)")
            break
            
except Exception as e:
    print(f"Could not list models: {e}")
    print("Will test specific models directly...")

print("\n" + "-"*30)
print("Testing BAAI/bge-small-en model:")

# Test the specific model mentioned
try:
    bge_small_model = TextEmbedding("BAAI/bge-small-en")
    test_text = "Test embedding"
    bge_embedding = list(bge_small_model.embed([test_text]))
    bge_array = np.array(bge_embedding[0])
    
    print(f"BAAI/bge-small-en dimensions: {bge_array.shape[0]}")
    print(f"Shape: {bge_array.shape}")
    
    # Test a few other common models to find dimensions
    print("\nTesting other models for dimension comparison:")
    
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-base-en",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    
    for model_name in models_to_test:
        try:
            test_model = TextEmbedding(model_name)
            test_emb = list(test_model.embed([test_text]))
            test_arr = np.array(test_emb[0])
            print(f"{model_name}: {test_arr.shape[0]} dimensions")
        except Exception as e:
            print(f"{model_name}: Error - {e}")
            
except Exception as e:
    print(f"Error testing BAAI/bge-small-en: {e}")

print("\nGiven options: 128, 256, 384, 512")
print("Based on the test results above, the smallest dimensionality should be one of these values.")

print("\n" + "="*50)
print("Q6: INDEXING WITH QDRANT")
print("="*50)

# Download ML Zoomcamp documents
print("Downloading ML Zoomcamp documents...")
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

# Filter for machine-learning-zoomcamp
ml_documents = []
for course in documents_raw:
    course_name = course['course']
    if course_name != 'machine-learning-zoomcamp':
        continue
    
    for doc in course['documents']:
        doc['course'] = course_name
        ml_documents.append(doc)

print(f"Found {len(ml_documents)} ML Zoomcamp documents")

# Show a sample document structure
if ml_documents:
    print(f"\nSample document structure:")
    sample_doc = ml_documents[0]
    for key, value in sample_doc.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

# Initialize the smaller embedding model for Q6
print(f"\nInitializing BAAI/bge-small-en model for Q6...")
small_model = TextEmbedding("BAAI/bge-small-en")

# Prepare texts for embedding (question + text)
print("Preparing texts for embedding...")
ml_texts = []
for doc in ml_documents:
    text = doc['question'] + ' ' + doc['text']
    ml_texts.append(text)

print(f"Prepared {len(ml_texts)} texts for embedding")

# Set up Qdrant (in-memory for this example)
print("\nSetting up Qdrant...")
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    
    # Create in-memory Qdrant client
    client = QdrantClient(":memory:")
    
    # Create collection
    collection_name = "ml_zoomcamp_faq"
    
    # Get vector dimension from the small model
    test_embedding = list(small_model.embed(["test"]))
    vector_dim = len(test_embedding[0])
    
    print(f"Vector dimension: {vector_dim}")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    
    print(f"Created collection '{collection_name}' with {vector_dim} dimensions")
    
    # Generate embeddings and insert into Qdrant
    print("Generating embeddings and inserting into Qdrant...")
    
    # Process in batches to avoid memory issues
    batch_size = 50
    total_docs = len(ml_documents)
    
    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        batch_texts = ml_texts[i:batch_end]
        batch_docs = ml_documents[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} (docs {i}-{batch_end-1})")
        
        # Generate embeddings for batch
        batch_embeddings = list(small_model.embed(batch_texts))
        
        # Prepare points for insertion
        points = []
        for j, (embedding, doc) in enumerate(zip(batch_embeddings, batch_docs)):
            point_id = i + j
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "question": doc["question"],
                    "text": doc["text"],
                    "section": doc["section"],
                    "course": doc["course"],
                    "full_text": batch_texts[j]
                }
            ))
        
        # Insert batch into Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    print(f"Successfully inserted {total_docs} documents into Qdrant")
    
    # Query with the question from Q1
    query_text = "I just discovered the course. Can I join now?"
    print(f"\nQuerying with: '{query_text}'")
    
    # Generate embedding for query
    query_embedding = list(small_model.embed([query_text]))[0]
    
    # Search in Qdrant
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=5
    )
    
    print(f"\nTop 5 search results:")
    for i, result in enumerate(search_results):
        score = result.score
        question = result.payload.get("question", "N/A")
        text_preview = result.payload.get("text", "")[:100]
        
        print(f"\nResult {i+1}:")
        print(f"  Score: {score:.6f}")
        print(f"  Question: {question}")
        print(f"  Text preview: {text_preview}...")
    
    # Get the highest score
    if search_results:
        highest_score = search_results[0].score
        print(f"\nHighest score (first result): {highest_score:.6f}")
        
        # Round to 2 decimal places and find closest option
        rounded_score = round(highest_score, 2)
        score_options = [0.97, 0.87, 0.77, 0.67]
        closest_option = min(score_options, key=lambda x: abs(x - highest_score))
        
        print(f"Rounded score: {rounded_score}")
        print(f"Given options: {score_options}")
        print(f"Closest option: {closest_option}")
        print(f"\nQ6 Answer: {closest_option}")
    
except ImportError:
    print("Qdrant client not installed. Installing...")
    import subprocess
    subprocess.run(["pip", "install", "qdrant-client"], check=True)
    print("Please run the script again after installation.")
except Exception as e:
    print(f"Error with Qdrant: {e}")
    print("Make sure qdrant-client is installed: pip install qdrant-client")

