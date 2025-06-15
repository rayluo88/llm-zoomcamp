import requests 
from elasticsearch import Elasticsearch
import tiktoken

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# Connect to Elasticsearch
es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "ZjdNygQr9i70gtE2qQFk"),
    verify_certs=False
)

# Define the index name
index_name = "course-questions"

# Create index mapping with course as keyword and other fields as text
mapping = {
    "mappings": {
        "properties": {
            "course": {
                "type": "keyword"
            },
            "text": {
                "type": "text"
            },
            "section": {
                "type": "text"
            },
            "question": {
                "type": "text"
            }
        }
    }
}

# Check if index exists, if not create it
if not es.indices.exists(index=index_name):
    print(f"Creating index: {index_name}")
    es.indices.create(index=index_name, body=mapping)
    print("Index created successfully!")
    
    # Index the documents
    print(f"Indexing {len(documents)} documents...")
    for doc in documents:
        es.index(index=index_name, document=doc)
    
    print("All documents indexed successfully!")
else:
    print(f"Index {index_name} already exists")

# Verify the mapping
mapping_response = es.indices.get_mapping(index=index_name)
print(f"Course field type: {mapping_response[index_name]['mappings']['properties']['course']['type']}")
print(f"Text field type: {mapping_response[index_name]['mappings']['properties']['text']['type']}")

# Search query
search_query = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": "How do copy a file to a Docker container?",
                    "fields": ["question^4", "text"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "machine-learning-zoomcamp"
                }
            }
        }
    }
}

print("\n" + "="*50)
print("SEARCH RESULTS")
print("="*50)
print(f"Query: 'How do copy a file to a Docker container?'")
print(f"Fields: question^4, text")
print(f"Type: best_fields")
print(f"Filter: course = machine-learning-zoomcamp")
print(f"Size: 3 results")
print("-"*50)

# Execute the search
response = es.search(index=index_name, body=search_query)

# Display results
for i, hit in enumerate(response['hits']['hits']):
    print(f"\nResult {i+1}:")
    print(f"Score: {hit['_score']}")
    print(f"Course: {hit['_source']['course']}")
    print(f"Section: {hit['_source']['section']}")
    print(f"Question: {hit['_source']['question'][:100]}...")
    if i == 0:  # Show the top result score prominently
        print(f"\n🎯 TOP RANKING RESULT SCORE: {hit['_score']}")
    print("-"*30)

# Build context using the template
context_template = """
Q: {question}
A: {text}
""".strip()

# Create context from search results
context_entries = []
for hit in response['hits']['hits']:
    context_entry = context_template.format(
        question=hit['_source']['question'],
        text=hit['_source']['text']
    )
    context_entries.append(context_entry)

# Join context entries with double linebreaks
context = "\n\n".join(context_entries)

# Build the final prompt
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

# The question we're asking
question = "How do copy a file to a Docker container?"

# Create the final prompt
final_prompt = prompt_template.format(
    question=question,
    context=context
)

print("\n" + "="*60)
print("PROMPT CONSTRUCTION")
print("="*60)
print(f"Question: {question}")
print(f"Number of context entries: {len(context_entries)}")
print(f"Context length: {len(context)} characters")
print(f"Final prompt length: {len(final_prompt)} characters")
print("\n" + "🎯 FINAL PROMPT LENGTH: " + str(len(final_prompt)))
print("="*60)

# Calculate tokens using tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode(final_prompt)
num_tokens = len(tokens)

print(f"\n📊 TOKEN ANALYSIS:")
print(f"Encoding model: gpt-4o")
print(f"Number of tokens: {num_tokens}")
print(f"Characters per token (avg): {len(final_prompt)/num_tokens:.2f}")
print(f"\n🎯 FINAL TOKEN COUNT: {num_tokens}")

# Optionally show the full prompt (uncomment to see)
# print("\nFULL PROMPT:")
# print("-" * 40)
# print(final_prompt)