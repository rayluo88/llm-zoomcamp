import requests
import dlt
from dlt.destinations import qdrant

@dlt.resource
def zoomcamp_data():
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    count = 0
    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            count += 1
            yield doc
    
    print(f"Total documents yielded: {count}")

# Define the Qdrant destination
qdrant_destination = qdrant(
    qd_path="db.qdrant", 
)

# Create the pipeline
pipeline = dlt.pipeline(
    pipeline_name="zoomcamp_pipeline",
    destination=qdrant_destination,
    dataset_name="zoomcamp_tagged_data"
)

# Run the pipeline
print("Running pipeline...")
load_info = pipeline.run(zoomcamp_data())

# Get and save the trace
trace = pipeline.last_trace
print("Pipeline trace:")
print(trace)

# Save trace to file
with open("pipeline_trace.txt", "w") as f:
    f.write(str(trace))

print(f"Load info: {load_info}") 