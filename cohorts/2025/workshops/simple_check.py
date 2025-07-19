import requests
import json

# First, let's just check the source data
print("Fetching source data...")
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

total_count = 0
print("Course breakdown:")
for course in documents_raw:
    course_name = course['course']
    course_count = len(course['documents'])
    total_count += course_count
    print(f"  {course_name}: {course_count} documents")

print(f"\nTotal documents in source: {total_count}")

# Now let's recreate the pipeline and check the trace
try:
    import dlt
    from dlt.destinations import qdrant
    
    print("\nRecreating the pipeline...")
    
    @dlt.resource
    def zoomcamp_data():
        for course in documents_raw:
            course_name = course['course']
            for doc in course['documents']:
                doc['course'] = course_name
                yield doc
    
    # Use existing destination
    qdrant_destination = qdrant(qd_path="db.qdrant")
    
    # Create pipeline
    pipeline = dlt.pipeline(
        pipeline_name="zoomcamp_pipeline",
        destination=qdrant_destination,
        dataset_name="zoomcamp_tagged_data"
    )
    
    print("Pipeline created. Checking last trace...")
    
    # Since we already ran it, let's just check the trace
    if hasattr(pipeline, '_traces') and pipeline._traces:
        last_trace = pipeline._traces[-1]
        print("Found trace:")
        print(last_trace)
        
        # Look for normalized data info
        if hasattr(last_trace, 'steps'):
            for step in last_trace.steps:
                if hasattr(step, 'info') and 'tables' in str(step.info):
                    print(f"Step info: {step.info}")
    else:
        print("No traces found in existing pipeline")
        
except ImportError:
    print("DLT not available in this environment")
except Exception as e:
    print(f"Error: {e}") 