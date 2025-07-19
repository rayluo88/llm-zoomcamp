import requests
import dlt

def count_zoomcamp_documents():
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    total_count = 0
    course_counts = {}
    
    for course in documents_raw:
        course_name = course['course']
        course_count = len(course['documents'])
        course_counts[course_name] = course_count
        total_count += course_count
        print(f"{course_name}: {course_count} documents")
    
    print(f"\nTotal documents: {total_count}")
    return total_count

if __name__ == "__main__":
    count_zoomcamp_documents() 