#!/usr/bin/env python3

"""
Based on the LLM RAG Workshop data structure, let's get the exact count
that would be in the dlt pipeline trace for "Normalized data for the following tables:"
"""

import requests
import sys

def main():
    print("="*60)
    print("DLT Pipeline Analysis - Zoomcamp Data")
    print("="*60)
    
    # Fetch the exact data that the dlt pipeline would process
    print("Fetching data from the workshop repository...")
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    
    try:
        docs_response = requests.get(docs_url)
        docs_response.raise_for_status()
        documents_raw = docs_response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Count documents exactly as the dlt resource would process them
    total_documents = 0
    course_breakdown = {}
    
    print("\nProcessing courses:")
    print("-" * 40)
    
    for course in documents_raw:
        course_name = course['course']
        course_doc_count = len(course['documents'])
        course_breakdown[course_name] = course_doc_count
        total_documents += course_doc_count
        print(f"{course_name:30} {course_doc_count:4d} documents")
    
    print("-" * 40)
    print(f"{'TOTAL DOCUMENTS':30} {total_documents:4d}")
    print("="*60)
    
    print("\nAnswer for Question 2:")
    print(f"The number of rows inserted into the zoomcamp_data collection is: {total_documents}")
    print("\nThis number would appear in the dlt pipeline trace under:")
    print("'Normalized data for the following tables: zoomcamp_data ({total_documents} items)'")
    
    return total_documents

if __name__ == "__main__":
    main() 