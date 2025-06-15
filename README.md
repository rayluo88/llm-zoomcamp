# LLM Zoomcamp

This repository contains code and exercises from the LLM Zoomcamp course.

## Contents

- `request.py` - Elasticsearch indexing and search functionality for course documents

## Features

- Downloads course documents from GitHub
- Indexes documents in Elasticsearch with proper field mappings
- Performs semantic search with boosting and filtering
- Builds prompts for LLM queries
- Token counting using tiktoken

## Requirements

- Python 3.x
- Elasticsearch 8.x
- Required packages: `requests`, `elasticsearch`, `tiktoken`

## Usage

1. Start Elasticsearch server
2. Run the script: `python3 request.py`

The script will:
- Create an index called "course-questions"
- Index course documents with proper field types
- Perform search queries
- Build prompts for LLM consumption
- Calculate token counts for OpenAI API usage