# Fast RAG LLM LangCahin


## Overview

This project implements a FastAPI application that integrates various Retrieval-Augmented Generation (RAG) methods with Large Language Models (LLMs) using the LangChain framework. The application enables document ingestion, retrieval, and query processing with adaptive strategies for different types of queries. It leverages embeddings from OpenAI, HuggingFace, and other models, providing a scalable solution for intelligent document handling and information retrieval.

## Features
**Environment Configuration:** Load environment variables and API keys using dotenv.

**FastAPI Setup:** Set up a robust API with CORS middleware support for cross-origin requests.

**Document Ingestion:** Upload and process documents using various chunking methods, storing them in a Qdrant vector database.

**Adaptive Retrieval Strategies:** Dynamically select the best retrieval strategy based on query characteristics.

**Query Handling:** Process queries using different RAG types and LLMs, generating contextually relevant responses.

**Post-Processing:** Support for post-processing retrieved documents, including context reordering and time-based sorting.

**Web Content Fetching:** Fallback to web search when relevant documents are not found in the database.

**Collection Management:** Utilities to manage Qdrant collections, including listing, dropping, and cleaning collections.

**Error Handling and Logging:** Robust error handling and logging for reliable operation and issue tracking.


## Installation
**Prerequisites**
- Python 3.8+
- pip package manager
- Environment variables for API keys (e.g., OpenAI, Google Custom Search) set in a .env file
  
**Steps**

1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```
3. Set Up Environment Variables

Create a .env file in the root directory and add the necessary environment variables:

```bash
OPENAI_API_KEY=<your-openai-api-key>
GOOGLE_API_KEY=<your-google-api-key>
GOOGLE_CX=<your-google-custom-search-engine-id>
HUGGINGFACEHUB_API_TOKEN=<your-huggingface-api-token>
```

4. Run the FastAPI Application

```bash
uvicorn main:app --reload
```

5. Access the API


Open your web browser and navigate to http://127.0.0.1:8000/docs to explore the API documentation and test endpoints.

