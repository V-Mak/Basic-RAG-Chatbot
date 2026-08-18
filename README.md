# Basic RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG) chatbot** that answers questions from a provided PDF document. The project combines **document retrieval using FAISS** with a lightweight **Hugging Face language model** to generate context-aware answers.

## Overview

This project demonstrates the basic RAG pipeline:

**PDF → Text Extraction → Text Chunking → Embeddings → FAISS Vector Store → Similarity Retrieval → LLM → Answer**

Instead of asking the language model to answer only from its pretrained knowledge, the chatbot first retrieves relevant information from the provided document and uses that information as context for generating the response.

## Features

* Load and process PDF documents
* Split documents into smaller text chunks
* Generate vector embeddings using Sentence Transformers
* Store embeddings in a FAISS vector database
* Retrieve relevant document chunks based on the user's query
* Generate answers using a Hugging Face instruction-tuned LLM
* Run completely with locally available/open-source models

## Tech Stack

* **Python**
* **LangChain**
* **Hugging Face Transformers**
* **Sentence Transformers**
* **FAISS**
* **PyTorch**
* **PyPDF**
* **Accelerate**

## Models Used

### Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`

Used to convert document chunks and user queries into numerical vector representations.

### Language Model

`Qwen/Qwen2.5-0.5B-Instruct`

Used to generate the final answer using the retrieved document context.

## Project Structure

```text
RAG-chatbot-basic/
│
├── data/
│   └── Python for Probability, Statistics, and Machine Learning.pdf
│
├── vector_store/
│   ├── index.faiss
│   └── index.pkl
│
├── app.py
├── build_index.py
├── requirements.txt
├── .gitignore
└── README.md
```

> `venv/` is used locally for the Python virtual environment and should not be uploaded to GitHub.

## How It Works

### 1. Document Loading

The PDF document is loaded using `PyPDFLoader`.

### 2. Text Chunking

The extracted text is divided into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

The current configuration uses:

* Chunk size: **1000 characters**
* Chunk overlap: **200 characters**

The overlap helps preserve context between neighboring chunks.

### 3. Embedding Generation

Each text chunk is converted into a vector using:

`sentence-transformers/all-MiniLM-L6-v2`

These vectors represent the semantic meaning of the document chunks.

### 4. FAISS Vector Store

The generated embeddings are stored in a **FAISS** index.

FAISS enables efficient similarity search between the user's question and the stored document vectors.

### 5. Retrieval

When a user asks a question:

1. The question is converted into an embedding.
2. FAISS searches for semantically similar document chunks.
3. The most relevant chunks are retrieved.

### 6. Answer Generation

The retrieved chunks are provided as context to:

`Qwen/Qwen2.5-0.5B-Instruct`

The model then generates an answer based on the retrieved information.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/V-Mak/basic-rag-chatbot.git
cd basic-rag-chatbot
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Build the Vector Store

Before running the chatbot, create the FAISS index from the PDF:

```powershell
python build_index.py
```

This creates the `vector_store/` directory containing the FAISS index.

## Run the Chatbot

After creating the vector store:

```powershell
python app.py
```

You can then enter questions related to the information contained in the PDF.

## Example

**User:**

```text
What is probability?
```

**RAG Pipeline:**

```text
User Query
    ↓
Query Embedding
    ↓
FAISS Similarity Search
    ↓
Relevant Document Chunks
    ↓
Context + Query
    ↓
Qwen2.5-0.5B-Instruct
    ↓
Generated Answer
```

## Limitations

This is a **basic RAG implementation** intended for learning and demonstrating the core RAG workflow.

Current limitations include:

* Uses a single document
* Basic similarity-based retrieval
* No reranking
* No query rewriting
* No hybrid search
* No conversational memory
* No advanced retrieval evaluation
* Small language model with limited generation capability


## Learning Objective

This project was built to understand the fundamental components of a **Retrieval-Augmented Generation system**, including document processing, chunking, embeddings, vector databases, similarity search, retrieval, and LLM-based generation.

## Author

**Vivek Makwana**

* LinkedIn: https://www.linkedin.com/in/vivek-makwana-2a7796243
