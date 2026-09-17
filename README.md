# GEN_AI-RAG

A lightweight Retrieval-Augmented Generation (RAG) project built for extracting knowledge from PDF documents and answering user questions using LLM-powered retrieval.

This repository demonstrates a typical RAG pipeline using:
- LangChain for orchestration
- PDF document loading
- Text chunking and vectorization
- Chroma vector database
- Google Generative AI embeddings
- Groq-hosted LLM for answer generation

## Overview

The project loads documents from the `Data/` folder, splits them into smaller chunks, converts them into embeddings, stores them in a vector database, and retrieves the most relevant context for a user query before generating a grounded answer.

This is useful for building Q&A systems over:
- research PDFs
- manuals and reports
- knowledge bases stored as documents
- internal documentation repositories

## Repository Structure

```text
GEN_AI-RAG/
├── Data/
│   ├── data_science_syllabus.pdf
│   └── medical_report.pdf
├── .gitignore
├── main.py
├── rag_bot.ipynb
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- LangChain
- LangChain Chroma
- LangChain Groq integration
- LangChain Google Generative AI integration
- PyPDF
- RecursiveCharacterTextSplitter
- Chroma vector store

## Features

- PDF ingestion from local files
- Document chunking for better retrieval
- Semantic search using embeddings
- Context-aware chat/answer generation
- Notebook-based implementation for rapid experimentation
- Easy extension for custom documents or external knowledge bases

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Pranjalsu1/GEN_AI-RAG.git
cd GEN_AI-RAG
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

Create a `.env` file in the project root and add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## How the RAG Pipeline Works

1. Load documents from the `Data/` folder using `PyPDFLoader`
2. Split documents into smaller chunks using `RecursiveCharacterTextSplitter`
3. Generate embeddings using `GoogleGenerativeAIEmbeddings`
4. Store embeddings in Chroma
5. Retrieve the most relevant chunks for a query
6. Pass retrieved context to an LLM (Groq / Gemini) for answer generation

## Example Workflow

The repository includes a notebook, `rag_bot.ipynb`, which demonstrates the full flow:

```python
loader = PyPDFLoader('./Data/medical_report.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splitted_data = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vector_store = Chroma.from_documents(documents=splitted_data, embedding=embeddings)

query = "what is the name of patient?"
result = vector_store.similarity_search(query=query)
```

## Notes

- `main.py` is currently a minimal starter file and can be expanded into a reusable script or application entry point.
- The notebook is the main place where the working RAG logic is demonstrated.
- The project can be extended for use with custom PDFs, API-driven retrieval, or a Streamlit/Flask frontend.

## Future Enhancements

- Add a proper CLI or web interface
- Support multiple document sources (CSV, TXT, DOCX, URLs)
- Add conversation memory for multi-turn Q&A
- Improve chunking and ranking strategies
- Add unit tests and deployment configuration

## License

This project is currently unlicensed unless you add one explicitly.

## Author

Pranjalsu1
