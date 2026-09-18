# GEN_AI-RAG

A lightweight Retrieval-Augmented Generation (RAG) project that uses an agentic workflow to answer questions from PDF-based medical reports. The project combines LangChain, Google Gemini models, document embeddings, in-memory vector search, and a tool-calling agent.

## Overview

The repository contains a Jupyter Notebook implementation of an agentic RAG chatbot. The workflow:

1. Loads a medical PDF from the `Data/` directory.
2. Splits the document into overlapping chunks.
3. Generates vector embeddings for each chunk with Google Gemini.
4. Stores the embeddings in an in-memory vector store.
5. Exposes similarity search through a LangChain tool.
6. Uses a Gemini-powered agent to decide when to retrieve information and formulate an answer.

The current notebook uses `Data/medical_report.pdf` as its knowledge source and demonstrates retrieving patient and doctor details from the report.

## Repository Structure

```text
GEN_AI-RAG/
├── Data/
│   ├── data_science_syllabus.pdf
│   └── medical_report.pdf
├── .gitignore
├── main.py
├── rag_bot.ipynb
├── rag_agentic_bot.ipynb
├── requirements.txt
└── README.md
```

## Tech Stack

- Python 3.13+
- Jupyter Notebook
- LangChain
- `langchain-google-genai`
- `langchain-community`
- `langchain-text-splitters`
- PyPDF
- Google Gemini chat and embedding models
- `InMemoryVectorStore`

## Features

- PDF ingestion with `PyPDFLoader`
- Recursive document chunking
- Semantic similarity search with Gemini embeddings
- A custom `retriever_tool` for accessing PDF context
- Agentic tool calling with LangChain's `create_agent`
- Gemini-powered answers grounded in retrieved document content
- Notebook-based experimentation with local PDF files

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Pranjalsu1/GEN_AI-RAG.git
cd GEN_AI-RAG
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

3. Install the project dependencies:

```bash
pip install -r requirements.txt
```

4. Configure your Gemini API key.

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

The notebook loads this value with `python-dotenv`. Do not commit real API keys to the repository.

## Running the Agentic RAG Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open [`rag_agentic_bot.ipynb`](./rag_agentic_bot.ipynb) and run the cells in order. The notebook currently loads:

```python
loader = PyPDFLoader("./Data/medical_report.pdf")
docs = loader.load()
```

To use another PDF, update the path passed to `PyPDFLoader`.

## How the Agentic RAG Pipeline Works

### 1. Load and split the document

The notebook loads the PDF and divides it into chunks of 1,000 characters with 200 characters of overlap:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splitted_data = splitter.split_documents(docs)
```

### 2. Create the vector store

Each chunk is embedded with the Gemini embedding model and stored in memory:

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)
vector_store = InMemoryVectorStore.from_documents(
    documents=splitted_data,
    embedding=embeddings,
)
```

The in-memory store is convenient for experimentation, but its contents are lost when the notebook process ends.

### 3. Define the retrieval tool

The agent accesses the document through a custom LangChain tool. The tool performs a similarity search and returns the most relevant results:

```python
@tool
def retriever_tool(query: str):
    docs = vector_store.similarity_search(query=query, k=4)
    return "\n\n".join(doc.page_content for doc in docs)
```

### 4. Create the agent

The notebook initializes a Gemini chat model and instructs the agent to use the retriever for questions requiring information from the PDF:

```python
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt=(
        "You are a helpful assistant that answers questions using retrieved context. "
        "Always use the retriever_tool for questions requiring external knowledge."
    ),
)
```

### 5. Ask a question

```python
query = "What is the name of patient, and what is the name of Doctors"
response = agent.invoke({
    "messages": [{"role": "user", "content": query}]
})
result = response["messages"][-1].content
print(result)
```

The agent may call the retrieval tool one or more times before producing a grounded response.

## Example Result

For the sample medical report, the notebook demonstrates an answer containing information such as:

- Patient name: Ms. Nikita Chudhary
- Referring physician: Dr. Nitin Nahar

Results depend on the contents of the selected PDF and the model's response.

## Security and Privacy

This project processes medical-report data. Use synthetic or properly anonymized documents when experimenting, and avoid sharing confidential patient information with external model APIs. Keep API keys and sensitive documents out of version control.

## Notes

- `rag_agentic_bot.ipynb` is the current agentic RAG implementation.
- `rag_bot.ipynb` contains the earlier notebook-based RAG workflow.
- `InMemoryVectorStore` is not persistent; a persistent vector database can be added for production use.
- Model names and package APIs may change over time. If a model is unavailable, select a currently supported Gemini model in the notebook.
- The retrieval tool should return all retrieved document contents when combining multiple matches; this is important for preserving context during retrieval.

## Future Enhancements

- Persist embeddings in a production-ready vector database.
- Add support for multiple PDF files and document collections.
- Add conversation memory for multi-turn questions.
- Improve source attribution and citation display.
- Add evaluation tests for retrieval and answer quality.
- Provide a CLI, Streamlit, or web-based interface.
- Add document redaction and stronger privacy controls.

## License

This project is currently unlicensed unless a license is added explicitly.

## Author

Pranjalsu1
