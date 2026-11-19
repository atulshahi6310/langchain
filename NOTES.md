# 🦜🔗 LangChain Complete Learning Notes & Comprehensive Study Guide

> **A production-grade reference, conceptual breakdown, and hands-on guide covering Document Loaders, Text Splitters, Embeddings, Vector Stores, LCEL, RAG Pipelines, LangSmith Observability, Local Ollama Models, and Streamlit Web Applications.**

---

## 📑 Table of Contents
1. [Introduction to LangChain & RAG Architecture](#chapter-1-introduction-to-langchain--rag-architecture)
2. [Document Loaders & Data Ingestion](#chapter-2-document-loaders--data-ingestion)
3. [Text Splitting Techniques](#chapter-3-text-splitting-techniques)
4. [Embeddings & Vector Representations](#chapter-4-embeddings--vector-representations)
5. [Vector Stores (Chroma & FAISS)](#chapter-5-vector-stores)
6. [Prompts, Output Parsers & LangChain Expression Language (LCEL)](#chapter-6-prompts-output-parsers--lcel)
7. [Building End-to-End RAG Pipelines](#chapter-7-building-end-to-end-rag-pipelines)
8. [Observability, Tracing & Cost Control with LangSmith](#chapter-8-observability-tracing--cost-control-with-langsmith)
9. [Local Models (Ollama) & Interactive Streamlit Web Apps](#chapter-9-local-models-ollama--interactive-streamlit-web-apps)
10. [Notebook & Python Script (.ipynb & .py) Codebase Mapping](#chapter-10-notebook--python-script-ipynb--py-codebase-mapping)
11. [Quick-Start Cheat Sheet & Production Reference](#chapter-11-quick-start-cheat-sheet--production-reference)

---

## Chapter 1: Introduction to LangChain & RAG Architecture

### What is LangChain?
**LangChain** is an open-source development framework designed to streamline the lifecycle of applications powered by Large Language Models (LLMs). It abstracts away low-level API calls and provides reusable, modular components:
- **Components:** Modular abstractions for working with LLMs, prompt templates, indexes, and tools.
- **LCEL (LangChain Expression Language):** A declarative, composable syntax using the Unix pipe (`|`) operator to connect components seamlessly.
- **Chains & Agents:** Structured cognitive architectures allowing models to reason, retrieve context, and interact with external systems.

### The RAG Pipeline Architecture
**Retrieval-Augmented Generation (RAG)** bridges the gap between static LLM training data and private, enterprise, or dynamically changing datasets. By injecting retrieved domain knowledge directly into the LLM context prompt, RAG reduces hallucinations and grounds model responses in truth.

```
┌─────────────────┐
│   Data Sources  │ ──> [ .txt, .pdf, Web URLs, Wikipedia, ArXiv, JSON ]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Loaders│ ──> Standardizes sources into LangChain `Document` objects
└────────┬────────┘     (page_content + metadata)
         │
         ▼
┌─────────────────┐
│  Text Splitters │ ──> Breaks text into semantically coherent, overlapping chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Embedding Models │ ──> Transforms chunks into dense high-dimensional vectors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Stores  │ ──> Indexes vectors (Chroma, FAISS) for similarity search
└────────┬────────┘
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
┌───────────────────┐                     ┌──────────────────┐
│  User Query (In)  │ ──> [ Embed Query ] ──> [ Vector Search│
└───────────────────┘                                │
                                                     ▼ Top-k Chunks
                                          ┌──────────────────┐
                                          │ Context + Prompt │
                                          └──────────┬───────┘
                                                     │
                                                     ▼
                                          ┌──────────────────┐
                                          │    LLM Engine    │
                                          └──────────┬───────┘
                                                     │
                                                     ▼
                                          ┌──────────────────┐
                                          │ Final Answer (Out│
                                          └──────────────────┘
```

### Repository Coverage Matrix

| Component | Modules & Techniques | Tracked Python Script (`.py`) |
| :--- | :--- | :--- |
| **OpenAI & Basics** | ChatOpenAI, GPT-4o, PromptTemplates, StrOutputParser | `1.1-openai/1.1.1-GettingStarted.py` |
| **RAG on Web Docs** | WebBaseLoader, FAISS, Retrieval Chains | `1.1-openai/1.1.2-Simpleapp.py` |
| **Local LLM App** | Ollama local models, Streamlit web UI | `1.2-ollama/1.2.1-Simpleapp.py`, `1.2-ollama/app.py` |
| **Document Loaders** | TextLoader, PyPDFLoader, WebBaseLoader, Arxiv | `dataintegratiion/langchain-1.py` |
| **Wikipedia Loading** | WikipediaLoader query API | `dataintegratiion/wikipdia.py` |
| **LCEL & Parsers** | ChatPromptTemplate, StrOutputParser, LCEL pipes | `dataintegratiion/components.py` |
| **Text Splitters** | RecursiveCharacter, Character, HTMLHeader, JsonSplitter | `text-spliting/*.py` |
| **Embeddings** | OpenAI, Ollama (`nomic-embed-text`), HuggingFace | `embedding/*.py` |
| **Vector Databases** | Chroma DB (persistent), FAISS (in-memory & disk) | `vector_store/*/*.py` |
| **Full RAG Project** | Complete end-to-end RAG with Virat Kohli dataset | `project1/projects.py` |
| **Streamlit Web RAG** | ChatOllama, Streamlit web application | `project1/project/app.py` |

---

## Chapter 2: Document Loaders & Data Ingestion

Document loaders extract unstructured data from raw formats and normalize it into a unified LangChain `Document` object:

```python
class Document:
    page_content: str  # The raw extracted textual content
    metadata: dict     # Source provenance (file path, URL, author, page number, etc.)
```

---

### 2.1 `TextLoader` — Local Plain Text Files
Reads plain `.txt` files directly into memory:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("linear.txt", encoding="utf-8")
docs = loader.load()

print(f"Loaded {len(docs)} document.")
print("Sample content:", docs[0].page_content[:150])
print("Metadata:", docs[0].metadata)  # {'source': 'linear.txt'}
```

---

### 2.2 `PyPDFLoader` — PDF Documents
Parses PDF files page by page. Each page is instantiated as a distinct `Document`, preserving the page index:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("a.pdf")
documents = loader.load()

print(f"Total Pages: {len(documents)}")
print(f"Page 1 Content:\n{documents[0].page_content[:200]}")
print(f"Page 1 Metadata: {documents[0].metadata}")
# Output: {'source': 'a.pdf', 'page': 0}
```

> 💡 **Best Practice:** If extracting text for downstream storage, loop through the documents and write pages into clean text format:
```python
with open("web_doc.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.page_content + "\n\n")
```

---

### 2.3 `WebBaseLoader` — HTML & Web Scraping
Leverages `urllib` and `BeautifulSoup` (`bs4`) to scrape web pages. Supports `bs_kwargs` with `SoupStrainer` to parse specific DOM classes or HTML elements:

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path="https://en.wikipedia.org/wiki/Linear_regression",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("mw-body-content", "mw-content-ltr mw-parser-output")
        )
    )
)
docs = loader.load()

# Normalize excessive whitespace
for doc in docs:
    doc.page_content = " ".join(doc.page_content.split())

print("Scraped length:", len(docs[0].page_content))
```

---

### 2.4 `WikipediaLoader` — Direct Wikipedia API
Queries the Wikipedia search API and extracts full text along with page summaries and metadata:

```python
import wikipedia
from langchain_community.document_loaders import WikipediaLoader

# Always assign a custom user agent to prevent HTTP 403 Forbidden
wikipedia.set_user_agent("LangChainEducationApp/1.0 (contact@example.com)")

loader = WikipediaLoader(query="Virat Kohli", load_max_docs=2)
docs = loader.load()

print("Title:", docs[0].metadata.get("title"))
print("Source URL:", docs[0].metadata.get("source"))
print("Summary Preview:", docs[0].page_content[:300])
```

---

### 2.5 `arxiv.Client` & Academic Research Papers
Fetches academic papers directly from the ArXiv API and encapsulates research abstracts and metadata into LangChain `Document` objects:

```python
import arxiv
from langchain_core.documents import Document

paper_id = "2606.11153"
client = arxiv.Client()
search = arxiv.Search(id_list=[paper_id])

docs = []
for paper in client.results(search):
    docs.append(
        Document(
            page_content=paper.summary,
            metadata={
                "Title": paper.title,
                "Authors": [author.name for author in paper.authors],
                "Published": str(paper.published),
                "Entry ID": paper.entry_id,
                "PDF URL": paper.pdf_url,
                "Primary Category": paper.primary_category,
                "Categories": paper.categories,
            }
        )
    )

print(f"Fetched {len(docs)} paper(s).")
print("Paper Title:", docs[0].metadata["Title"])
print("Abstract:", docs[0].page_content[:200])
```

---

## Chapter 3: Text Splitting Techniques

Large documents must be divided into smaller chunks because:
1. **Context Window Limits:** LLMs have finite input token allowances.
2. **Retrieval Precision:** Searching for fine-grained answers within 500-word chunks yields higher semantic similarity than searching across an entire 50-page book.
3. **Cost & Latency:** Smaller contextual prompts reduce latency and token consumption.

### Key Parameters
- `chunk_size`: The maximum character (or token) count per chunk.
- `chunk_overlap`: The number of overlapping characters between consecutive chunks. Overlap prevents information loss when critical facts land on chunk borders.

---

### 3.1 `RecursiveCharacterTextSplitter` (⭐ Industry Standard Default)
Attempts to split on hierarchical delimiters: `["\n\n", "\n", " ", ""]`. Keeps paragraphs, sentences, and words together before resorting to raw character splits:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(docs)
print(f"Divided {len(docs)} documents into {len(chunks)} chunks.")
```

---

### 3.2 `CharacterTextSplitter`
Splits strictly on a single defined delimiter (e.g. `\n\n`):

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
```

> ⚠️ **Important:** `CharacterTextSplitter` will not force-split paragraphs that exceed `chunk_size` unless secondary delimiters are provided. Use `RecursiveCharacterTextSplitter` unless you have explicit strict separator rules.

---

### 3.3 `HTMLHeaderTextSplitter`
Preserves document structure by splitting HTML text based on header tags (`<h1>`, `<h2>`, `<h3>`). The header text is preserved in the chunk's `metadata`:

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html>
<body>
    <h1>Deep Learning Basics</h1>
    <p>Neural networks form the backbone of modern deep learning.</p>
    <h2>Convolutional Neural Networks</h2>
    <p>CNNs are specialized for computer vision and image processing tasks.</p>
    <h2>Recurrent Neural Networks</h2>
    <p>RNNs and LSTMs process sequential data like time series and text.</p>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)

for split in html_header_splits:
    print("Content:", split.page_content)
    print("Metadata:", split.metadata)
```

---

### 3.4 `RecursiveJsonSplitter`
Splits nested JSON data structures into chunks while ensuring valid JSON hierarchy in each chunk:

```python
import json
from langchain_text_splitters import RecursiveJsonSplitter

json_data = {
    "company": "TechCorp",
    "departments": [
        {"name": "AI Research", "headcount": 50, "projects": ["RAG", "Autonomous Agents"]},
        {"name": "Cloud Infrastructure", "headcount": 120, "projects": ["Kubernetes Migration"]}
    ]
}

json_splitter = RecursiveJsonSplitter(max_chunk_size=150)
json_chunks = json_splitter.split_json(json_data)
docs = json_splitter.create_documents(texts=json_chunks)

for doc in docs:
    print("JSON Chunk:\n", doc.page_content)
```

---

## Chapter 4: Embeddings & Vector Representations

Embeddings transform unstructured text strings into numerical vectors in a high-dimensional continuous space. The distance between vectors (e.g., Cosine Similarity) represents semantic similarity.

$$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

---

### 4.1 OpenAI Embeddings
High-dimensional dense vectors powered by OpenAI's embedding models:

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 1536 dimensions default, can be reduced to 512/1024
    dimensions=1024
)

# Embedding a single search query
query_vector = embeddings.embed_query("What is vector similarity search?")
print(f"Query Vector Dimensions: {len(query_vector)}")

# Embedding multiple document chunks in batches
doc_vectors = embeddings.embed_documents([
    "LangChain simplifies LLM application development.",
    "FAISS is an efficient similarity search library."
])
print(f"Embedded {len(doc_vectors)} documents.")
```

---

### 4.2 Local Embeddings with Ollama (`nomic-embed-text`)
Free, open-source embeddings running entirely on your local machine:

```python
from langchain_ollama import OllamaEmbeddings
# Or: from langchain_community.embeddings import OllamaEmbeddings

# Requires Ollama running locally: `ollama run nomic-embed-text`
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector = embeddings.embed_query("Local AI embeddings with Ollama")
print(f"Local Embedding Dimensions: {len(vector)}")
```

---

### 4.3 Hugging Face Embeddings (In-Memory PyTorch / Transformers)
Runs state-of-the-art HuggingFace Sentence Transformers locally:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectors = embeddings.embed_documents(["First chunk", "Second chunk"])
print(f"HuggingFace Vector Dimensions: {len(vectors[0])}")
```

---

## Chapter 5: Vector Stores

Vector stores index embeddings to facilitate sub-second nearest-neighbor search across millions of documents.

### Vector Store Comparison

| Feature | Chroma DB | FAISS (Facebook AI Similarity Search) |
| :--- | :--- | :--- |
| **Type** | Embedded / Client-Server Database | High-performance C++ In-Memory Index |
| **Storage** | SQLite + Flat Binary on Disk | Compressed flat / IVF index files (`.faiss`, `.pkl`) |
| **Metadata Filtering** | Built-in rich SQL/Document metadata filters | Primarily vector IDs (LangChain wraps with DocStore) |
| **Persistence** | Native `persist_directory` | `save_local()` and `load_local()` |
| **Best For** | Local persistent applications, prototyping | Ultra-fast similarity search, massive vector scaling |

---

### 5.1 Chroma DB Implementation
Creates a persistent vector database on disk and queries it:

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. Ingestion
loader = TextLoader("k.txt")
docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Persist to disk
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 3. Reload from disk
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 4. Similarity Search
query = "Why do they talk about rivers?"
results = db.similarity_search(query, k=2)
print("Top Result:", results[0].page_content)

# 5. Search with Distance Scores (Lower score = closer match in L2 distance)
results_with_scores = db.similarity_search_with_score(query, k=2)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | Content: {doc.page_content[:80]}...")
```

---

### 5.2 FAISS Implementation
Creates an in-memory index, serializes it to disk, and reloads it:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 1. Create FAISS index from documents
vectorstore = FAISS.from_documents(docs, embeddings)

# 2. Save index to disk
vectorstore.save_local("faiss_index")

# 3. Load index from disk
loaded_vectorstore = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # Required because LangChain uses pickle for metadata
)

# 4. As a Retriever
retriever = loaded_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("Key career accomplishments")
```

---

## Chapter 6: Prompts, Output Parsers & LCEL

LangChain Expression Language (LCEL) provides a unified way to compose custom chains using the pipe operator (`|`).

```
Chain = Prompt | LLM | OutputParser
```

Every LCEL component implements the `Runnable` protocol, exposing:
- `invoke(input)`: Synchronous single-item invocation.
- `batch([inputs])`: Parallel processing of multiple inputs.
- `stream(input)`: Generator streaming chunks in real-time.
- `ainvoke()`, `abatch()`, `astream()`: Asynchronous counterparts.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI Engineer. Answer clearly and concisely."),
    ("user", "Explain {topic} in 2 sentences.")
])

# 2. Initialize Model
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# 3. Define Parser
output_parser = StrOutputParser()

# 4. Compose LCEL Chain
chain = prompt | llm | output_parser

# 5. Execute
response = chain.invoke({"topic": "Vector Embeddings"})
print("Response:\n", response)
```

---

## Chapter 7: Building End-to-End RAG Pipelines

A production RAG chain retrieves relevant passages and feeds them into a context-bound prompt:

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

# Step 1: Load Web Data
loader = WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")
raw_docs = loader.load()

# Step 2: Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(raw_docs)

# Step 3: Embed and Index with FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 4: Construct Stuff Documents Chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the provided context below.
If the answer cannot be found in the context, reply "I don't have enough context to answer."

<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Step 5: Construct Retrieval Chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Step 6: Query the RAG Chain
query = "What are the two usage limits in LangSmith to manage spend?"
result = retrieval_chain.invoke({"input": query})

print("Answer:\n", result["answer"])
print(f"\nRetrieved {len(result['context'])} context chunks.")
```

---

## Chapter 8: Observability, Tracing & Cost Control with LangSmith

When deploying LLM applications, understanding internal chain executions, token expenditures, and debugging prompt failures requires observability.

### 8.1 Configuring LangSmith Tracing
LangChain automatically pipes trace telemetry directly to LangSmith when the following environment variables are set in `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_langsmith_api_key_here
LANGCHAIN_PROJECT=My-LangChain-Project
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

In your Python code:
```python
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Default-Project")
```

### 8.2 What LangSmith Captures
1. **Full Run Trees:** Visual breakdown of user input ➔ prompt rendering ➔ vector retrieval ➔ LLM completion ➔ output parsing.
2. **Token Accounting:** Exact prompt tokens, completion tokens, and associated dollar costs per run.
3. **Latency Profiling:** Identifies slow retrievers or high time-to-first-token (TTFT) model calls.
4. **Error Logs:** Full stack traces for failed API calls or malformed output parsing.

### 8.3 Managing LangSmith Spend & Retention
*(As detailed in `1.1-openai/1.1.2-Simpleapp.ipynb`)*

LangSmith charges by two primary metrics:
1. **Base Traces:** Standard traces retained for 14 days (cost-effective for testing).
2. **Extended Retention Traces:** Upgraded traces stored for 400 days (for long-term audits/evals).

#### Optimization Strategies:
- **Set Retention Defaults:** Set organization and project defaults to **Base Retention (14 days)**.
- **Selective Sampling (Run Rules):** Configure rules to sample only ~10% of production runs (matching root runs) for 400-day extended storage.
- **Usage Limits:** In *Settings ➔ Usage Configuration*, configure monthly caps on total traces and extended retention traces to prevent runaway bills.

---

## Chapter 9: Local Models (Ollama) & Interactive Streamlit Web Apps

LangChain allows local open-source LLMs (such as Google's Gemma or Meta's Llama) to be integrated into full web interfaces using **Streamlit**.

### 9.1 Local Model Calling (`ChatOllama` vs `Ollama`)
- `langchain_ollama.ChatOllama`: Latest chat-model interface supporting modern chat messages, tool calling, and streaming.
- `langchain_community.llms.Ollama`: Traditional completion-model wrapper.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma:2b",  # Or "gemma3:1b", "llama3:8b", etc.
    temperature=0.3
)
```

### 9.2 Building an Interactive Streamlit App
*(As implemented in `1.2-ollama/app.py` and `project1/project/app.py`)*

```python
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Environment & LangSmith Setup
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "Streamlit-Ollama-Demo"

# 2. UI Layout
st.set_page_config(page_title="Local AI Assistant", page_icon="🤖")
st.title("🤖 LangChain + Ollama Gemma Assistant")
st.caption("Powered by local open-weight models and LangChain Expression Language (LCEL).")

# 3. Input Widget
input_text = st.text_input("Ask a question:", placeholder="e.g. What is gradient descent?")

# 4. Chain Configuration
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent, concise AI assistant."),
    ("user", "Question: {question}")
])

llm = ChatOllama(model="gemma:2b", temperature=0.2)
chain = prompt | llm | StrOutputParser()

# 5. Execution & Display
if input_text:
    with st.spinner("Generating local response..."):
        response = chain.invoke({"question": input_text})
        st.write("### Response:")
        st.write(response)
```

To run the Streamlit app:
```bash
streamlit run 1.2-ollama/app.py
```

---

## Chapter 10: Python Script (.py) Codebase Mapping

All hands-on labs are organized into executable Python scripts (`.py`) for version control and CLI execution (with local `.ipynb` files ignored via `.gitignore`):

| Category | Primary Python Script (`.py`) | Core Focus |
| :--- | :--- | :--- |
| **OpenAI Quickstart** | `1.1-openai/1.1.1-GettingStarted.py` | OpenAI setup, PromptTemplates, StrOutputParser, LangSmith |
| **OpenAI Web RAG** | `1.1-openai/1.1.2-Simpleapp.py` | WebBaseLoader, FAISS, stuff_documents_chain, retrieval_chain |
| **Ollama RAG App** | `1.2-ollama/1.2.1-Simpleapp.py` | Local Ollama RAG pipeline with FAISS |
| **Ollama Streamlit UI**| `1.2-ollama/app.py` | Streamlit web interface with Gemma:2b & LangSmith |
| **LCEL Components** | `dataintegratiion/components.py` | ChatPromptTemplate, StrOutputParser, LCEL chains |
| **Data Ingestion** | `dataintegratiion/langchain-1.py` | TextLoader, PyPDFLoader, WebBaseLoader, Arxiv client |
| **Wikipedia Loading**| `dataintegratiion/wikipdia.py` | WikipediaLoader with user-agent config |
| **OpenAI Embeddings** | `embedding/4.1-embedding.py` | OpenAI text-embedding-3-large / dimension testing |
| **HuggingFace Embed** | `embedding/hf.py` | all-MiniLM-L6-v2 local sentence transformers |
| **Ollama Embeddings** | `embedding/ollama_embedding.py` | Local nomic-embed-text embeddings |
| **OpenAI Embedding Demo**| `embedding/openai-embedding.py` | OpenAI embed_query & embed_documents |
| **Project 1 Full RAG** | `project1/projects.py` | End-to-end RAG over Virat Kohli biography data |
| **Project 1 Web UI** | `project1/project/app.py` | Streamlit RAG web UI using ChatOllama gemma3:1b |
| **JSON Text Splitter**| `text-spliting/json-text-spliter.py` | RecursiveJsonSplitter for nested JSON structures |
| **Character Splitter**| `text-spliting/text-split-teq-2.py` | CharacterTextSplitter with strict separators |
| **Recursive Splitter**| `text-spliting/text-split.py` | RecursiveCharacterTextSplitter with chunk overlap |
| **HTML Header Split** | `text-spliting/text-spliter-html.py` | HTMLHeaderTextSplitter extracting structural metadata |
| **Chroma Database** | `vector_store/chroma_db/chroma.py` | Persistent ChromaDB storage, similarity search & scores |
| **FAISS Vector Index**| `vector_store/fassi/5.1-Faiss.py` | FAISS index creation, local saving, loading & retrieval |

---

## Chapter 11: Quick-Start Cheat Sheet & Production Reference

### Consolidated Import Reference
```python
# ─── Loaders ─────────────────────────────────────────────────────────────
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    WikipediaLoader
)

# ─── Text Splitters ──────────────────────────────────────────────────────
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    RecursiveJsonSplitter
)

# ─── Embedding Models ────────────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# ─── Vector Stores ───────────────────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# ─── Models, Prompts, Parsers & LCEL ────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── High-Level Chains ───────────────────────────────────────────────────
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
```

### Complete 1-File Production Pipeline
```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 0. Load Configuration
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# 1. Load Data
docs = TextLoader("project1/virat_kohli.txt", encoding="utf-8").load()

# 2. Split Document into Semantic Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 3. Embed & Persist to Vector Index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Build Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5. Define Contextual Prompt & Stuff Chain
prompt = ChatPromptTemplate.from_template("""
You are a precise sports analyst. Answer the question using ONLY the provided context.
Context:
{context}

Question: {input}
""")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
document_chain = create_stuff_documents_chain(llm, prompt)

# 6. Compose Final Retrieval Chain
rag_chain = create_retrieval_chain(retriever, document_chain)

# 7. Execute Query
query = "What are the major IPL and international records of Virat Kohli?"
response = rag_chain.invoke({"input": query})

print("="*60)
print("ANSWER:\n", response["answer"])
print("="*60)
print(f"Retrieved {len(response['context'])} relevant document chunks.")
```

---
*LangChain Study Notes & Architecture Reference — Fully synchronized with repository notebooks, Python scripts, and projects.*
