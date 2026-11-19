# 🦜🔗 LangChain Learning Repository 

A hands-on learning project covering **LangChain** fundamentals — from loading data from various sources to splitting text using multiple strategies, generating embeddings, using vector stores, building complete RAG pipelines, monitoring with LangSmith, running local models with Ollama, and building interactive web apps with Streamlit.

All practical examples and modules are structured and version-controlled as **Python scripts (`.py`)** (with `.ipynb` notebooks and `.html` notes ignored via `.gitignore` to keep the repository lightweight and clean).

---

## 📁 Project Structure

```
langchain/
├── .env                              # Environment variables (API keys & LangSmith config - ignored)
├── .gitignore                        # Git ignore rules (*.ipynb, environments, secrets, caches)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview and guide
├── NOTES.md                          # Comprehensive 11-Chapter study notes & reference
├── langchain_notes.html              # Styled, printable HTML study notes (local only, ignored in git)
│
├── 1.1-openai/                       # Module 1: OpenAI & Core Components
│   ├── 1.1.1-GettingStarted.py       # Setup, ChatOpenAI, Prompts, StrOutputParser
│   └── 1.1.2-Simpleapp.py            # WebBaseLoader, FAISS & LangSmith trace setup
│
├── 1.2-ollama/                       # Module 2: Local LLMs with Ollama
│   ├── 1.2.1-Simpleapp.py            # Local RAG with Ollama & FAISS
│   └── app.py                        # Interactive Streamlit UI with Gemma:2b
│
├── dataintegratiion/                 # Module 3: Data Integration & Document Loaders
│   ├── components.py                 # PromptTemplates, OutputParsers & LCEL Chains
│   ├── langchain-1.py                # TextLoader, PyPDFLoader, WebBaseLoader, ArXiv
│   ├── wikipdia.py                   # WikipediaLoader query API
│   ├── a.pdf                         # Sample PDF document
│   ├── linear.txt                    # Scraped Wikipedia text (Linear Regression)
│   └── story.txt                     # Sample text file (DSA Algorithm)
│
├── text-spliting/                    # Module 4: Text Splitting Techniques
│   ├── text-split.py                 # RecursiveCharacterTextSplitter
│   ├── text-split-teq-2.py           # CharacterTextSplitter (strict delimiter)
│   ├── text-spliter-html.py          # HTMLHeaderTextSplitter
│   ├── json-text-spliter.py          # RecursiveJsonSplitter
│   └── story.txt                     # Sample text file for splitting tests
│
├── embedding/                        # Module 5: Embeddings
│   ├── 4.1-embedding.py              # OpenAI text-embedding-3-large & 1024 dims
│   ├── openai-embedding.py           # OpenAI Embeddings demo
│   ├── ollama_embedding.py           # Ollama Embeddings (nomic-embed-text)
│   └── hf.py                         # Hugging Face Embeddings (MiniLM-L6-v2)
│
├── vector_store/                     # Module 6: Vector Stores
│   ├── chroma_db/
│   │   ├── chroma.py                 # Chroma DB persistent vector storage & queries
│   │   └── k.txt                     # Sample text for Chroma indexing
│   └── fassi/
│       ├── 5.1-Faiss.py              # FAISS index creation, saving & loading
│       └── a.txt                     # Sample text for FAISS
│
├── LCEL/                             # Module 7: LangChain Expression Language
│   └── lcel-part1.py                 # LCEL reference file
│
└── project1/                         # Module 8: Full End-to-End RAG Project
    ├── projects.py                   # End-to-End RAG Pipeline over Virat Kohli dataset
    ├── virat_kohli.txt               # Source dataset for RAG pipeline
    └── project/
        └── app.py                    # Streamlit RAG Web App with ChatOllama
```

---

## 🔧 Setup & Installation

### 1. Create and Activate Virtual Environment
```bash
conda create -n genAi python=3.14
conda activate genAi
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here

# LangSmith Observability & Tracing (Optional but Recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=My-LangChain-Project
```

---

## 📦 Key Dependencies

| Package | Purpose |
| :--- | :--- |
| `langchain` | Core LangChain framework |
| `langchain-community` | Community document loaders, vector stores & integrations |
| `langchain-openai` | OpenAI LLM & embedding integration (`ChatOpenAI`, `OpenAIEmbeddings`) |
| `langchain-text-splitters` | Text splitting utilities (`RecursiveCharacterTextSplitter`, etc.) |
| `langchain-huggingface` | Hugging Face models & embeddings |
| `langchain-chroma` | Chroma vector database integration |
| `langchain-ollama` | Updated Ollama integration (`ChatOllama`, `OllamaEmbeddings`) |
| `faiss-cpu` | FAISS vector store (fast local similarity search) |
| `streamlit` | Interactive web application framework for AI apps |
| `python-dotenv` | Load environment variables from `.env` |
| `pypdf`, `pymupdf` | PDF file loading and parsing |
| `bs4` (BeautifulSoup) | HTML parsing and web scraping |
| `wikipedia`, `arxiv` | Wikipedia and ArXiv research paper APIs |

---

## 📚 Topics Covered

### 1. 📥 Data Integration — Document Loaders
- **`TextLoader`**: Load unstructured `.txt` files.
- **`PyPDFLoader`**: Page-by-page PDF extraction with page metadata.
- **`WebBaseLoader`**: Scrape online websites with `bs4.SoupStrainer`.
- **`WikipediaLoader`**: Query Wikipedia articles directly with custom user-agents.
- **`arxiv.Client`**: Fetch research paper abstracts, titles, and author metadata.

### 2. ✂️ Text Splitting Strategies
- **`RecursiveCharacterTextSplitter`**: Delimiter hierarchy (`\n\n`, `\n`, ` `, `""`) preserving paragraph and sentence boundaries.
- **`CharacterTextSplitter`**: Strict single-delimiter splitting.
- **`HTMLHeaderTextSplitter`**: Split HTML based on `<h1>`, `<h2>`, `<h3>` tags with structural metadata.
- **`RecursiveJsonSplitter`**: Chunk structured JSON documents preserving valid keys and hierarchy.

### 3. 🧠 Embeddings
- **OpenAI**: `text-embedding-3-small` and `text-embedding-3-large` with custom dimensions (e.g. 1024).
- **Ollama**: Local, private embeddings with `nomic-embed-text`.
- **HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2` running on CPU/GPU.

### 4. 🗄️ Vector Databases
- **Chroma DB**: Embedded, local persistent storage with SQLite backend and similarity score search.
- **FAISS**: In-memory vector index with `save_local` and `load_local` serialization.

### 5. 🔗 LCEL & Prompt Engineering
- **`ChatPromptTemplate`**: Parameterized system and human message templates.
- **`StrOutputParser`**: Clean string extraction from model responses.
- **LCEL Pipe (`|`)**: Composing `chain = prompt | llm | output_parser`.

### 6. 🚀 Retrieval-Augmented Generation (RAG)
- `create_stuff_documents_chain`: Combines retrieved documents into prompt context.
- `create_retrieval_chain`: Connects retriever with document combination chain to answer questions.

### 7. 🔍 Observability with LangSmith
- Trace visualization, run trees, latency inspection, and token cost tracking.
- Retention management (Base 14-day vs Extended 400-day) and spend limits.

### 8. 🌐 Interactive Web Applications (Streamlit)
- `1.2-ollama/app.py`: Gemma:2b chat interface with LangSmith tracing.
- `project1/project/app.py`: Interactive RAG question-answering app with ChatOllama.

---

## 🚀 How to Run

### Option A: Running Python Scripts
Run any script directly from your terminal:
```bash
python 1.1-openai/1.1.1-GettingStarted.py
python 1.1-openai/1.1.2-Simpleapp.py
python 1.2-ollama/1.2.1-Simpleapp.py
python dataintegratiion/langchain-1.py
python text-spliting/text-split.py
python vector_store/fassi/5.1-Faiss.py
python project1/projects.py
```

### Option B: Launching Streamlit Web Apps
```bash
# Launch Ollama Assistant
streamlit run 1.2-ollama/app.py

# Launch Project 1 RAG App
streamlit run project1/project/app.py
```

---

## 📖 Study Notes & Documentation

Detailed notes and reference guides are maintained in this repository:
- **[NOTES.md](file:///d:/langchain/NOTES.md)**: Full 11-Chapter comprehensive study notes with diagrams, mathematical formulations, and production examples.
- **[langchain_notes.html](file:///d:/langchain/langchain_notes.html)**: Styled HTML reference document suitable for in-browser reading or PDF export (`Ctrl+P → Save as PDF`).

---

## 📄 License

Educational and learning repository. Feel free to use and adapt the code for your projects.
