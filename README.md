# Adityabot: News Research & Q&A Tool 📈

AdityaBot is an end-to-end news research tool powered by Google Gemini, LangChain, and Streamlit. It enables users to input live article URLs from any news portal, automatically extract and chunk content, build vector embeddings, and perform factual Question & Answering (Q&A) with clear source citations across any topic or domain.

---

## 🌟 Key Features

- **Automated Web Scraping:** Uses `WebBaseLoader` with browser headers to bypass common bot-detection and paywall overlays.
- **Recursive Text Chunking:** Splits unstructured text documents into optimized token windows with overlap using `RecursiveCharacterTextSplitter`.
- **Vector Search Engine:** Leverages `models/gemini-embedding-001` paired with a local `FAISS` vector database for low-latency similarity search.
- **Context-Grounded Q&A:** Employs `gemini-2.5-flash` with a tailored prompt template to ensure answers are strictly factual and reference source URLs.
- **Pure Python Frontend:** Interactive, responsive web interface built entirely with Streamlit.

---

## 🏗️ Architecture & Pipeline

```text
[ Article URLs ] 
       │
       ▼
[ WebBaseLoader (BeautifulSoup) ] ──▶ Scrapes clean text
       │
       ▼
[ RecursiveCharacterTextSplitter ] ──▶ Splits into 1,000-char chunks
       │
       ▼
[ Google Gemini Embeddings ] ────────▶ Generates vector representations
       │
       ▼
[ FAISS Vector Store ] ──────────────▶ Indexes locally (faiss_index_gemini)
       │
       ▼ (User Query)
[ Similarity Retrieval (Top-k) ] ───▶ Relevant Context + User Question
       │
       ▼
[ Gemini 2.5 Flash LLM ] ────────────▶ Structured Answer + Source Citations
```

---

## 📁 Project Structure

```text
langchain/
├── .env                     # Private API keys (not checked into Git)
├── .gitignore               # Ignored files (venv, .env, faiss indices)
├── main.py                  # Main Streamlit application and RAG pipeline
├── requirements.txt         # Project dependencies
├── rockybot.jpg             # Project branding/logo image
└── README.md                # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 to 3.12 installed.
- A Google AI Studio API Key. Get one for free at [Google AI Studio](https://aistudio.google.com/).

### 2. Clone and Setup Environment

```bash
# Clone repository
git clone <your-repository-url>
cd langchain

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
touch .env
```

Add your Gemini API key inside `.env`:

```env
GOOGLE_API_KEY="AIzaSyYourActualGoogleGeminiApiKey"
```

---

## 💻 Running the Application

Launch the Streamlit app:

```bash
streamlit run main.py
```

Streamlit will open the user interface in your default browser at `http://localhost:8501`.

### How to Use:
1. Paste up to 3 article URLs into the sidebar inputs.
2. Click **Process URLs** to download content, generate embeddings, and build the FAISS index.
3. Type any question related to the articles into the main prompt box.
4. Review the generated answer along with the referenced source links.

---

## 📦 Core Dependencies

| Package | Purpose |
| :--- | :--- |
| `streamlit` | Reactive web user interface |
| `langchain` / `langchain-core` | Core framework for orchestrating RAG workflows |
| `langchain-google-genai` | Integration package for Google Gemini LLM and embeddings |
| `langchain-community` | Document loaders and community vector store wrappers |
| `faiss-cpu` | High-performance vector database for similarity search |
| `beautifulsoup4` | HTML parsing engine for `WebBaseLoader` |
| `python-dotenv` | Secure loading of environment variables from `.env` |

---

## 🛡️ License

This project is licensed under the MIT License.