import os
import streamlit as st
from dotenv import load_dotenv

# LangChain & Gemini imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load environment variables
load_dotenv()

# Explicitly fetch the API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Aditya: News Research Tool", page_icon="📈")
st.title("Aditya: News Research Tool 📈")

# Validate API key presence before running model operations
if not api_key:
    st.error(
        "Missing Google API Key! Please verify that your `.env` file exists in the "
        "root folder and contains: GOOGLE_API_KEY=your_key_here"
    )
    st.stop()

st.sidebar.title("News Article URLs")

# Sidebar inputs for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
db_folder_path = "faiss_index_gemini"

main_placeholder = st.empty()

# 2. Initialize Gemini LLM & Embeddings with explicit api_key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=api_key,
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",  # or "gemini-embedding-001"
    google_api_key=api_key,
)

# 3. Process URLs and Build Vector Store
if process_url_clicked:
    if not urls:
        st.sidebar.warning("Please provide at least one valid URL.")
    else:
        try:
            main_placeholder.text("Data Loading... Started... 🌐")
            loader = WebBaseLoader(
                web_paths=urls,
                header_template={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                },
            )
            data = loader.load()

            # Split text into manageable chunks
            main_placeholder.text("Splitting Text into Chunks... ✂️")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""],
            )
            docs = text_splitter.split_documents(data)

            # Generate embeddings and store in FAISS
            main_placeholder.text("Generating Embeddings & Building Index... 🧠")
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Save FAISS index locally
            vectorstore.save_local(db_folder_path)
            main_placeholder.success("Index built and saved successfully! ✅")
        except Exception as e:
            main_placeholder.error(f"Error processing URLs: {str(e)}")

# 4. Question Answering Interface
query = st.text_input("Ask a question about the processed articles:")

if query:
    if os.path.exists(db_folder_path):
        try:
            # Load vector database
            vectorstore = FAISS.load_local(
                db_folder_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # Retrieve relevant chunks
            retrieved_docs = retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            sources = list({doc.metadata.get("source", "Unknown") for doc in retrieved_docs})

            # QA Prompt
            prompt = ChatPromptTemplate.from_template(
                """Answer the following question based only on the provided context:

Context:
{context}

Question:
{question}

Provide a direct, comprehensive answer. If the answer is not in the context, state that clearly."""
            )

            # LCEL Pipeline
            rag_chain = prompt | llm | StrOutputParser()
            answer = rag_chain.invoke({"context": context_text, "question": query})

            st.header("Answer")
            st.write(answer)

            if sources:
                st.subheader("Sources:")
                for source in sources:
                    st.write(f"- {source}")
        except Exception as e:
            st.error(f"Error during retrieval/generation: {str(e)}")
    else:
        st.warning("Please process URLs first before asking questions.")