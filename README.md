## Project Giva: AI-Powered Document Q\&A

### 1. Project Overview

Project Giva is a **Python-based RAG (Retrieval-Augmented Generation) system** that allows users to ask questions about their own documents. It uses a **vector database** to store document information and leverages a **Large Language Model (LLM)** to generate human-like answers based on the retrieved context.

The core functionalities include:

* **Document Ingestion:** Loads text files, splits them into chunks, creates embeddings, and stores them in a vector database for efficient searching.
* **Query Enhancement:** Automatically expands short user queries to improve the relevance of search results.
* **Conversational Memory:** Remembers the last few questions and answers to provide context-aware responses in an ongoing conversation.
* **Advanced RAG Pipeline:** Retrieves relevant document chunks, assesses a confidence score, and generates a detailed, well-structured answer.
* **Source Citing:** Lists the source documents (and pages) that were used to generate the answer, along with a relevance score for each.
* **Interactive Console:** Provides a user-friendly command-line interface to interact with the system.

### 2. File Structure

The main project logic is contained within the `e:\2025\Project_Learning\RAG\` directory.

| File/Folder | Description |
| :--- | :--- |
| `app.py` | Main application entry point and RAG orchestration logic. |
| `main.py` | A placeholder/scaffolding file, not used by `app.py`. |
| `src/` | (Inferred) A package containing core RAG components. |
| `src/__init__.py` | |
| `src/document_loader.py` | (Inferred) Contains `load_all_documents`. |
| `src/embedding.py` | (Inferred) Contains `EmbeddingPipeline`. |
| `src/vectorstore.py` | (Inferred) Contains `VectorStore`. |
| `src/retriever.py` | (Inferred) Contains `RAGRetriever`. |
| *Other files* | Unrelated scripts or temporary files (see Section 6). |

### 3. Code Explanation: `app.py`

This is the central file that orchestrates the entire RAG pipeline.

#### Key Components & Functions

| Function | Purpose | Logic |
| :--- | :--- | :--- |
| `enhance_query_with_ai(...)` | To improve document retrieval for short or ambiguous user questions. | If a user's query is less than three words long, it asks the LLM to expand it into a more comprehensive search query, adding synonyms and context. For longer queries, it uses the original query. |
| `rag_simple(...)` | A basic RAG implementation. | Retrieves the top *k* documents, combines their content into a single context, and passes it to the LLM with a straightforward prompt. **Note:** This function is defined but not used in the main execution loop. |
| `rag_advanced(...)` | The primary, feature-rich RAG pipeline used by the application. | Retrieves documents based on the query, *top\_k* count, and a minimum similarity score (`min_score`). Constructs a list of **sources** (file name, page, similarity score). Calculates a **confidence** score (highest similarity score). Incorporates `conversation_history` into the LLM prompt. Uses a detailed, structured prompt to instruct the AI ("giva") on how to analyze and structure its answer. Returns a dictionary with `answer`, `sources`, `confidence`. |

#### Main Execution Block (`if __name__ == "__main__":`)

This block runs the interactive console application.

1.  **Setup:** Loads environment variables (specifically `GEMINI_API_KEY`) and initializes the `ChatGoogleGenerativeAI` model (`gemini-1.5-flash`).
2.  **Component Initialization:** Creates instances of the core components from the `src` package: `EmbeddingPipeline`, `VectorStore`, and `RAGRetriever`.
3.  **Data Ingestion:**
    * Checks if the vector store is already populated.
    * If not, it loads all documents from the `data/text_files` directory.
    * Uses the `EmbeddingPipeline` to process and embed the documents and adds them to the `VectorStore`.
4.  **Interactive Console Loop:**
    * Starts an infinite `while` loop, prompting the user for a question.
    * Listens for special commands: `exit`/`quit` to terminate and `reset` to clear the conversation history.
    * The user's query is passed to `enhance_query_with_ai`.
    * The (potentially enhanced) query is then passed to `rag_advanced` along with the current conversation history.
    * The new question and the start of the AI's answer are appended to the `conversation_history`. The history is trimmed to the last 10 exchanges.
    * The results are displayed in a clean, formatted block.

### 4. How to Run the Project

#### Prerequisites

Ensure you have Python installed and the required packages from `requirements.txt` (listed below).

# Expected Output

--- Starting RAG Pipeline Orchestration ---
✅ LLM initialized (gemini-1.5-flash)

--- INGESTION: Starting Document Loading and Indexing ---
--- INGESTION COMPLETE ---

--- Giva-Powered RAG Console Ready ---
🤖 Enhanced with giva conversation memory and intelligent document analysis
Commands: 'exit'/'quit' to close, 'reset' to clear conversation history
Ask questions about your documents for intelligent AI-powered answers!

Ask a question about your documents (or type 'exit'):

# Subsequent Runs & Interaction:

Ask a question about your documents (or type 'exit'): What is the project about?
🔍 Searching and generating answer...


🤖 GIVA-POWERED RAG RESPONSE

📝 QUESTION: What is the project about?
🎯 CONFIDENCE: 0.8123 (High)
📚 DOCUMENTS ANALYZED: 5

💡 GIVA ANSWER:

Based on the provided documents, this project is a Retrieval-Augmented Generation (RAG) system designed to answer questions from a local document knowledge base.

Key features include:
* **AI-Powered Analysis:** It uses an AI assistant named "giva" to provide answers.
* **Source-Based Answers:** The system relies on context retrieved from documents to formulate its responses.
* **Conversational Context:** It can maintain a conversation history to provide more coherent answers over multiple questions.

📖 SOURCES (5 documents):

  1. 🔥 my_document_1.pdf (Page: 2) - Relevance: 0.8123
     Preview: ...The core of the RAG pipeline involves retrieving relevant text chunks from the indexed vector store and feeding them...
  2. 📄 another_doc.txt (Page: 1) - Relevance: 0.6543
     Preview: ...giva is an AI assistant specialized in analyzing and answering questions based on provided documents. Use the following...
  ...


#### Installation

```bash
pip install langchain langchain_community langchain_core langchain_google_genai pypdf pymupdf chromadb faiss-cpu sentence-transformers python-dotenv 
