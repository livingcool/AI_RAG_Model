import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, Union

# Import core components from the local package
from src import load_all_documents, EmbeddingPipeline, VectorStore, RAGRetriever

# --- CONVERSATION MEMORY ---
conversation_history = []

# --- GIVA QUERY ENHANCEMENT ---
def enhance_query_with_ai(query: str, llm: ChatGoogleGenerativeAI) -> str:
    """Use giva to enhance and expand user queries for better retrieval."""
    if len(query.split()) < 3:  # Only enhance short queries
        prompt = f"""You are giva. Given this user query: "{query}"
        
        Expand it into a more comprehensive search query that would help find relevant information in business documents. 
        Keep the core meaning but add related terms, synonyms, and context that would improve document retrieval.
        
        Return only the enhanced query, nothing else."""
        
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except:
            return query
    return query 


# --- LLM ORCHESTRATION FUNCTIONS ---

def rag_simple(query: str, retriever: RAGRetriever, llm: ChatGoogleGenerativeAI, top_k: int = 3) -> str:
    """Performs a simple RAG pipeline with enhanced AI prompting."""
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."

    prompt = f"""You are giva, an AI assistant specialized in analyzing and answering questions based on provided documents. 
Use the following context from documents to provide a comprehensive, accurate, and well-structured answer.

Context from Documents:
{context}

Question: {query}

Instructions:
1. Answer based primarily on the provided context
2. If the context doesn't contain enough information, state what's missing
3. Provide specific examples or quotes from the context when relevant
4. Structure your answer clearly with bullet points or numbered lists when appropriate
5. Be concise but thorough

Answer:"""
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating answer with LLM: {e}"

def rag_advanced(
    query: str, 
    retriever: RAGRetriever, 
    llm: ChatGoogleGenerativeAI, 
    top_k: int = 5, 
    min_score: float = 0.2, 
    return_context: bool = False,
    conversation_history: List[str] = None
) -> Dict[str, Union[str, float, List[Dict[str, Any]], None]]:
    """RAG pipeline with structured output, sources, and confidence score."""
    
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found to answer the question.', 'sources': [], 'confidence': 0.0, 'context': None}

    context = "\n\n".join([doc['content'] for doc in results])
    sources = [
        {'source': doc['metadata'].get('source_file', 'unknown'), 'page': doc['metadata'].get('page', 'unknown'), 'score': doc['similarity_score'], 'preview': doc['content'][:120] + '...'} 
        for doc in results
    ]
    confidence = max([doc['similarity_score'] for doc in results])
    
    
    # Build conversation context
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_context = f"""
Previous Conversation Context:
{chr(10).join(conversation_history[-3:])}  # Last 3 exchanges
"""
    
    # Create an enhanced AI prompt for better RAG responses with conversation memory
    prompt = f"""You are giva, an expert AI assistant with access to a comprehensive document knowledge base. 
Your task is to provide accurate, detailed, and insightful answers based on the retrieved context.
{conversation_context}
Retrieved Context (from {len(results)} relevant documents):
{context}

Current User Question: {query}

Analysis Instructions:
1. **Primary Analysis**: Base your answer primarily on the provided context
2. **Conversation Awareness**: Consider the conversation history to provide coherent responses
3. **Confidence Assessment**: If context is insufficient, clearly state what information is missing
4. **Source Integration**: Reference specific information from the context with relevant details
5. **Structure**: Organize your response with clear headings, bullet points, or numbered lists
6. **Completeness**: Provide comprehensive coverage of the question topic
7. **Accuracy**: Only include information that can be supported by the provided context
8. **Continuity**: Build upon previous conversation topics when relevant

Answer:"""
    
    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"Error generating answer with LLM: {e}"
        
    output = {'answer': answer, 'sources': sources, 'confidence': confidence, 'context': None}
    if return_context:
        output['context'] = context
    return output


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    print("\n--- Starting RAG Pipeline Orchestration ---")
    
    # 1. Setup Environment and LLM
    load_dotenv() 
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_key
    )
    print("✅ LLM initialized (gemini-2.5-flash)")
    
    # 2. Initialize Core Components
    embedding_pipeline = EmbeddingPipeline()
    vectorstore = VectorStore()
    
    # 3. Data Ingestion (Only run if the vector store is empty)
    if vectorstore.collection is None or vectorstore.collection.count() == 0:
        print("\n--- INGESTION: Starting Document Loading and Indexing ---")
        
        docs = load_all_documents("data/text_files")
        
        if docs:
            embeddings, chunks = embedding_pipeline.embed_documents(docs)
            vectorstore.add_documents(chunks, embeddings)
            print("--- INGESTION COMPLETE ---")
        else:
            print("⚠️ No documents were loaded. Skipping indexing.")
            
    else:
        print("\n--- VECTOR STORE ALREADY POPULATED (Skipping Ingestion) ---")
    
    # 4. Initialize Retriever 
    rag_retriever = RAGRetriever(vectorstore, embedding_pipeline)

    # 5. Run the Interactive RAG Console Loop
    print("\n--- Giva-Powered RAG Console Ready ---")
    print("🤖 Enhanced with giva conversation memory and intelligent document analysis")
    print("Commands: 'exit'/'quit' to close, 'reset' to clear conversation history")
    print("Ask questions about your documents for intelligent AI-powered answers!")

    while True:
        user_query = input("\nAsk a question about your documents (or type 'exit'): ").strip()
        
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting Giva RAG Console. Goodbye! 👋")
            break
            
        if user_query.lower() == "reset":
            conversation_history.clear()
            print("🔄 Conversation history cleared. Starting fresh!")
            continue
        
        if not user_query:
            continue
            
        if len(user_query.strip()) < 3:
            print("⚠️ Please enter a question with at least 3 characters.")
            continue

        print("🔍 Searching and generating answer...")
        
        try:
            # Enhance query with giva for better retrieval
            enhanced_query = enhance_query_with_ai(user_query, llm)
            if enhanced_query != user_query:
                print(f"🤖 giva-enhanced query: {enhanced_query}")
            
            # Execute the RAG pipeline using the advanced function with conversation context
            answer_adv = rag_advanced(
                query=enhanced_query,  # Use enhanced query for retrieval
                retriever=rag_retriever, 
                llm=llm, 
                top_k=5,  # Increased for better context
                min_score=0.15,  # Slightly higher threshold for quality
                return_context=False,
                conversation_history=conversation_history
            )
            
            # Add to conversation history
            conversation_history.append(f"Q: {user_query}")
            conversation_history.append(f"A: {answer_adv['answer'][:200]}...")  # Truncate for memory
            
            # Keep only last 10 exchanges to manage memory
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
        except Exception as e:
            print(f"❌ Error processing your question: {e}")
            continue
        
        # 6. Display Enhanced Results with giva Analysis
        print("\n" + "="*60)
        print(f"🤖 GIVA-POWERED RAG RESPONSE")
        print("="*60)
        print(f"📝 QUESTION: {user_query}")
        print(f"🎯 CONFIDENCE: {answer_adv['confidence']:.4f} {'(High)' if answer_adv['confidence'] > 0.7 else '(Medium)' if answer_adv['confidence'] > 0.4 else '(Low)'}")
        print(f"📚 DOCUMENTS ANALYZED: {len(answer_adv['sources'])}")
        print("\n💡 GIVA ANSWER:")
        print("-" * 40)
        print(answer_adv['answer'])
        
        if answer_adv['sources']:
            print(f"\n📖 SOURCES ({len(answer_adv['sources'])} documents):")
            print("-" * 40)
            for i, source in enumerate(answer_adv['sources'], 1):
                relevance_indicator = "🔥" if source['score'] > 0.7 else "📄" if source['score'] > 0.4 else "📋"
                print(f"  {i}. {relevance_indicator} {source['source']} (Page: {source['page']}) - Relevance: {source['score']:.4f}")
                print(f"     Preview: {source['preview']}")
        print("="*60)