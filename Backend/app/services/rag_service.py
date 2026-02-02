import os
import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.vector_store_dir = os.path.join(os.getcwd(), "outputs", "vector_stores")
        os.makedirs(self.vector_store_dir, exist_ok=True)
        # Check API Key
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set. RAG Service will fail.")

    def _get_index_path(self, task_id: str) -> str:
        return os.path.join(self.vector_store_dir, task_id)

    async def ingest_report(self, task_id: str, text_content: str):
        """
        Chunk text and build FAISS index. Save to disk.
        """
        try:
            logger.info(f"Building RAG index for task {task_id}...")
            
            # 1. Text Splitter (from user snippet)
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len
            )
            chunks = text_splitter.split_text(text_content)
            
            if not chunks:
                logger.warning("No text chunks generated for RAG.")
                return

            # 2. Embeddings & Vector Store
            embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            vector_store = FAISS.from_texts(chunks, embeddings)
            
            # 3. Save
            index_path = self._get_index_path(task_id)
            vector_store.save_local(index_path)
            logger.info(f"RAG index saved to {index_path}")
            
        except Exception as e:
            logger.error(f"RAG Ingestion failed: {e}")
            # Don't crash the main pipeline; chat just won't work.

    async def chat_with_report(self, task_id: str, question: str) -> str:
        """
        Retrieve context and answer question.
        """
        try:
            index_path = self._get_index_path(task_id)
            if not os.path.exists(index_path):
                return "I'm sorry, I don't have enough context about this report yet. Please ensure the analysis is complete."
            
            # 1. Load Vector Store
            embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            # allow_dangerous_deserialization=True is required for loading pickled FAISS 
            # (SAFE here because we created the files ourselves in a protected dir)
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            
            # 2. Retrieval
            docs = vector_store.similarity_search(question, k=4)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            if not context_text:
                return "I couldn't find any relevant details in the report to answer that."

            # 3. Generation (ChatOpenAI)
            llm = ChatOpenAI(
                model_name="gpt-4", # Or gpt-3.5-turbo
                temperature=0.3, 
                api_key=settings.OPENAI_API_KEY
            )
            
            prompt = ChatPromptTemplate.from_template(
                """You are a helpful data analyst assistant. 
Answer the question based ONLY on the following context from the analysis report:

{context}

Question: {question}

If the context doesn't contain the answer, say "I don't see that information in the report."
"""
            )
            
            chain = prompt | llm
            response = await chain.ainvoke({"context": context_text, "question": question})
            
            return response.content

        except Exception as e:
            logger.error(f"RAG Chat failed: {e}")
            return "I encountered an error while processing your question."

rag_service = RAGService()
