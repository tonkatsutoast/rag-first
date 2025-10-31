from typing import List, Optional, Dict, Any, Callable
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from rag_first.vectorstore.chroma_manager import ChromaManager 
from rag_first.embeddings.embedding_manager import get_embeddings
from rag_first.llm.cloud_llm import get_llm
from rag_first.config.settings import settings
import logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG Pipeline implementation"""
    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        embedding_provider: str = "ollama",
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Initialize RAG Pipeline
        Args:
            llm_provider: LLM provider
            llm_model: LLM model name
            embedding_model: embedding model name
            embedding_provider: embedding provider 
            collection_name: ChromaDB collection name
            temperature: LLM temperature
        """
        # Initialize embeddings
        self.chroma = ChromaManager(
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            collection_name=collection_name or settings.COLLECTION_NAME
        )

        # Initialize the LLM
        self.llm = get_llm(
            provider=llm_provider,
            model=llm_model,
            temperature=temperature
        )

        # Default prompt template
        self.prompt_template = self._create_default_prompt()
        # Create chain
        self.chain = self._create_chain()
        logger.info(
            f"RAG Pipeline initialized: "
            f"LLM: {llm_provider}"
            f"Embedding provider: {embedding_provider}"
            f"Embedding model: {embedding_model}"
        )

    def _create_default_prompt(self) -> ChatPromptTemplate:
        """Create default RAG prompt template"""
        template = """You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        return ChatPromptTemplate.from_template(template)

    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for input to prompt template"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _create_chain(self):
        """Create LangChain LCEL chain"""
        retriever = self.chroma.get_retriever(
            search_kwargs={"k": settings.TOP_K_RESULTS}
        )
        chain = (
            RunnableParallel(
                {
                    "context": retriever | self._format_docs,
                    "question": RunnablePassthrough()
                }
            )
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        return chain

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]
        """Add documents to vector store"""
        return self.chroma.add_documents(documents, ids)
    
    def query(
        self,
        question: str,
        return_source_documents: bool = False
    ) -> Dict[str,Any]:
        """
        Query the RAG Pipeline
        Args:
            question: User input
            return_source_documents: Whether to return the source docs
        Returns:
            Dictionary with answer and optional source docs
        """
        # Get answer
        answer = self.chain.invoke(question)
        result = {"answer": answer, "question": question}
        # Return source documents if requested
        if return_source_documents:
            retriever = self.chroma.get_retriever()
            source_docs = retriever.invoke(question)
            result["source_documents"] = source_docs
        return result

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """Query and return sources"""
        return self.query(question, return_source_documents=True)
    
    def set_custom_prompt(self, prompt_template: ChatPromptTemplate):
        """Set custom prompt template"""
        self.prompt_template = prompt_template
        self.chain = self_create_chain()
        logger.info("Custom prompt template set")
    
    def update_retrieval_config(self, **kwargs):
        """Udate retrieval configuration"""
        self.chain = self._create_chain()
        logger.info(f"Retrieval config updated: {kwargs}")

        