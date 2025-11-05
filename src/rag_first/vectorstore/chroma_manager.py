from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manage ChromaDB vector store operations"""

    def __init__(
        self,
        embedding_function: Embeddings,
        persist_directory: str | Path,
        collection_name: str = "rag_collection"
    ):
        """
        Args:
            embedding_function: LangChain embeddings instance
            persist_directory: Path to persist ChromaDB
            collection_name: Name of the vectorstore collection
        """
        self.embedding_function = embedding_function
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name

        # Initialize the Chroma vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=self.persist_directory
        )

        logger.info(
            f"ChromaDB initialized: {collection_name} at {persist_directory}"
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional [List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Args:
            documents: List of documents to add
            id: Optional list of document IDs
            batch_size: Number of documents to process per batch
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        # Add docs in batches
        all_ids = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size] if ids else None
            added_ids = self.vector_store.add_documents(
                documents=batch_docs,
                ids=batch_ids
            )
            all_ids.extend(added_ids)
            logger.info(
                f"Added batch {i//batch_size + 1}: "
                f"{len(batch_docs)} documents"
            )
        logger.info(f"Total documents added: {len(all_ids)}")
        return all_ids
    
    def similary_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Args:
            query: Search Query - the original user input used as the natural language query
            k: Number of results to return
            filter: Metadata filter
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str,Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        Args:
            query: Search Query - the original user input used as the natural language query
            k: Number of results to return
            filter: Metadata filter
        Returns:
            List of similar documents along with a similarity score
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Get a retriever instance
        Args:
            search_type: type of search
            search_kwargs: additional search criteria
        Returns:
            LangChain retriever object
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.vector_store.delete_collection()
        logger.info(f"Deleted collection: f{self.collection_name}")

    def get_collection_stats(self) -> Dict[str, Any]:
        collection = self.vector_store._collection
        return{
            "name": self.collection_name,
            "count": collection.count(),
            "persiste_directory": self.persist_directory
        } 
        