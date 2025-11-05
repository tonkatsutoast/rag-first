from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)
from typing import List
from langchain_core.documents import Document

class DocumentChunker:
    """Advanced document chunking strategies"""

    @staticmethod
    def recursive_chunker(
        chunk_size: size = 1000,
        chunk_overlap: int = 200
    ) -> RecursiveCharacterTextSplitter:
        """Best for general text"""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n"," ", ""],
            length_function=len
        )
    
    @staticmethod
    def token_chunker(
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> TokenTextSplitter
        """Best for precise token control"""
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk=chunk_overlap
        )
    
    @staticmethod
    def semantic_chunker(documents: List[Document]) -> List[Document]:
        """
        Chunk by semantic boundaries (paragraph, sections, etc...)
        Preserves context 
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", ".", " ", ""]
        )
        return splitter.split_documents(documents)
    
# Add metadata for better filtering
def enrich_document_metadata(doc: Document) -> Document:
    """Add computed metadata fields"""
    doc.metadata.update({
        "char_count": len(doc.page_content),
        "word_count": len(doc_page_content.split()),
        "has_code": "``" in doc.page_content,
        "language": detect_language(doc.page_content)
    })
    return doc