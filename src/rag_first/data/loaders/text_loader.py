from pathlib import Path
from typing import List, Optional, Dict
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextFileLoader:
    """Load various text file formats"""
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n","\n"," ", ""]
        )
    
    def load_txt_file(self, file_path: str | Path) -> List[Document]:
        """Load a plain text file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_markdown_file(self, file_path: str | Path) -> List[Document]:
        """Load Markdown file"""
        loader = UnstructuredMarkdownLoader(str(file_path))
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_pdf_file(self, file_path: str | Path) -> List[Document]:
        """Load a PDF file"""
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_directory(
        self,
        directory_path: str | Path,
        glob_pattern: str = "**/*.txt",
        show_progress: bool = True
    )-> List[Document]:
        """Load all files from directory with automatic file type detection"""
        # Determine loader class based on glob pattern
        pattern_lower = glob_pattern.lower()
        if '.pdf' in pattern_lower:
            loader_cls = PyPDFLoader
            loader_kwargs = {}
        elif '.md' in pattern_lower or '.markdown' in pattern_lower:
            loader_cls = UnstructuredMarkdownLoader
            loader_kwargs = {}
        else:
            # Default to text loader
            loader_cls = TextLoader
            loader_kwargs = {"encoding": "utf-8"}

        loader = DirectoryLoader(
            str(directory_path),
            glob=glob_pattern,
            loader_cls=loader_cls,
            show_progress=show_progress,
            loader_kwargs=loader_kwargs
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)