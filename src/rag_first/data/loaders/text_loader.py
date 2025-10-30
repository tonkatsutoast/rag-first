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
        glob_pattern: str = "**.*.txt",
        show_progress: bool = True
    )-> List[Document]:
        """Load all text files from directory"""
        loader = DirectoryLoader(
            str(directory_path),
            glob_pattern=glob_pattern,
            loader_cls=TextLoader,
            show_progress=show_progress,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)