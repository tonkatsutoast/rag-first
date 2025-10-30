import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

class CSVLoader:
    """Load and process CSV files into LangChain documents"""
    def __init__(
        self, 
        file_path: str | Path,
        content_columns: List[str],
        metadata_columns: Optional[List[str]] = None,
        encoding: str = "utf-8"
    ):
        """
        Args:
            file_path: Path to CSV file
            content_columns : columns to combine for data content
            metadata_columns: columns to store metadata
            encoding: File encoding 
        """
        self.file_path = Path(file_path)
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load CSV and convert to documents"""
        df = pd.read_csv(self.file_path, encoding=self.encoding)
        return self._dataframe_to_documents(df)
    
    def _dataframe_to_documents(self, df: pd.DataFrame) -> List[Document]:
        """Concert DataFrame to LangChain documents"""
        documents = []
        for idx, row in df.iterrows():
            # Combine content columns
            content_parts = [
                str(row[col]) for col in self.content_columns
                if pd.notna(row[col])
            ]
            content = " ".join(content_parts)

            # Extract the metadata
            metadata = {
                col: row[col] for col in self.metadata_columns
                if col in df.columns and pd.notna(row[col])
            }
            metadata["source"] = str(self.file_path)
            metadata["row_index"] = idx
            documents.append(
                Document(page_content=content, metadata=metadata)
            )
        return documents
    

