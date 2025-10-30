from datasets import load_dataset, Dataset
from typing import List, Optional, Any
from langchain_core.documents import Document
import pandas as pd

class HuggingFaceLoader:
    """Load datasets from HuggingFace Hub"""
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        streaming: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            datasets_name: HF dataset identifier
            split: dataset split to load
            content_columns: Columns used for document content
            metadata_columns: Columns to store as metadata
            streaming: Whether to stream dataset
            max_samples: Maximum number of samples to load
        """
        self.dataset_name = dataset_name
        self.split = split
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.streaming = streaming
        self.max_samples = max_samples

    def load(self) -> List[Document]:
        """Load HuggingFace dataset and convert into documents"""
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming
        )
        # Limit sample size if specified
        if self.max_samples and not self.streaming:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
        # Convert to documents
        documents = []
        for idx, sample in enumerate(dataset):
            if self.max_samples and idx >= self.max_samples:
                break
            # Determine content
            if self.content_columns:
                content_parts = [
                    str(sample.get(col, ""))
                    for col in self.content_columns
                ]
                content = " ".join(content_parts)
            else:
                # Use all string fields
                content = " ".join([
                    str(v) for v in sample.values()
                    if isinstance(v, str)
                ])
            # Extract metadata 
            metadata = {
                "source": f"huggingface:{self.dataset_name}",
                "split": self.split,
                "index": idx
            }
            if self.metadata_columns:
                for col in self.metadata_columns:
                    if col in sample:
                        metadata[col] = sample[col]
            documents.append(
                Document(page_content=content, metadata=metadata)
            )
        return documents