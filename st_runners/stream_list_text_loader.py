"""
Streamlit wrapper for TextFileLoader
Run with: streamlit run st_runners/stream_list_text_loader.py
"""

import streamlit as st
from pathlib import Path
import sys
import logging
import traceback

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_first.data.loaders.text_loader import TextFileLoader

# Set page config
st.set_page_config(
    page_title="Text File Loader",
    page_icon="document",
    layout="wide"
)

# Title
st.title("Text File Loader")
st.markdown("Interactive wrapper for loading and chunking text files")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Chunking parameters
chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100,
    help="Size of each text chunk in characters"
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=200,
    step=50,
    help="Number of overlapping characters between chunks"
)

# File type selection
st.sidebar.header("File Selection")
file_type = st.sidebar.selectbox(
    "Choose file type",
    ["Text File (.txt)", "Markdown File (.md)", "PDF File (.pdf)", "Directory (multiple files)"]
)

# Initialize loader with current settings
@st.cache_resource
def get_loader(_chunk_size, _chunk_overlap):
    """Cache the loader instance"""
    return TextFileLoader(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)

loader = get_loader(chunk_size, chunk_overlap)

# Main content area
st.header("Load Documents")

# Different inputs based on file type
if file_type == "Text File (.txt)":
    file_path = st.text_input(
        "File Path",
        value="data/raw/documentation.txt",
        help="Path to your .txt file"
    )

    if st.button("Load Text File", type="primary"):
        try:
            # Check if path is a directory
            if Path(file_path).is_dir():
                st.warning(f"'{file_path}' is a directory. Please select 'Directory (multiple files)' from the file type dropdown instead.")
                logger.warning(f"Attempted to load directory as file: {file_path}")
            else:
                logger.info(f"Loading text file: {file_path}")
                with st.spinner("Loading text file..."):
                    docs = loader.load_txt_file(file_path)
                    logger.info(f"Successfully loaded {len(docs)} document chunks")
                    st.success(f"Loaded {len(docs)} document chunks")

                    # Store in session state
                    st.session_state['documents'] = docs
                    st.session_state['file_path'] = file_path
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)
            with st.expander("Show Full Traceback"):
                st.code(traceback.format_exc())

elif file_type == "Markdown File (.md)":
    file_path = st.text_input(
        "File Path",
        value="data/raw/README.md",
        help="Path to your .md file"
    )

    if st.button("Load Markdown File", type="primary"):
        try:
            # Check if path is a directory
            if Path(file_path).is_dir():
                st.warning(f"'{file_path}' is a directory. Please select 'Directory (multiple files)' from the file type dropdown instead.")
                logger.warning(f"Attempted to load directory as file: {file_path}")
            else:
                logger.info(f"Loading markdown file: {file_path}")
                with st.spinner("Loading markdown file..."):
                    docs = loader.load_markdown_file(file_path)
                    logger.info(f"Successfully loaded {len(docs)} document chunks")
                    st.success(f"Loaded {len(docs)} document chunks")

                    st.session_state['documents'] = docs
                    st.session_state['file_path'] = file_path
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)
            with st.expander("Show Full Traceback"):
                st.code(traceback.format_exc())

elif file_type == "PDF File (.pdf)":
    file_path = st.text_input(
        "File Path",
        value="data/raw/manual.pdf",
        help="Path to your .pdf file"
    )

    if st.button("Load PDF File", type="primary"):
        try:
            # Check if path is a directory
            if Path(file_path).is_dir():
                st.warning(f"'{file_path}' is a directory. Please select 'Directory (multiple files)' from the file type dropdown instead.")
                logger.warning(f"Attempted to load directory as file: {file_path}")
            else:
                logger.info(f"Loading PDF file: {file_path}")
                with st.spinner("Loading PDF file..."):
                    docs = loader.load_pdf_file(file_path)
                    logger.info(f"Successfully loaded {len(docs)} document chunks")
                    st.success(f"Loaded {len(docs)} document chunks")

                    st.session_state['documents'] = docs
                    st.session_state['file_path'] = file_path
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)
            with st.expander("Show Full Traceback"):
                st.code(traceback.format_exc())

elif file_type == "Directory (multiple files)":
    col1, col2 = st.columns(2)

    with col1:
        directory_path = st.text_input(
            "Directory Path",
            value="data/raw/docs",
            help="Path to directory containing files"
        )

    with col2:
        glob_pattern = st.text_input(
            "Glob Pattern",
            value="**/*.md",
            help="Pattern to match files (e.g., **/*.txt, **/*.md)"
        )

    if st.button("Load Directory", type="primary"):
        try:
            logger.info(f"Loading directory: {directory_path} with pattern: {glob_pattern}")
            with st.spinner(f"Loading files from {directory_path}..."):
                docs = loader.load_directory(
                    directory_path,
                    glob_pattern=glob_pattern,
                    show_progress=True
                )
                logger.info(f"Successfully loaded {len(docs)} document chunks from directory")
                st.success(f"Loaded {len(docs)} document chunks from directory")

                st.session_state['documents'] = docs
                st.session_state['file_path'] = directory_path

        except Exception as e:
            error_msg = f"Error loading directory: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)
            st.expander("Show Full Traceback").code(traceback.format_exc())

# Display loaded documents
st.divider()

if 'documents' in st.session_state:
    st.header("Loaded Documents")

    docs = st.session_state['documents']

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", len(docs))
    with col2:
        total_chars = sum(len(doc.page_content) for doc in docs)
        st.metric("Total Characters", f"{total_chars:,}")
    with col3:
        avg_chunk = total_chars // len(docs) if docs else 0
        st.metric("Avg Chunk Size", avg_chunk)

    st.divider()

    # Document viewer
    st.subheader("Preview Documents")

    if docs:
        # Document selector
        doc_index = st.selectbox(
            "Select chunk to preview",
            range(len(docs)),
            format_func=lambda i: f"Chunk {i+1} of {len(docs)}"
        )

        selected_doc = docs[doc_index]

        # Display metadata
        st.markdown("**Metadata:**")
        st.json(selected_doc.metadata)

        # Display content
        st.markdown("**Content:**")
        st.text_area(
            "Document Content",
            value=selected_doc.page_content,
            height=300,
            disabled=True,
            label_visibility="collapsed"
        )

        # Character count for this chunk
        st.caption(f"Characters in this chunk: {len(selected_doc.page_content)}")

    st.divider()

    # Export options
    st.subheader("Export Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export as JSON"):
            import json

            export_data = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]

            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="documents.json",
                mime="application/json"
            )

    with col2:
        if st.button("Export as CSV"):
            import pandas as pd

            df = pd.DataFrame([
                {
                    "chunk_index": i,
                    "content": doc.page_content,
                    "char_count": len(doc.page_content),
                    **doc.metadata
                }
                for i, doc in enumerate(docs)
            ])

            csv_str = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_str,
                file_name="documents.csv",
                mime="text/csv"
            )

else:
    st.info("Select a file type and load documents to see preview")

# Footer
st.divider()
st.markdown("""
### Tips
- **Chunk Size**: Larger chunks preserve context but may exceed LLM limits
- **Chunk Overlap**: Helps prevent information loss at chunk boundaries
- **Glob Patterns**: Use `**/*.txt` for all .txt files recursively, `*.md` for .md files in root only
""")
