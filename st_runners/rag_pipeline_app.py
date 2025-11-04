"""
Streamlit RAG Pipeline App
Load documents, create embeddings, store in ChromaDB, and query with RAG
Run with: streamlit run st_runners/rag_pipeline_app.py
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
from rag_first.embeddings.embedding_manager import get_embeddings
from rag_first.vectorstore.chroma_manager import ChromaManager
from rag_first.llm.cloud_llm import get_llm
from rag_first.config.settings import settings

# Set page config
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="book",
    layout="wide"
)

# Title
st.title("RAG Pipeline - Load, Index, and Query")
st.markdown("Complete workflow: Load documents → Create embeddings → Store in ChromaDB → Query with LLM")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Step selection
step = st.sidebar.radio(
    "Select Step",
    ["1. Load Documents", "2. Create Vector Store", "3. Query RAG System"],
    help="Follow steps in order for first-time setup"
)

st.sidebar.divider()

# ==================== STEP 1: LOAD DOCUMENTS ====================
if step == "1. Load Documents":
    st.header("Step 1: Load Documents")
    st.markdown("Load and chunk your documents from files or directories")

    # Chunking parameters
    with st.expander("Chunking Settings", expanded=False):
        chunk_size = st.slider("Chunk Size", 100, 3000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)

    # Initialize loader
    loader = TextFileLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # File type selection
    load_type = st.radio(
        "Choose how to load documents",
        ["Single File", "Directory"],
        horizontal=True
    )

    if load_type == "Single File":
        col1, col2 = st.columns([3, 1])
        with col1:
            file_path = st.text_input("File Path", value="data/raw/baseball/World_Series.pdf")
        with col2:
            file_type = st.selectbox("File Type", [".txt", ".md", ".pdf"])

        if st.button("Load File", type="primary"):
            try:
                with st.spinner(f"Loading {file_type} file..."):
                    if file_type == ".txt":
                        docs = loader.load_txt_file(file_path)
                    elif file_type == ".md":
                        docs = loader.load_markdown_file(file_path)
                    elif file_type == ".pdf":
                        docs = loader.load_pdf_file(file_path)

                    st.session_state['documents'] = docs
                    st.session_state['source_info'] = f"File: {file_path}"
                    st.success(f"Loaded {len(docs)} document chunks")
                    logger.info(f"Loaded {len(docs)} chunks from {file_path}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(traceback.format_exc())

    else:  # Directory
        col1, col2 = st.columns(2)
        with col1:
            directory_path = st.text_input("Directory Path", value="data/raw/baseball")
        with col2:
            glob_pattern = st.text_input("Glob Pattern", value="**/*.pdf")

        if st.button("Load Directory", type="primary"):
            try:
                with st.spinner(f"Loading files from {directory_path}..."):
                    docs = loader.load_directory(
                        directory_path,
                        glob_pattern=glob_pattern,
                        show_progress=True
                    )
                    st.session_state['documents'] = docs
                    st.session_state['source_info'] = f"Directory: {directory_path} ({glob_pattern})"
                    st.success(f"Loaded {len(docs)} document chunks from directory")
                    logger.info(f"Loaded {len(docs)} chunks from {directory_path}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(traceback.format_exc())

    # Display loaded documents
    if 'documents' in st.session_state:
        st.divider()
        docs = st.session_state['documents']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", len(docs))
        with col2:
            total_chars = sum(len(doc.page_content) for doc in docs)
            st.metric("Total Characters", f"{total_chars:,}")
        with col3:
            avg_chunk = total_chars // len(docs) if docs else 0
            st.metric("Avg Chunk Size", avg_chunk)

        st.info(f"Source: {st.session_state.get('source_info', 'Unknown')}")
        st.markdown("**Next:** Go to 'Step 2: Create Vector Store' to index these documents")

# ==================== STEP 2: CREATE VECTOR STORE ====================
elif step == "2. Create Vector Store":
    st.header("Step 2: Create Vector Store")
    st.markdown("Create embeddings and store documents in ChromaDB for semantic search")

    if 'documents' not in st.session_state:
        st.warning("No documents loaded. Please go to Step 1 first.")
        st.stop()

    docs = st.session_state['documents']
    st.info(f"Ready to index {len(docs)} document chunks")

    # Embedding settings
    st.subheader("Embedding Configuration")

    col1, col2 = st.columns(2)
    with col1:
        embedding_provider = st.selectbox(
            "Embedding Provider",
            ["ollama", "openai", "huggingface"],
            help="Ollama is free and local, OpenAI requires API key"
        )
    with col2:
        if embedding_provider == "ollama":
            embedding_model = st.text_input("Model", value="mxbai-embed-large")
        elif embedding_provider == "openai":
            embedding_model = st.text_input("Model", value="text-embedding-3-small")
        else:
            embedding_model = st.text_input("Model", value="sentence-transformers/all-MiniLM-L6-v2")

    # Collection settings
    collection_name = st.text_input(
        "Collection Name",
        value="baseball_docs",
        help="Name for this ChromaDB collection"
    )

    if st.button("Create Vector Store", type="primary"):
        try:
            with st.spinner("Creating embeddings and storing in ChromaDB..."):
                # Initialize embeddings
                logger.info(f"Initializing {embedding_provider} embeddings with model {embedding_model}")
                embeddings = get_embeddings(provider=embedding_provider, model=embedding_model)
                st.write(f"Initialized {embedding_provider} embeddings")

                # Initialize ChromaDB
                logger.info(f"Creating ChromaDB collection: {collection_name}")
                chroma = ChromaManager(
                    embedding_function=embeddings,
                    persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
                    collection_name=collection_name
                )
                st.write(f"Initialized ChromaDB at: {settings.CHROMA_PERSIST_DIRECTORY}")

                # Add documents
                logger.info(f"Adding {len(docs)} documents to vector store")
                doc_ids = chroma.add_documents(docs, batch_size=50)

                # Save to session state
                st.session_state['chroma_manager'] = chroma
                st.session_state['collection_name'] = collection_name
                st.session_state['embedding_provider'] = embedding_provider

                st.success(f"Successfully created vector store with {len(doc_ids)} document chunks")
                logger.info(f"Vector store created successfully")

                # Show stats
                stats = chroma.get_collection_stats()
                st.json(stats)

                st.markdown("**Next:** Go to 'Step 3: Query RAG System' to ask questions")

        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            with st.expander("Show Full Traceback"):
                st.code(traceback.format_exc())
            logger.error(traceback.format_exc())

# ==================== STEP 3: QUERY RAG SYSTEM ====================
elif step == "3. Query RAG System":
    st.header("Step 3: Query RAG System")
    st.markdown("Ask questions and get AI-generated answers based on your documents")

    # Check if vector store exists
    if 'chroma_manager' not in st.session_state:
        st.warning("No vector store found. Please complete Step 2 first, or load an existing collection below.")

        # Option to load existing collection
        with st.expander("Load Existing Collection"):
            col1, col2 = st.columns(2)
            with col1:
                collection_name = st.text_input("Collection Name", value="baseball_docs")
            with col2:
                embedding_provider = st.selectbox("Embedding Provider", ["ollama", "openai", "huggingface"])

            if st.button("Load Collection"):
                try:
                    embeddings = get_embeddings(provider=embedding_provider)
                    chroma = ChromaManager(
                        embedding_function=embeddings,
                        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
                        collection_name=collection_name
                    )
                    st.session_state['chroma_manager'] = chroma
                    st.session_state['collection_name'] = collection_name
                    st.session_state['embedding_provider'] = embedding_provider
                    st.success(f"Loaded collection: {collection_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading collection: {str(e)}")
        st.stop()

    chroma = st.session_state['chroma_manager']

    # Display collection info
    stats = chroma.get_collection_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Collection", stats['name'])
    with col2:
        st.metric("Documents", stats['count'])

    st.divider()

    # LLM settings
    st.subheader("LLM Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        llm_provider = st.selectbox(
            "LLM Provider",
            ["ollama", "openai", "anthropic", "google"],
            help="Choose your language model provider"
        )
    with col2:
        if llm_provider == "ollama":
            llm_model = st.text_input("Model", value="llama3.2")
        elif llm_provider == "openai":
            llm_model = st.text_input("Model", value="gpt-4o-mini")
        elif llm_provider == "anthropic":
            llm_model = st.text_input("Model", value="claude-3-5-sonnet-20241022")
        else:
            llm_model = st.text_input("Model", value="gemini-1.5-flash")
    with col3:
        top_k = st.number_input("Top K Results", 1, 20, 5, help="Number of relevant chunks to retrieve")

    st.divider()

    # Query interface
    st.subheader("Ask Questions")

    question = st.text_input(
        "Your Question",
        placeholder="e.g., What teams have won the most World Series?",
        key="question_input"
    )

    if st.button("Get Answer", type="primary") and question:
        try:
            with st.spinner("Searching documents and generating answer..."):
                # Initialize LLM
                logger.info(f"Initializing {llm_provider} LLM")
                llm = get_llm(provider=llm_provider, model=llm_model)

                # Search for relevant documents
                logger.info(f"Searching for: {question}")
                results = chroma.similarity_search_with_score(question, k=top_k)

                if not results:
                    st.warning("No relevant documents found")
                    st.stop()

                # Display retrieved documents
                with st.expander(f"Retrieved Documents ({len(results)} chunks)"):
                    for i, (doc, score) in enumerate(results, 1):
                        st.markdown(f"**Chunk {i}** (similarity: {score:.3f})")
                        st.text_area(
                            "Content",
                            value=doc.page_content,
                            height=150,
                            key=f"retrieved_{i}",
                            label_visibility="collapsed"
                        )
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                        st.divider()

                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc, _ in results])

                # Create prompt
                prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""

                # Generate answer
                logger.info("Generating answer with LLM")
                answer = llm.invoke(prompt)

                # Display answer
                st.subheader("Answer")
                if hasattr(answer, 'content'):
                    st.markdown(answer.content)
                else:
                    st.markdown(str(answer))

                logger.info("Answer generated successfully")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            with st.expander("Show Full Traceback"):
                st.code(traceback.format_exc())
            logger.error(traceback.format_exc())

    st.divider()

    # Example questions
    with st.expander("Example Questions"):
        st.markdown("""
        Try asking questions like:
        - What is this document about?
        - Summarize the main points
        - What are the key findings?
        - Who are the main people/teams mentioned?
        """)

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
### Workflow
1. Load documents from files/directories
2. Create embeddings and vector store
3. Query with natural language

### Tips
- Use Ollama for free local models
- Larger chunk sizes preserve context
- Higher Top K retrieves more documents
""")
