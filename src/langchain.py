from pathlib import Path
import random
from IPython.display import Markdown, display

from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    CSVLoader, 
    PyPDFLoader, 
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

import tiktoken
from typing import List, Union
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import Optional

from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from typing import List, Tuple

from typing import List, Union, Iterable
from langchain_core.documents import Document

# Map extensions to their specific loader classes
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}

def langchain_document_loader(tmp_dir: Path):
    documents = []
    
    for ext, loader_cls in LOADER_MAPPING.items():
        # Using DirectoryLoader for each type to handle recursive globbing
        loader = DirectoryLoader(
            str(tmp_dir), 
            glob=f"**/*{ext}", 
            loader_cls=loader_cls,
            # Pass encoding only to loaders that need/support it
            loader_kwargs={"encoding": "utf8"} if ext == ".csv" else {}
        )
        documents.extend(loader.load())
        
    return documents

import random
import json
from IPython.display import Markdown, display

def show_random_preview(docs):
    """
    Displays a formatted preview of a random document from the list.
    Handles imports internally to prevent NameErrors.
    """
    # 1. Safety check for empty list
    if not docs:
        print("❌ Error: The document list is empty.")
        return

    # 2. Select a random document
    idx = random.randint(0, len(docs) - 1)
    doc = docs[idx]
    
    # 3. Clean up content: Escape '$' to avoid Markdown math errors
    content_preview = doc.page_content[:1000].replace("$", r"\$").strip()
    
    # 4. Format metadata for readability
    metadata_formatted = json.dumps(doc.metadata, indent=2)
    
    # 5. Build the Markdown string
    md_text = (
        f"## 📄 Document Preview\n"
        f"**Index:** `{idx}` | **Source:** `{doc.metadata.get('source', 'Unknown')}`\n\n"
        f"--- \n"
        f"### **Page Content (First 1000 chars):**\n"
        f"{content_preview}...\n\n"
        f"--- \n"
        f"### **Metadata:**\n"
        f"```json\n"
        f"{metadata_formatted}\n"
        f"```"
    )
    
    # 6. Render it
    display(Markdown(md_text))

def create_text_splitter(
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
    separators: list[str] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a reusable RecursiveCharacterTextSplitter.
    
    Args:
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        separators (list[str], optional): List of separators in order of priority.
            Default is ["\n\n", "\n", " ", ""].
    
    Returns:
        RecursiveCharacterTextSplitter instance
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter


def split_documents(documents, splitter: RecursiveCharacterTextSplitter):
    """
    Split a list of documents into chunks using the provided splitter.
    
    Args:
        documents (list): List of LangChain Document objects.
        splitter (RecursiveCharacterTextSplitter): Initialized text splitter.
    
    Returns:
        list: List of text chunks
    """
    chunks = splitter.split_documents(documents)
    print(f"Number of chunks created: {len(chunks)}")
    return chunks

def tiktoken_tokens(
    documents: Union[List[Document], List[str]], 
    model: str = "gpt-3.5-turbo"
) -> List[int]:
    """
    Return the number of tokens for each document using tiktoken (OpenAI tokenizer).
    
    Args:
        documents (List[Document] | List[str]): List of LangChain Document objects or raw text strings.
        model (str): OpenAI model name to determine token encoding (default "gpt-3.5-turbo").
    
    Returns:
        List[int]: Number of tokens per document.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # fallback to default encoding if model is unknown
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_length = []
    
    for doc in documents:
        if isinstance(doc, Document):
            text = doc.page_content
        elif isinstance(doc, str):
            text = doc
        else:
            raise TypeError("Each item in documents must be a LangChain Document or a string.")
        
        tokens_length.append(len(encoding.encode(text)))
    
    return tokens_length


def select_embeddings_model(
    LLM_service: str = "OpenAI",
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    HF_key: Optional[str] = None
):
    """
    Connect to the embeddings API endpoint by specifying the embedding model.

    Args:
        LLM_service (str): Name of the service to use: "OpenAI", "Google", or "HuggingFace".
        openai_api_key (str, optional): API key for OpenAI (required if LLM_service="OpenAI").
        google_api_key (str, optional): API key for Google (required if LLM_service="Google").
        HF_key (str, optional): API key for Hugging Face (required if LLM_service="HuggingFace").

    Returns:
        embeddings object: Initialized embeddings instance for the chosen provider.

    Raises:
        ValueError: If LLM_service is unknown or the required API key is missing.
    """
    
    LLM_service = LLM_service.lower()  # Case-insensitive handling
    
    if LLM_service == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API key is required.")
        return OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
    
    elif LLM_service == "google":
        if not google_api_key:
            raise ValueError("Google API key is required.")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    
    elif LLM_service == "huggingface":
        if not HF_key:
            raise ValueError("Hugging Face API key is required.")
        return HuggingFaceInferenceAPIEmbeddings(model_name="thenlper/gte-large", api_key=HF_key)
    
    else:
        raise ValueError(f"Unknown LLM_service '{LLM_service}'. Choose from 'OpenAI', 'Google', or 'HuggingFace'.")

LOCAL_VECTOR_STORE_DIR = Path("vectorstores")  # example path, adjust as needed

def create_vectorstore(
    embeddings: Embeddings,
    documents: List[str],
    vectorstore_name: str,
    vectorstore_dir: str  # <-- new parameter for base directory
) -> Chroma:
    """
    Create a Chroma vector database and persist it to disk.

    Args:
        embeddings (Embeddings): The embedding model to use.
        documents (List[str]): List of documents to add to the vector store.
        vectorstore_name (str): Name of the vector store folder.
        vectorstore_dir (str): Base directory where the vector store will be saved.

    Returns:
        Chroma: The created Chroma vector store.
    """
    persist_directory = Path(vectorstore_dir) / vectorstore_name
    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory.as_posix()
    )
    return vector_store

def print_documents(
    docs: Iterable[Union[Document, tuple]],
    search_with_score: bool = False,
    separator: str = "-" * 100
) -> None:
    """
    Helper function to print documents in a readable format.

    Args:
        docs (Iterable[Union[Document, tuple]]): 
            A list of Document objects or tuples (Document, score).
        search_with_score (bool): 
            If True, expects tuples with scores (used in similarity_search_with_score).
        separator (str): 
            String used to separate printed documents.

    Returns:
        None
    """
    output_lines = []

    for i, doc in enumerate(docs):
        if search_with_score:
            # Expecting tuple: (Document, score)
            page_content = doc[0].page_content
            score = doc[-1]  # assuming last element is the score
            output_lines.append(
                f"Document {i+1}:\n\n{page_content}\n\nScore: {score:.3f}\n"
            )
        else:
            # Just a Document object
            output_lines.append(
                f"Document {i+1}:\n\n{doc.page_content}\n"
            )

    print(f"\n{separator}\n".join(output_lines))