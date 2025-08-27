"""
requirements.txt
----------------
langchain
langchain-community
pymupdf
sentence-transformers
python-dotenv
faiss-cpu
requests
tqdm
"""

# rag_app.py
# Full, runnable RAG script:
# - loads PDFs from a local directory
# - chunks text (chunk ~1000 chars, overlap ~200)
# - creates sentence-transformers embeddings (all-MiniLM-L6-v2)
# - stores embeddings in a local FAISS index and persists metadata
# - when a question is given: embeds it, searches FAISS, constructs context
# - calls OpenRouter chat completions endpoint with model (mistralai/mistral-medium)
#
# Notes:
# - Put your OpenRouter API key into a .env file as: OPENROUTER_API_KEY=sk-...
# - The script is intentionally modular; main functions:
#     get_or_create_vector_store(...)
#     create_rag_chain(...)
#     answer_question(...)
#
# Author: generated following specification

import os
import glob
import fitz  # PyMuPDF
import faiss
import pickle
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------
# Configuration (change if needed)
# ---------------------------
PDF_DIR = "./pdfs"
VECTOR_STORE_PATH = "./faiss_index"  # directory where index and metadata are stored
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # all-mpnet-base-v2 # jinaai/jina-embeddings-v4
CHUNK_SIZE = 1000  # approx characters per chunk
CHUNK_OVERLAP = 200
TOP_K = 5  # number of top chunks to retrieve
OPENROUTER_API_KEY = None  # loaded from .env; left None here by design
OPENROUTER_MODEL = "openai/gpt-oss-120b"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------
# PDF Loading & Text Extraction
# ---------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF.
    Returns a single string with the file's full text.
    """
    text_parts = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            # getText("text") is robust for text extraction
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)
        doc.close()
    except Exception as e:
        print(f"[WARN] Failed to parse {pdf_path}: {e}")
    return "\n".join(text_parts)


def load_pdfs_from_directory(directory: str) -> Dict[str, str]:
    """
    Loads all PDFs from the directory, extracts text.
    Returns dict: {filename: text}
    """
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    docs = {}
    for p in pdf_files:
        filename = os.path.basename(p)
        text = extract_text_from_pdf(p)
        if text.strip():
            docs[filename] = text
        else:
            print(f"[INFO] No text extracted from {filename} (may be scanned images).")
    return docs


# ---------------------------
# Chunking
# ---------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Splits text into chunks of roughly chunk_size characters with overlap.
    For production, consider sentence-aware splitting.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # move back by overlap
    # Filter out empty chunks
    chunks = [c for c in chunks if len(c) > 20]
    return chunks


# ---------------------------
# Embedding model wrapper
# ---------------------------

class LocalEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        # Loads sentence-transformers model (local)
        print(f"[INFO] Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Returns a numpy array of shape (len(texts), dim) dtype=float32
        """
        # sentence-transformers returns numpy arrays
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # ensure float32
        embeddings = embeddings.astype("float32")
        return embeddings

    def embed_text(self, text: str) -> np.ndarray:
        emb = self.embed_texts([text])[0]
        return emb


# ---------------------------
# FAISS index helpers
# ---------------------------

def _l2_normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    """
    L2-normalize rows in-place and return them (for cosine similarity using inner product).
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0.0] = 1.0
    return vecs / norms


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for the provided embeddings.
    We use IndexFlatIP (inner product) on L2-normalized vectors to perform cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # If you want to use more advanced indexes (IVF, HNSW), do so here.
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, metadata: List[Dict[str, Any]], index_dir: str):
    """
    Persist FAISS index and metadata to disk.
    """
    ensure_dir(index_dir)
    index_file = os.path.join(index_dir, "index.faiss")
    meta_file = os.path.join(index_dir, "metadata.pkl")
    print(f"[INFO] Saving FAISS index to {index_file}")
    faiss.write_index(index, index_file)
    print(f"[INFO] Saving metadata to {meta_file}")
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)


def load_faiss_index(index_dir: str) -> Tuple[Optional[faiss.Index], Optional[List[Dict[str, Any]]]]:
    """
    Load FAISS index and metadata from disk. Returns (index, metadata) or (None, None) if not found.
    """
    index_file = os.path.join(index_dir, "index.faiss")
    meta_file = os.path.join(index_dir, "metadata.pkl")
    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        return None, None
    try:
        print(f"[INFO] Loading FAISS index from {index_file}")
        index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        print(f"[ERROR] Failed to load index/metadata: {e}")
        return None, None


# ---------------------------
# Main orchestration: create or load vector store
# ---------------------------

def get_or_create_vector_store(pdf_dir: str = PDF_DIR,
                               index_dir: str = VECTOR_STORE_PATH,
                               embedder: Optional[LocalEmbedder] = None,
                               chunk_size: int = CHUNK_SIZE,
                               chunk_overlap: int = CHUNK_OVERLAP,force_rebuild: bool = False) -> Tuple[faiss.Index, List[Dict[str, Any]], LocalEmbedder]:
    """
    Main helper to either load an existing FAISS index + metadata or build a new one from PDFs.
    Returns (index, metadata, embedder).
    """
    ensure_dir(index_dir)
    if force_rebuild:
        # удаляем старые файлы если есть
        idx_file = os.path.join(index_dir, "index.faiss")
        meta_file = os.path.join(index_dir, "metadata.pkl")
        if os.path.exists(idx_file): os.remove(idx_file)
        if os.path.exists(meta_file): os.remove(meta_file)

    # Load embedder
    if embedder is None:
        embedder = LocalEmbedder(EMBEDDING_MODEL_NAME)

    # Try to load existing
    index, metadata = load_faiss_index(index_dir)
    if index is not None and metadata is not None:
        print("[INFO] Existing index loaded.")
        return index, metadata, embedder

    # Else: build fresh index
    print("[INFO] Building a new FAISS index from PDFs...")

    # 1) Load PDFs
    docs = load_pdfs_from_directory(pdf_dir)
    if not docs:
        raise RuntimeError(f"No PDF texts found in {pdf_dir}. Please add PDFs and retry.")

    # 2) Chunk documents
    chunk_texts = []
    metadata = []
    doc_id = 0
    for filename, text in docs.items():
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for i, c in enumerate(chunks):
            meta = {
                "doc_id": doc_id,
                "source": filename,
                "chunk_index": i,
                "text": c  # store full chunk text so model receives whole chunk as context
            }
            chunk_texts.append(c)
            metadata.append(meta)
        doc_id += 1

    print(f"[INFO] Created {len(chunk_texts)} chunks.")

    # 3) Embed chunks (batch)
    embeddings = []
    BATCH = 64
    for i in tqdm(range(0, len(chunk_texts), BATCH), desc="Embedding chunks"):
        batch = chunk_texts[i:i+BATCH]
        emb = embedder.embed_texts(batch)  # shape (batch_size, dim)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")
    # Normalize embeddings for cosine-sim via inner product
    embeddings = _l2_normalize_vectors(embeddings)

    # 4) Build FAISS index and persist
    index = build_faiss_index(embeddings)
    save_faiss_index(index, metadata, index_dir)

    print("[INFO] FAISS index built and saved.")
    return index, metadata, embedder


# ---------------------------
# RAG chain & retrieval
# ---------------------------

def retrieve_relevant_chunks(question: str,
                             index: faiss.Index,
                             metadata: List[Dict[str, Any]],
                             embedder: LocalEmbedder,
                             top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Given a question, embed it, query FAISS and return a list of metadata dicts for top_k results.
    Each returned dict will have keys: 'score' (similarity) and metadata fields.
    """
    q_emb = embedder.embed_text(question).astype("float32")
    # normalize
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    q_emb = q_emb.reshape(1, -1)

    D, I = index.search(q_emb, top_k)  # D: similarities (inner product), I: indices
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = dict(metadata[idx])  # copy
        meta["score"] = float(score)
        results.append(meta)
    return results


# ---------------------------
# OpenRouter call (Chat Completions)
# ---------------------------

def call_openrouter_chat_completion(api_key: str,
                                    model: str,
                                    messages: List[Dict[str, str]],
                                    endpoint: str = OPENROUTER_ENDPOINT,
                                    extra_request_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calls OpenRouter chat completions (OpenAI-compatible) and returns the JSON response.
    This function uses the /chat/completions endpoint.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    if extra_request_kwargs:
        payload.update(extra_request_kwargs)

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except Exception as e:
        # surface detailed error
        raise RuntimeError(f"OpenRouter API request failed: {e}. Response content: {resp.text}")
    return resp.json()


# ---------------------------
# RAG chain creation wrapper
# ---------------------------

def create_rag_chain(index: faiss.Index,
                     metadata: List[Dict[str, Any]],
                     embedder: LocalEmbedder,
                     openrouter_api_key: str,
                     openrouter_model: str = OPENROUTER_MODEL):
    """
    Returns an object (dict of helper functions) that can answer queries using local FAISS retrieval
    + OpenRouter LLM. The chain ensures the model uses only provided context to answer.
    """

    def answer_question(question: str, top_k: int = TOP_K) -> str:
        """
        1) retrieve top_k chunks
        2) assemble a context prompt with those chunks
        3) call OpenRouter to produce an answer constrained to the context
        """
        # 1) retrieve
        retrieved = retrieve_relevant_chunks(question, index, metadata, embedder, top_k=top_k)
        if not retrieved:
            return "No relevant context found in the vector store."

        # 2) assemble context
        # Include provenance and similarity score; order by score descending
        retrieved_sorted = sorted(retrieved, key=lambda x: x["score"], reverse=True)
        context_pieces = []
        for i, r in enumerate(retrieved_sorted):
            piece = f"---\nSource: {r.get('source','unknown')}\nChunk index: {r.get('chunk_index')}\nSimilarity: {r.get('score'):.4f}\nText:\n{r.get('text')}\n"
            context_pieces.append(piece)
        context = "\n\n".join(context_pieces)

        
        # Заранее определяем шаблон нашего профессионального промпта
        PROFESSIONAL_PROMPT_TEMPLATE = """
        <System_Role>
        Ты — AI-ассистент, специализирующийся на высокоточном извлечении ответов из текста (Question Answering). Твоя задача — отвечать на вопросы, основываясь исключительно на предоставленном ограниченном контексте.
        </System_Role>

        <Task>
        <Objective>
            Сформулировать краткий и точный ответ на `<Input_Query>`, используя только информацию из `<Knowledge_Capsule>`. Ответ должен быть на русском языке и включать ссылки на источники в заданном формате.
        </Objective>

        <Knowledge_Capsule id="CONTEXT_FOR_ANSWER">
            <Header>
            Этот блок содержит единственный источник правдивой информации для твоего ответа. Любые знания, выходящие за рамки этого контекста, должны быть полностью проигнорированы.
            </Header>
            <Corpus name="Контекст">
            {context}
            </Corpus>
            <Footer>
            Конец контекста.
            </Footer>
        </Knowledge_Capsule>

        <Input_Query name="Вопрос">
            {question}
        </Input_Query>

        <Output_Specification>
            <Audience>Конечный пользователь.</Audience>
            <Language>Русский.</Language>
            <Style>Краткий, лаконичный, по существу, не более двух абзацев.</Style>
            <Rules>
            <Rule priority="critical">
                Ответ должен быть сформирован СТРОГО на основе информации из блока `<Corpus>`.
            </Rule>
            <Rule priority="critical">
                ЗАПРЕЩЕНО использовать любые внешние знания, личный опыт или делать предположения, не подтвержденные текстом.
            </Rule>
            <Rule name="Fallback_Response">
                Если ответ на `<Input_Query>` не может быть найден в `<Corpus>`, твой ответ должен состоять ИСКЛЮЧИТЕЛЬНО из фразы: "Я не знаю на основании предоставленных документов."
            </Rule>
            <Rule name="Citation_Format">
                При цитировании информации обязательно указывай источник в формате `[имя_файла:chunk_index]`. Ссылка должна стоять в том же предложении, где используется информация из этого источника.
            </Rule>
            </Rules>
        </Output_Specification>

        <Final_Query>
            Проанализируй `<Input_Query>` и `<Knowledge_Capsule>`, после чего сгенерируй ответ, строго следуя всем правилам и форматам, указанным в `<Output_Specification>`.
        </Final_Query>
        </Task>
        ...
        """

        # В функции answer_question форматируем его с реальными данными
        final_prompt = PROFESSIONAL_PROMPT_TEMPLATE.format(context=context, question=question)

        # Отправляем в модель
        messages = [
            # Системное сообщение теперь может быть минимальным или даже отсутствовать
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": final_prompt}
        ]
        

        # 4) Call OpenRouter
        response_json = call_openrouter_chat_completion(
            api_key=openrouter_api_key,
            model=openrouter_model,
            messages=messages
        )

        # Parse model response (OpenAI-compatible structure)
        # Chat completions normally have choices[0].message.content
        try:
            choices = response_json.get("choices", [])
            if not choices:
                raise ValueError("No choices in OpenRouter response.")
            reply = choices[0].get("message", {}).get("content", "")
            if not reply:
                # Some providers return 'text' under 'choices[0].text'
                reply = choices[0].get("text", "")
            return reply.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to parse OpenRouter response: {e}. Full response: {json.dumps(response_json)}")

    # Return a small object with the function and internals for debug if needed
    return {
        "answer_question": answer_question,
        "index": index,
        "metadata": metadata,
        "embedder": embedder
    }


# ---------------------------
# Entrypoint example
# ---------------------------

def _load_api_key_from_env() -> str:
    """
    Loads OPENROUTER_API_KEY from environment (via .env if present).
    """
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY") or None
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found in environment. Create a .env file with OPENROUTER_API_KEY=your_key")
    return key


if __name__ == "__main__":
    # Example usage demonstrating full pipeline
    print("[INFO] Starting RAG pipeline demo...")

    # 0) Ensure .env loaded and API key present
    try:
        OPENROUTER_API_KEY = _load_api_key_from_env()
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Exiting. Create a .env file with OPENROUTER_API_KEY and re-run.")
        raise SystemExit(1)

    # 1) Create or load vector store
    index, metadata, embedder = get_or_create_vector_store(pdf_dir=PDF_DIR,
                                                           index_dir=VECTOR_STORE_PATH,
                                                           embedder=None,
                                                           chunk_size=CHUNK_SIZE,
                                                           chunk_overlap=CHUNK_OVERLAP,
                                                           force_rebuild=True)

    # 2) Create RAG chain
    rag = create_rag_chain(index=index,
                           metadata=metadata,
                           embedder=embedder,
                           openrouter_api_key=OPENROUTER_API_KEY,
                           openrouter_model=OPENROUTER_MODEL)

    # 3) Ask a sample question (change as needed)
    sample_question = "Расскажи про Transformer and Controller?"
    print(f"\n[INFO] Asking sample question: {sample_question}\n")
    try:
        answer = rag["answer_question"](sample_question, top_k=TOP_K)
        print("\n[--- MODEL ANSWER ---]\n")
        print(answer)
        print("\n[--- END ---]\n")
    except Exception as e:
        print(f"[ERROR] Failed to get answer: {e}")
        # For debugging: print some retrieved contexts
        try:
            retrieved = retrieve_relevant_chunks(sample_question, index, metadata, embedder, top_k=TOP_K)
            print("[DEBUG] Retrieved contexts (top):")
            for r in retrieved:
                print(f"Source={r.get('source')}, chunk_index={r.get('chunk_index')}, score={r.get('score'):.4f}")
        except Exception as e2:
            print(f"[DEBUG] Retrieval also failed: {e2}")
        raise

