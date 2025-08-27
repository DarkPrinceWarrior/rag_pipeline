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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder # <-- Добавляем CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer # <-- Добавляем
from sklearn.metrics.pairwise import cosine_similarity # <-- Добавляем

# ---------------------------
# Configuration (change if needed)
# ---------------------------
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = str(BASE_DIR / "pdfs")
VECTOR_STORE_PATH = str(BASE_DIR / "faiss_index")  # directory where index and metadata are stored
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # <-- Новая модель
CHUNK_SIZE = 1000  # approx characters per chunk
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 15  # <-- Увеличиваем, чтобы реранкеру было из чего выбирать
TOP_K_FINAL = 5       # <-- Финальное количество
OPENROUTER_API_KEY = None  # loaded from .env; left None here by design
OPENROUTER_MODEL = "openai/gpt-oss-120b"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Глобальные переменные для хранения индексов и моделей
embedder = None
reranker = None
tfidf_vectorizer = None
tfidf_matrix = None
faiss_index = None
chunks_metadata = []

def initialize_models():
    """Инициализирует и кэширует модели в глобальных переменных."""
    global embedder, reranker
    if embedder is None:
        print(f"[INFO] Загрузка embedding модели '{EMBEDDING_MODEL_NAME}'...")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    if reranker is None:
        print(f"[INFO] Загрузка reranker модели '{RERANKER_MODEL_NAME}'...")
        reranker = CrossEncoder(RERANKER_MODEL_NAME)

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
    Более "умная" нарезка текста с помощью LangChain.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        # Старается резать по этим разделителям в первую очередь
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    # Отфильтровываем слишком короткие/пустые чанки
    chunks = [c.strip() for c in chunks if len(c.strip()) > 20]
    return chunks

def build_and_load_knowledge_base(pdf_dir: str, index_dir: str, force_rebuild: bool = False):
    """
    Создает или загружает полную базу знаний, включая dense и sparse индексы.
    Возвращает True, если база готова к использованию.
    """
    global faiss_index, chunks_metadata, tfidf_vectorizer, tfidf_matrix

    initialize_models() # Убедимся, что модели загружены

    # --- Пути к файлам ---
    ensure_dir(index_dir)
    faiss_path = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    tfidf_vec_path = os.path.join(index_dir, "tfidf_vectorizer.pkl")
    tfidf_matrix_path = os.path.join(index_dir, "tfidf_matrix.pkl")

    if force_rebuild:
        for path in [faiss_path, metadata_path, tfidf_vec_path, tfidf_matrix_path]:
            if os.path.exists(path):
                os.remove(path)

    # --- Загрузка из кэша ---
    if all(os.path.exists(p) for p in [faiss_path, metadata_path, tfidf_vec_path, tfidf_matrix_path]):
        print("[INFO] Загрузка существующей базы знаний...")
        faiss_index = faiss.read_index(faiss_path)
        with open(metadata_path, "rb") as f:
            chunks_metadata = pickle.load(f)
        with open(tfidf_vec_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open(tfidf_matrix_path, "rb") as f:
            tfidf_matrix = pickle.load(f)
        print("[INFO] База знаний успешно загружена.")
        return True

    # --- Создание новой базы ---
    print("[INFO] Создание новой базы знаний...")
    docs = load_pdfs_from_directory(pdf_dir)
    if not docs:
        raise RuntimeError(f"PDF-файлы не найдены в {pdf_dir}.")

    # 1. Чанкинг
    all_chunks_text = []
    chunks_metadata = []
    for doc_id, (filename, text) in enumerate(docs.items()):
        raw_chunks = chunk_text(text)
        for i, c_text in enumerate(raw_chunks):
            chunks_metadata.append({"source": filename, "chunk_index": i, "text": c_text})
            all_chunks_text.append(c_text)

    print(f"[INFO] Создано {len(all_chunks_text)} чанков.")

    # 2. Создание Dense индекса (FAISS)
    print("[INFO] Создание Dense (векторного) индекса...")
    embeddings = embedder.encode(
        all_chunks_text, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, faiss_path)

    # 3. Создание Sparse индекса (TF-IDF)
    print("[INFO] Создание Sparse (TF-IDF) индекса...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_chunks_text)
    with open(tfidf_vec_path, "wb") as f: pickle.dump(tfidf_vectorizer, f)
    with open(tfidf_matrix_path, "wb") as f: pickle.dump(tfidf_matrix, f)

    # Сохраняем метаданные
    with open(metadata_path, "wb") as f: pickle.dump(chunks_metadata, f)

    print("[INFO] База знаний успешно создана и сохранена.")
    return True

def hybrid_search(question: str) -> List[Dict[str, Any]]:
    """
    Выполняет гибридный поиск с последующим переранжированием.
    """
    global faiss_index, chunks_metadata, tfidf_vectorizer, tfidf_matrix, embedder, reranker

    # 1. Dense Search
    q_emb = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, indices = faiss_index.search(q_emb, TOP_K_RETRIEVAL)
    dense_results = {idx: score for idx, score in zip(indices[0], scores[0]) if idx != -1}

    # 2. Sparse Search
    q_tfidf = tfidf_vectorizer.transform([question])
    sparse_scores = cosine_similarity(q_tfidf, tfidf_matrix).flatten()
    top_sparse_indices = np.argsort(sparse_scores)[-TOP_K_RETRIEVAL:][::-1]
    sparse_results = {idx: sparse_scores[idx] for idx in top_sparse_indices if sparse_scores[idx] > 0}

    # 3. Объединение и сборка кандидатов
    all_indices = set(dense_results.keys()) | set(sparse_results.keys())
    
    candidates = []
    for idx in all_indices:
        candidate = chunks_metadata[idx].copy()
        candidate['dense_score'] = dense_results.get(idx, 0.0)
        candidate['sparse_score'] = sparse_results.get(idx, 0.0)
        candidates.append(candidate)

    # 4. Переранжирование (Reranking)
    pairs = [(question, cand['text']) for cand in candidates]
    if not pairs:
        return []

    rerank_scores = reranker.predict(pairs, show_progress_bar=False)

    for cand, score in zip(candidates, rerank_scores):
        cand['rerank_score'] = float(score)
        # Комбинированный финальный скор: реранкер имеет наибольший вес
        cand['final_score'] = 0.8 * cand['rerank_score'] + 0.15 * cand['dense_score'] + 0.05 * cand['sparse_score']

    # 5. Сортировка и возврат лучших
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return candidates[:TOP_K_FINAL]


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
    We use IndexHNSWFlat, a fast and accurate index for Approximate Nearest Neighbor search.
    """
    dim = embeddings.shape[1]
    
    # M - это ключевой параметр HNSW, который контролирует количество "связей" у каждой точки в графе.
    # Значение 32 или 64 является хорошим балансом между скоростью поиска и точностью.
    M = 64  
    
    # Создаем индекс HNSW. Очень важно указать faiss.METRIC_INNER_PRODUCT,
    # потому что мы используем нормализованные векторы и ищем максимальное скалярное произведение 
    # (эквивалент косинусного сходства). По умолчанию используется L2 (евклидово расстояние).
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    
    # Настраиваем параметры HNSW для лучшей производительности во время поиска
    index.hnsw.efSearch = 128 # Контролирует глубину поиска. Чем выше, тем точнее, но медленнее.
    
    print("[INFO] Building HNSW index...")
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

# --- ГЛАВНАЯ ОБНОВЛЕННАЯ ФУНКЦИЯ ---
def create_rag_chain(openrouter_api_key: str, openrouter_model: str = OPENROUTER_MODEL):
    """
    Создает RAG-цепочку, интегрируя перевод, гибридный поиск и переранжирование.
    """

    def translate_query_to_english(question: str) -> str:
        """Ваша функция перевода (без изменений)."""
        try:
            if sum(1 for char in question if 'а' <= char.lower() <= 'я') < len(question) / 4:
                print("[INFO] Запрос уже на английском, перевод не требуется.")
                return question
        except Exception: pass

        print(f"[INFO] Переводим запрос на английский: '{question}'")
        prompt = f"Translate the following user question into English. Return ONLY the translated text, without any explanation or quotation marks.\n\nUser question: \"{question}\"\n\nEnglish translation:"
        messages = [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": prompt}]
        
        try:
            response_json = call_openrouter_chat_completion(
                api_key=openrouter_api_key, model=openrouter_model, messages=messages,
                extra_request_kwargs={"max_tokens": 100, "temperature": 0.0}
            )
            translation = response_json.get("choices", [])[0].get("message", {}).get("content", "").strip()
            print(f"[INFO] Переведенный запрос: '{translation}'")
            return translation if translation else question
        except Exception as e:
            print(f"[WARN] Не удалось перевести запрос, используется оригинал: {e}")
            return question
    
    def answer_question(question: str) -> tuple[str, str]:
        # --- ШАГ 1: ПЕРЕВОД ЗАПРОСА ---
        english_question = translate_query_to_english(question)

        # --- ШАГ 2: ГИБРИДНЫЙ ПОИСК И ПЕРЕРАНЖИРОВАНИЕ ---
        # Мы вызываем новую мощную функцию вместо старого retrieve_relevant_chunks
        retrieved = hybrid_search_with_rerank(english_question, TOP_K_RETRIEVAL, TOP_K_FINAL)
        
        if not retrieved:
            return "На основе предоставленных документов я не могу найти ответ на этот вопрос.", "Контекст не был найден в базе знаний."

        # --- ШАГ 3: СБОРКА КОНТЕКСТА (с новой информацией) ---
        context_pieces = []
        for r in retrieved:
            piece = (f"---\nИсточник: {r.get('source','unknown')} [чанк {r.get('chunk_index')}]\n"
                     f"Оценка релевантности (Reranker): {r.get('final_score'):.4f}\n"
                     f"Текст:\n{r.get('text')}\n")
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

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": final_prompt}
        ]
        
        response_json = call_openrouter_chat_completion(
            api_key=openrouter_api_key, model=openrouter_model, messages=messages
        )
        try:
            reply = response_json.get("choices", [])[0].get("message", {}).get("content", "").strip()
            if not reply:
                reply = response_json.get("choices", [])[0].get("text", "")
            return reply.strip(), context
        except Exception as e:
            raise RuntimeError(f"Failed to parse OpenRouter response: {e}. Full response: {json.dumps(response_json)}")

    return { "answer_question": answer_question }


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


