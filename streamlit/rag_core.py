
import os
import glob
import faiss
import pickle
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_docling import DoclingLoader
import bm25s

# ---------------------------
# Configuration (change if needed)
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = str(BASE_DIR / "pdfs")
VECTOR_STORE_PATH = str(BASE_DIR / "faiss_index")  # directory where index and metadata are stored
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v4"  # <-- ИЗМЕНЕНО: Замена на Jina v4
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 15
TOP_K_FINAL = 5
OPENROUTER_MODEL = "openai/gpt-oss-120b"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Глобальные переменные для хранения моделей и индексов
embedder = None
reranker = None
faiss_index = None
chunks_metadata = []

# BM25S глобальные объекты
bm25_retriever = None
bm25_corpus_ids: List[int] = []

def initialize_models():
    """Инициализирует и кэширует модели в глобальных переменных."""
    global embedder, reranker
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    if reranker is None:
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
    Извлекает структурированный текст из PDF в markdown с сохранением форматирования,
    используя LangChain Docling loader.
    """
    try:
        loader = DoclingLoader(file_path=[pdf_path])
        docs = loader.load()
        if not docs:
            return ""
        text = "\n\n".join(d.page_content for d in docs if d.page_content and d.page_content.strip())
        return text.strip()
    except Exception:
        return ""


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
    """Создает или загружает полную базу знаний, включая dense и sparse индексы."""
    global faiss_index, chunks_metadata, bm25_retriever, bm25_corpus_ids
    initialize_models()

    ensure_dir(index_dir)
    faiss_path = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    bm25_dir = os.path.join(index_dir, "bm25")
    bm25_ids_path = os.path.join(index_dir, "bm25_ids.pkl")

    if force_rebuild:
        for path in [faiss_path, metadata_path]:
            if os.path.exists(path): os.remove(path)
        if os.path.isdir(bm25_dir):
            # Полное удаление каталога BM25 индекса
            for root, _, files in os.walk(bm25_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            os.rmdir(bm25_dir)
        if os.path.exists(bm25_ids_path):
            os.remove(bm25_ids_path)

    bm25_loaded = False
    if os.path.exists(faiss_path) and os.path.exists(metadata_path) and os.path.isdir(bm25_dir):
        faiss_index = faiss.read_index(faiss_path)
        with open(metadata_path, "rb") as f: chunks_metadata = pickle.load(f)
        # Загрузка BM25 индекса и ID корпуса
        try:
            bm25_retriever = bm25s.BM25.load(bm25_dir, load_corpus=False, mmap=False)
            if os.path.exists(bm25_ids_path):
                with open(bm25_ids_path, "rb") as f:
                    bm25_corpus_ids = pickle.load(f)
            else:
                bm25_corpus_ids = []
            bm25_loaded = bool(bm25_corpus_ids)
        except Exception:
            bm25_loaded = False
        if bm25_loaded:
            return True

    docs = load_pdfs_from_directory(pdf_dir)
    if not docs: raise RuntimeError(f"PDF-файлы не найдены в {pdf_dir}.")

    all_chunks_text, chunks_metadata = [], []
    doc_counter = 0
    for filename, text in docs.items():
        raw_chunks = chunk_text(text)
        for i, c_text in enumerate(raw_chunks):
            chunks_metadata.append({"source": filename, "chunk_index": doc_counter, "text": c_text})
            all_chunks_text.append(c_text)
            doc_counter += 1
            
    # Создание Dense (векторного) индекса
    # ИЗМЕНЕНО: Используем Jina v4 с task="retrieval" и prompt_name="passage" для документов
    embeddings = embedder.encode(
        sentences=all_chunks_text,
        task="retrieval",
        prompt_name="passage",
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    
    # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ---
    # Используем быстрый HNSW-индекс вместо медленного IndexFlatIP
    dim = embeddings.shape[1]
    M = 64
    faiss_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    faiss_index.hnsw.efSearch = 128
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, faiss_path)

    # Построение BM25S sparse-индекса
    texts_for_bm25 = []
    bm25_corpus_ids = []
    for idx, meta in enumerate(chunks_metadata):
        t = meta.get("text")
        if isinstance(t, str):
            s = t.strip()
            if s:
                texts_for_bm25.append(s)
                bm25_corpus_ids.append(idx)
    if not texts_for_bm25:
        raise RuntimeError("BM25 корпус пуст после фильтрации.")
    corpus_tokens = bm25s.tokenize(texts_for_bm25)
    bm25_retriever = bm25s.BM25(method="lucene")
    bm25_retriever.index(corpus_tokens)
    ensure_dir(bm25_dir)
    bm25_retriever.save(bm25_dir)
    with open(bm25_ids_path, "wb") as f:
        pickle.dump(bm25_corpus_ids, f)

    with open(metadata_path, "wb") as f: pickle.dump(chunks_metadata, f)
    return True

# ---------------------------
# OpenRouter call (Chat Completions)
# ---------------------------

def call_openrouter_chat_completion(api_key, model, messages, endpoint=OPENROUTER_ENDPOINT, extra_request_kwargs=None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    if extra_request_kwargs: payload.update(extra_request_kwargs)
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"OpenRouter API request failed: {e}. Response: {resp.text if 'resp' in locals() else 'No response'}")


# ---------------------------
# RAG chain creation wrapper
# ---------------------------

# Новая функция гибридного поиска, которая будет вызываться из answer_question
def hybrid_search_with_rerank(question: str) -> List[Dict[str, Any]]:
    
    # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Защита от пустого запроса ---
    if not question or not question.strip():
        return []

    global faiss_index, chunks_metadata, embedder, reranker, bm25_retriever, bm25_corpus_ids

    # 1. Dense Search
    # ИЗМЕНЕНО: Используем Jina v4 с task="retrieval" и prompt_name="query" для запросов
    q_emb = embedder.encode(
        sentences=[question], 
        task="retrieval", 
        prompt_name="query", 
        normalize_embeddings=True
    ).astype("float32")
    scores, indices = faiss_index.search(q_emb, TOP_K_RETRIEVAL)
    dense_map = {idx: score for idx, score in zip(indices[0], scores[0]) if idx != -1}

    # 2. Sparse Search (BM25S)
    sparse_map: Dict[int, float] = {}
    if bm25_retriever is not None and bm25_corpus_ids:
        query_tokens = bm25s.tokenize([question])
        results, bm25_scores = bm25_retriever.retrieve(query_tokens, k=TOP_K_RETRIEVAL, corpus=bm25_corpus_ids)
        if len(results) > 0:
            retrieved_ids = results[0]
            retrieved_scores = bm25_scores[0]
            for cid, s in zip(retrieved_ids, retrieved_scores):
                if isinstance(cid, (int, np.integer)) and 0 <= int(cid) < len(chunks_metadata):
                    sparse_map[int(cid)] = float(s)
    
    # 3. Объединение кандидатов
    all_indices = set(dense_map.keys()) | set(sparse_map.keys())
    candidates = []
    for idx in all_indices:
        if idx < len(chunks_metadata):
            candidate = chunks_metadata[idx].copy()
            candidate['dense_score'] = dense_map.get(idx, 0.0)
            candidate['sparse_score'] = sparse_map.get(idx, 0.0)
            candidates.append(candidate)

    if not candidates: return []

    # 4. Переранжирование (Reranking)
    pairs = [(question, cand['text']) for cand in candidates]
    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
    for cand, score in zip(candidates, rerank_scores):
        cand['final_score'] = float(score)

    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return candidates[:TOP_K_FINAL]

# --- ГЛАВНАЯ ОБНОВЛЕННАЯ ФУНКЦИЯ ---
def create_rag_chain(openrouter_api_key: str, openrouter_model: str = OPENROUTER_MODEL):
    """
    Создает RAG-цепочку, интегрируя перевод, гибридный поиск и переранжирование.
    """
    
    def simple_query_translation(question: str) -> str:
        """Простой перевод запроса на английский без доступа к корпусу."""
        messages = [
            {"role": "system", "content": "Translate this query to English. Do not add information."},
            {"role": "user", "content": question},
        ]
        try:
            response_json = call_openrouter_chat_completion(
                api_key=openrouter_api_key,
                model=openrouter_model,
                messages=messages,
                extra_request_kwargs={"temperature": 0.2},
            )
            text = response_json.get("choices", [])[0].get("message", {}).get("content", "").strip()
            if not text:
                text = response_json.get("choices", [])[0].get("text", "").strip()
            return text or question
        except Exception:
            return question
    
    def answer_question(question: str) -> tuple[str, str]:
        english_question = simple_query_translation(question)
        retrieved = hybrid_search_with_rerank(english_question) # <-- Теперь это безопасно
        
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
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY не найден в .env файле.")
    return key

