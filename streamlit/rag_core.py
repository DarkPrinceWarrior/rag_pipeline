"""
requirements.txt
----------------
langchain
langchain-community
langchain-docling
sentence-transformers
python-dotenv
faiss-cpu
requests
docling
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_docling import DoclingLoader

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
tfidf_vectorizer = None
tfidf_matrix = None
faiss_index = None
chunks_metadata = []
translation_context_glossary = ""

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

def extract_text_from_pdf_pages(pdf_path: str, num_pages: int = 5) -> str:
    """
    Извлекает приблизительно первые N страниц PDF, используя LangChain Docling loader.
    Возвращает markdown-текст.
    """
    try:
        loader = DoclingLoader(file_path=[pdf_path])
        docs = loader.load()
        if not docs:
            return ""
        full_text = "\n\n".join(d.page_content for d in docs if d.page_content and d.page_content.strip())
        if not full_text:
            return ""
        blocks = full_text.split('\n\n')
        selected = '\n\n'.join(blocks[:min(num_pages * 3, len(blocks))])
        return selected.strip()
    except Exception:
        return ""
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

def create_translation_glossary(pdf_dir: str, api_key: str) -> str:
    """Создает краткий глоссарий терминов из первых страниц документов."""
    initial_texts = []
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    for pdf_path in pdf_files[:3]: # Берем первые 3 документа для скорости
        initial_texts.append(extract_text_from_pdf_pages(pdf_path, num_pages=5))
    
    full_initial_text = "\n\n".join(initial_texts)
    
    # Ограничиваем текст, чтобы не превысить лимиты LLM
    truncated_text = full_initial_text[:8000] 

    prompt = f"""
    Analyze the following text from a document.
    Identify and list the top 10-15 key technical terms, abbreviations, and concepts.
    This list will be used as a glossary to help a translator accurately convert Russian questions into English search queries.
    Return only the list of terms.

    Text:
    \"\"\"
    {truncated_text}
    \"\"\"

    Key terms and glossary:
    """

    messages = [{"role": "system", "content": "You are a technical analyst."}, {"role": "user", "content": prompt}]
    try:
        response_json = call_openrouter_chat_completion(
            api_key, OPENROUTER_MODEL, messages, extra_request_kwargs={"max_tokens": 500, "temperature": 0.2}
        )
        glossary = response_json.get("choices", [])[0].get("message", {}).get("content", "").strip()
        return glossary
    except Exception as e:
        return ""

def build_and_load_knowledge_base(pdf_dir: str, index_dir: str, api_key: str, force_rebuild: bool = False):
    """Создает или загружает полную базу знаний, включая dense и sparse индексы."""
    global faiss_index, chunks_metadata, tfidf_vectorizer, tfidf_matrix, translation_context_glossary
    initialize_models()

    ensure_dir(index_dir)
    faiss_path = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    tfidf_vec_path = os.path.join(index_dir, "tfidf_vectorizer.pkl")
    tfidf_matrix_path = os.path.join(index_dir, "tfidf_matrix.pkl")
    glossary_path = os.path.join(index_dir, "glossary.txt")

    if force_rebuild:
        for path in [faiss_path, metadata_path, tfidf_vec_path, tfidf_matrix_path]:
            if os.path.exists(path): os.remove(path)

    if all(os.path.exists(p) for p in [faiss_path, metadata_path, tfidf_vec_path, tfidf_matrix_path]):
        faiss_index = faiss.read_index(faiss_path)
        with open(metadata_path, "rb") as f: chunks_metadata = pickle.load(f)
        with open(tfidf_vec_path, "rb") as f: tfidf_vectorizer = pickle.load(f)
        with open(tfidf_matrix_path, "rb") as f: tfidf_matrix = pickle.load(f)
        with open(glossary_path, "r", encoding="utf-8") as f: translation_context_glossary = f.read()
        return True

    docs = load_pdfs_from_directory(pdf_dir)
    if not docs: raise RuntimeError(f"PDF-файлы не найдены в {pdf_dir}.")
    
    translation_context_glossary = create_translation_glossary(pdf_dir, api_key)
    with open(glossary_path, "w", encoding="utf-8") as f:
        f.write(translation_context_glossary)

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

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_chunks_text)
    with open(tfidf_vec_path, "wb") as f: pickle.dump(tfidf_vectorizer, f)
    with open(tfidf_matrix_path, "wb") as f: pickle.dump(tfidf_matrix, f)

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

    global faiss_index, chunks_metadata, tfidf_vectorizer, tfidf_matrix, embedder, reranker

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

    # 2. Sparse Search
    q_tfidf = tfidf_vectorizer.transform([question])
    sparse_scores = cosine_similarity(q_tfidf, tfidf_matrix).flatten()
    top_sparse_indices = np.argsort(sparse_scores)[-TOP_K_RETRIEVAL:][::-1]
    sparse_map = {idx: sparse_scores[idx] for idx in top_sparse_indices if sparse_scores[idx] > 0}
    
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
    
    translation_cache = {}

    def translate_query_to_english(question: str) -> str:
        global translation_context_glossary
        if question in translation_cache: return translation_cache[question]
        
        try:
            if sum(1 for char in question if 'а' <= char.lower() <= 'я') < len(question) / 4:
                return question
        except Exception: pass

        # НОВОЕ: Улучшенный промпт с глоссарием
        prompt = f"""
        You are an expert technical translator. Your task is to translate a Russian user question into an English search query.
        Use the provided glossary to ensure high accuracy of technical terms.

        **Glossary of key terms:**
        \"\"\"
        {translation_context_glossary}
        \"\"\"

        **User question in Russian:** "{question}"

        **Instructions:**
        - Translate the Russian question into a clear English search query.
        - Prioritize correct technical terminology based on the glossary.
        - Return ONLY the translated text, without any explanations or quotation marks.

        **English search query:**
        """
        messages = [{"role": "system", "content": "You are a helpful and expert technical translator."}, {"role": "user", "content": prompt}]
        
        try:
            response_json = call_openrouter_chat_completion(
                api_key=openrouter_api_key, model=openrouter_model, messages=messages,
                extra_request_kwargs={"max_tokens": 500, "temperature": 0.2}
            )
            # --- УЛУЧШЕНИЕ: Более надежный парсинг ---
            raw_translation = response_json.get("choices", [])[0].get("message", {}).get("content", "")
            
            # Сначала убираем возможные кавычки и пробелы
            clean_translation = raw_translation.strip('" ')
            
            # Если после очистки что-то осталось, используем это
            if clean_translation:
                translation_cache[question] = clean_translation
                return clean_translation
            else:
                # Если перевод пустой, возвращаем оригинальный вопрос и кэшируем его
                translation_cache[question] = question
                return question

        except Exception as e:
            return question
    
    def answer_question(question: str) -> tuple[str, str]:
        english_question = translate_query_to_english(question)
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


