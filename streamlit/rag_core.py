import os
from pathlib import Path
from typing import Any

from rag_config import (
    BASE_DIR,
    PDF_DIR,
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MIN_CHARS,
    OPENROUTER_MODEL,
    OPENROUTER_ENDPOINT,
    OPENROUTER_TIMEOUT,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH_BASE,
    HNSW_EF_SEARCH_PER_TOPK_MULT,
    HNSW_USE_DYNAMIC_EF_SEARCH,
    HNSW_EF_SEARCH_MAX,
    HNSW_FAILSAFE_RATIO,
    HNSW_FAILSAFE_MAX_RETRY,
    FAISS_CPU_THREADS,
    TOP_K_DENSE_BRANCH,
    TOP_K_BM25_BRANCH,
    RRF_K,
    TOP_K_RERANK_INPUT,
    TOP_K_FINAL,
    RRF_WEIGHT_DENSE,
    RRF_WEIGHT_BM25,
    RERANK_MAX_TOKENS,
    RERANK_MAX_LENGTH,
    RERANK_BATCH_SIZE,
    LANG_DETECT_MIN_CHARS,
    SAME_LANG_RATIO,
    CONTEXT_TOP_K,
    MMR_LAMBDA,
    MMR_POOL_K,
    DUP_EMB_COS_THRESHOLD,
    DUP_CHAR_IOU_THRESHOLD,
    PER_DOC_CAP,
    PER_PAGE_CAP,
    LANG_MIN_COVER,
    CONTEXT_TOKENS_BUDGET,
    SNIPPET_MAX_CHARS,
    REGEN_LANG_RATIO_THRESHOLD,
    MAX_QUOTE_WORDS,
    INSUFFICIENT_PREFIX,
    INSUFFICIENT_ANSWER,
    INSUFFICIENT_MIN_RERANK_SUM,
    LLM_DEFAULT_TEMPERATURE,
    LLM_DEFAULT_MAX_TOKENS,
    RELAX_DUP_EMB_COS_STEP,
    RELAX_DUP_EMB_COS_MAX,
    EMBED_GPU_IDS,
    RERANK_GPU_IDS,
    RERANK_GPU_ID,
    EMBED_BATCH_SIZE,
)

# Рантайм-настройки генерации / перевода (устанавливаются в create_rag_chain)
runtime_openrouter_api_key: str | None = None
runtime_openrouter_model: str | None = None
runtime_llm_temperature: float | None = None
runtime_llm_max_tokens: int | None = None
runtime_enforce_citations: bool = True
runtime_language_enforcement: bool = True

# ---------------------------
# Глобальное состояние (индексы/модели/инструменты)
# ---------------------------
embedder = None
reranker = None
faiss_index = None
chunks_metadata: list[dict[str, Any]] = []
EMB_MATRIX = None

bm25_retriever = None
bm25_corpus_ids: list[int] = []

language_detector = None
ru_stemmer = None
en_stemmer = None
ru_stopwords: set[str] = set()
en_stopwords: set[str] = set()

# ---------------------------
# Утилиты общего назначения
# ---------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def debug_log(message: str, *args: Any) -> None:
    """Условное debug-логирование при RAG_DEBUG=1."""
    if os.getenv("RAG_DEBUG") != "1":
        return
    try:
        import logging
        logging.getLogger(__name__).debug(message, *args)
    except Exception:
        pass

def format_citation(meta: dict[str, Any]) -> str:
    """Формат ссылки на источник из метаданных чанка."""
    source = meta.get("source")
    page = meta.get("page")
    c0 = meta.get("char_start")
    c1 = meta.get("char_end")
    if isinstance(page, int) and page > 0:
        if isinstance(c0, int) and isinstance(c1, int):
            return f"[{source} p.{page} {c0}–{c1}]"
        return f"[{source} p.{page}]"
    return f"[{source}:{meta.get('chunk_index')}]"

# ---------------------------
# Публичный API (прокси на модули)
# ---------------------------
def build_and_load_knowledge_base(pdf_dir: str, index_dir: str, force_rebuild: bool = False) -> bool:
    """Построение/загрузка базы знаний (ingestion)."""
    from rag_ingestion import build_and_load_knowledge_base as _impl
    return _impl(pdf_dir, index_dir, force_rebuild)

def hybrid_search_with_rerank(question: str, apply_lang_quota: bool = True):
    """Гибридный поиск + RRF + реранк (pipeline)."""
    from rag_pipeline import hybrid_search_with_rerank as _impl
    return _impl(question, apply_lang_quota=apply_lang_quota)

def create_rag_chain(
    openrouter_api_key: str,
    openrouter_model: str = OPENROUTER_MODEL,
    temperature: float | None = None,
    max_tokens: int | None = None,
    enforce_citations: bool = True,
    language_enforcement: bool = True,
):
    """Создаёт RAG-цепочку (pipeline)."""
    from rag_pipeline import create_rag_chain as _impl
    return _impl(
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        temperature=temperature,
        max_tokens=max_tokens,
        enforce_citations=enforce_citations,
        language_enforcement=language_enforcement,
    )

def _estimate_tokens(text: str | None, lang: str | None) -> int:
    """Прокси на оценку числа токенов (pipeline)."""
    from rag_pipeline import _estimate_tokens as _impl
    return _impl(text, lang)

# ---------------------------
# Утилита загрузки API-ключа из .env
# ---------------------------
def _load_api_key_from_env() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY не найден в .env файле.")
    return key
