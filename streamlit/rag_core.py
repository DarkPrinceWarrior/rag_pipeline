import os
import multiprocessing as mp
import json
from pathlib import Path
from typing import Any

# -------------------------------------------------
# Минимальные ранние настройки среды (GPU/OMP/XLA)
# -------------------------------------------------
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
try:
    mp.set_start_method("spawn", force=True)
except Exception:
    pass
_xla_frac = os.getenv("RAG_XLA_MEM_FRACTION")
if _xla_frac:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = _xla_frac

# ---------------------------
# Конфигурация / Константы
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = str(BASE_DIR / "pdfs")
VECTOR_STORE_PATH = str(BASE_DIR / "faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 15
TOP_K_FINAL = 5
OPENROUTER_MODEL = "openai/gpt-oss-120b"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# FAISS HNSW
HNSW_M = int(os.getenv("RAG_HNSW_M", "64"))
HNSW_EF_CONSTRUCTION = int(os.getenv("RAG_HNSW_EF_CONSTRUCTION", "512"))
HNSW_EF_SEARCH_BASE = int(os.getenv("RAG_HNSW_EF_SEARCH_BASE", "128"))
HNSW_EF_SEARCH_PER_TOPK_MULT = int(os.getenv("RAG_HNSW_EF_SEARCH_PER_TOPK_MULT", "2"))
HNSW_USE_DYNAMIC_EF_SEARCH = os.getenv("RAG_HNSW_USE_DYNAMIC_EF_SEARCH", "1") not in {"0", "false", "False"}
HNSW_EF_SEARCH_MAX = int(os.getenv("RAG_HNSW_EF_SEARCH_MAX", "512"))
HNSW_FAILSAFE_RATIO = float(os.getenv("RAG_HNSW_FAILSAFE_RATIO", "0.8"))
HNSW_FAILSAFE_MAX_RETRY = int(os.getenv("RAG_HNSW_FAILSAFE_MAX_RETRY", "1"))
_cpu_threads_env = os.getenv("RAG_FAISS_CPU_THREADS")
FAISS_CPU_THREADS = (None if _cpu_threads_env in ("None", "none", "") else (int(_cpu_threads_env) if _cpu_threads_env is not None else mp.cpu_count()))
try:
    if FAISS_CPU_THREADS is not None:
        import faiss as _faiss  # локальный импорт
        _faiss.omp_set_num_threads(int(FAISS_CPU_THREADS))
except Exception:
    pass

# Веточный ретривал / RRF / реранк
TOP_K_DENSE_BRANCH = 100
TOP_K_BM25_BRANCH = 100
RRF_K = 60
TOP_K_RERANK_INPUT = 150
RRF_WEIGHT_DENSE = 1.0
RRF_WEIGHT_BM25 = 1.0

# Параметры реранкера
RERANK_MAX_TOKENS = 380
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 32

# Детекция языка
LANG_DETECT_MIN_CHARS = 40
SAME_LANG_RATIO = 0.8

# Контекст и бюджет
CONTEXT_TOP_K = int(os.getenv("RAG_CONTEXT_TOP_K", "6"))
MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))
MMR_POOL_K = int(os.getenv("RAG_MMR_POOL_K", "24"))
DUP_EMB_COS_THRESHOLD = float(os.getenv("RAG_DUP_EMB_COS_THRESHOLD", "0.90"))
DUP_CHAR_IOU_THRESHOLD = float(os.getenv("RAG_DUP_CHAR_IOU_THRESHOLD", "0.60"))
PER_DOC_CAP = int(os.getenv("RAG_PER_DOC_CAP", "2"))
PER_PAGE_CAP = int(os.getenv("RAG_PER_PAGE_CAP", "1"))
_lang_min_cover_env = os.getenv("RAG_LANG_MIN_COVER", "{\"ru\":1, \"en\":1}")
try:
    LANG_MIN_COVER = json.loads(_lang_min_cover_env)
except Exception:
    LANG_MIN_COVER = {"ru": 1, "en": 1}
CONTEXT_TOKENS_BUDGET = int(os.getenv("RAG_CONTEXT_TOKENS_BUDGET", "1200"))
SNIPPET_MAX_CHARS = int(os.getenv("RAG_SNIPPET_MAX_CHARS", "650"))
REGEN_LANG_RATIO_THRESHOLD = float(os.getenv("RAG_LANG_RATIO_THRESHOLD", "0.9"))
MAX_QUOTE_WORDS = int(os.getenv("RAG_MAX_QUOTE_WORDS", "30"))
INSUFFICIENT_MIN_RERANK_SUM = float(os.getenv("RAG_MIN_RERANK_SUM", "0.0"))

# LLM
LLM_DEFAULT_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))
LLM_DEFAULT_MAX_TOKENS = int(os.getenv("RAG_LLM_MAX_TOKENS", "550"))

# Ослабление порогов
RELAX_DUP_EMB_COS_STEP = float(os.getenv("RAG_RELAX_DUP_COS_STEP", "0.02"))
RELAX_DUP_EMB_COS_MAX = float(os.getenv("RAG_RELAX_DUP_COS_MAX", "0.96"))

# GPU настройки
def _parse_gpu_ids(val: str) -> list[int]:
    ids: list[int] = []
    for part in (val or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            ids.append(int(s))
        except ValueError:
            continue
    return ids or [0]

EMBED_GPU_IDS: list[int] = _parse_gpu_ids(os.getenv("RAG_GPU_IDS_EMBED", "0,1"))
RERANK_GPU_IDS: list[int] = _parse_gpu_ids(os.getenv("RAG_GPU_IDS_RERANK", "2"))
RERANK_GPU_ID: int = RERANK_GPU_IDS[0]
EMBED_BATCH_SIZE: int = int(os.getenv("RAG_EMBED_BATCH", "64"))

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
