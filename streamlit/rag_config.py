import os
import json
import multiprocessing as mp
from pathlib import Path

"""
Единый файл конфигурации RAG-пайплайна.

Содержит все настраиваемые константы, сгруппированные по областям:
- Модели
- Устройства/батчи
- Ингест/индекс
- Ретрива/фьюжн
- Реранк
- MMR/контекст
- Язык/валидация/цитаты и тексты
- LLM
"""


# ---------------------------
# Базовые пути проекта
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = str(BASE_DIR / "pdfs")
VECTOR_STORE_PATH = str(BASE_DIR / "faiss_index")


# ---------------------------
# Модели
# ---------------------------
EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "google/embeddinggemma-300m")
RERANKER_MODEL_NAME = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
OPENROUTER_MODEL = os.getenv("RAG_OPENROUTER_MODEL", "openai/gpt-oss-120b")
OPENROUTER_ENDPOINT = os.getenv("RAG_OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_TIMEOUT = int(os.getenv("RAG_OPENROUTER_TIMEOUT", "60"))


# ---------------------------
# Устройства / батчи
# ---------------------------
def _parse_gpu_ids(val: str) -> list[int]:
    """Разбор списка GPU-идентификаторов из строки окружения."""
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
DEFAULT_CUDA_DEVICE: str = os.getenv("RAG_DEFAULT_CUDA_DEVICE", "cuda:0")
EMBED_BATCH_SIZE: int = int(os.getenv("RAG_EMBED_BATCH", "64"))


# ---------------------------
# Ингест / индекс
# ---------------------------
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
CHUNK_MIN_CHARS = int(os.getenv("RAG_CHUNK_MIN_CHARS", "20"))

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
FAISS_CPU_THREADS = (
    None
    if _cpu_threads_env in ("None", "none", "")
    else (int(_cpu_threads_env) if _cpu_threads_env is not None else mp.cpu_count())
)

# Использовать ли FAISS на GPU (IndexShards). По умолчанию выключено.
FAISS_USE_GPU = os.getenv("RAG_FAISS_USE_GPU", "0") not in {"0", "false", "False"}


# ---------------------------
# Ретрива / фьюжн
# ---------------------------
TOP_K_DENSE_BRANCH = int(os.getenv("RAG_TOP_K_DENSE_BRANCH", "100"))
TOP_K_BM25_BRANCH = int(os.getenv("RAG_TOP_K_BM25_BRANCH", "100"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
TOP_K_RERANK_INPUT = int(os.getenv("RAG_TOP_K_RERANK_INPUT", "150"))
TOP_K_FINAL = int(os.getenv("RAG_TOP_K_FINAL", "5"))
RRF_WEIGHT_DENSE = float(os.getenv("RAG_RRF_WEIGHT_DENSE", "1.0"))
RRF_WEIGHT_BM25 = float(os.getenv("RAG_RRF_WEIGHT_BM25", "1.0"))

# Метки веток ретрива
RETRIEVAL_DENSE_RU = "dense_ru"
RETRIEVAL_BM25_RU = "bm25_ru"
RETRIEVAL_DENSE_EN = "dense_en"
RETRIEVAL_BM25_EN = "bm25_en"
RETRIEVAL_DENSE_ORIG = "dense_orig"
RETRIEVAL_BM25_ORIG = "bm25_orig"

# Приоритет веток при агрегации
RRF_BRANCH_PRIORITY = [
    RETRIEVAL_DENSE_RU,
    RETRIEVAL_BM25_RU,
    RETRIEVAL_DENSE_EN,
    RETRIEVAL_BM25_EN,
]


# ---------------------------
# Реранк
# ---------------------------
RERANK_MAX_TOKENS = int(os.getenv("RAG_RERANK_MAX_TOKENS", "380"))
RERANK_MAX_LENGTH = int(os.getenv("RAG_RERANK_MAX_LENGTH", "512"))
RERANK_BATCH_SIZE = int(os.getenv("RAG_RERANK_BATCH_SIZE", "32"))


# ---------------------------
# Язык / детекция / квоты
# ---------------------------
LANG_DETECT_MIN_CHARS = int(os.getenv("RAG_LANG_DETECT_MIN_CHARS", "40"))
SAME_LANG_RATIO = float(os.getenv("RAG_SAME_LANG_RATIO", "0.8"))


# ---------------------------
# MMR / контекст и бюджет
# ---------------------------
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


# ---------------------------
# Валидация языка / цитат и тексты
# ---------------------------
REGEN_LANG_RATIO_THRESHOLD = float(os.getenv("RAG_LANG_RATIO_THRESHOLD", "0.9"))
MAX_QUOTE_WORDS = int(os.getenv("RAG_MAX_QUOTE_WORDS", "30"))
INSUFFICIENT_MIN_RERANK_SUM = float(os.getenv("RAG_MIN_RERANK_SUM", "0.0"))

# Текстовые сообщения
INSUFFICIENT_PREFIX = os.getenv("RAG_INSUFFICIENT_PREFIX", "Недостаточно данных")
INSUFFICIENT_ANSWER = os.getenv(
    "RAG_INSUFFICIENT_ANSWER",
    "Недостаточно данных для надёжного ответа. Нужны релевантные источники по вашему вопросу.",
)


# ---------------------------
# LLM
# ---------------------------
LLM_DEFAULT_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))
LLM_DEFAULT_MAX_TOKENS = int(os.getenv("RAG_LLM_MAX_TOKENS", "550"))


# ---------------------------
# Ослабление порогов
# ---------------------------
RELAX_DUP_EMB_COS_STEP = float(os.getenv("RAG_RELAX_DUP_COS_STEP", "0.02"))
RELAX_DUP_EMB_COS_MAX = float(os.getenv("RAG_RELAX_DUP_COS_MAX", "0.96"))


