
import os
import multiprocessing as mp
# -------------------------------------------------
# Конфигурация GPU/Видеопамяти до импортов
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
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from langchain_docling import DoclingLoader
import bm25s

# ---------------------------
# Configuration (change if needed)
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = str(BASE_DIR / "pdfs")
VECTOR_STORE_PATH = str(BASE_DIR / "faiss_index")  # directory where index and metadata are stored
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 15
TOP_K_FINAL = 5
OPENROUTER_MODEL = "openai/gpt-oss-120b"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Параметры веточного ретрива (RU/EN)
TOP_K_DENSE_BRANCH = 100
TOP_K_BM25_BRANCH = 100

# Параметры RRF и входа в реранкер
RRF_K = 60
TOP_K_RERANK_INPUT = 150
RRF_WEIGHT_DENSE = 1.0
RRF_WEIGHT_BM25 = 1.0

# Ограничение длины текста кандидата для реранка (примерно 350–400 токенов)
RERANK_MAX_TOKENS = 380
RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 32

# Рантайм-параметры для перевода (устанавливаются при создании цепочки)
runtime_openrouter_api_key: str | None = None
runtime_openrouter_model: str | None = None

# ---------------------------
# GPU-настройки (задаются через переменные окружения)
# ---------------------------

def _parse_gpu_ids(val: str) -> List[int]:
    ids: List[int] = []
    for part in (val or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            ids.append(int(s))
        except ValueError:
            continue
    return ids or [0]

# Список GPU для эмбеддинга и реранка
EMBED_GPU_IDS: List[int] = _parse_gpu_ids(os.getenv("RAG_GPU_IDS_EMBED", "0,1"))
RERANK_GPU_IDS: List[int] = _parse_gpu_ids(os.getenv("RAG_GPU_IDS_RERANK", "2"))
RERANK_GPU_ID: int = RERANK_GPU_IDS[0]
EMBED_BATCH_SIZE: int = int(os.getenv("RAG_EMBED_BATCH", "64"))

# ---------------------------
# Простой перевод RU → EN без доступа к корпусу
# ---------------------------

def simple_query_translation(question: str) -> str:
    """Простой машинный перевод запроса на английский без обращения к корпусу."""
    if not question or not question.strip():
        return ""
    if not runtime_openrouter_api_key or not runtime_openrouter_model:
        return question
    messages = [
        {"role": "system", "content": "Translate this query to English. Do not add information."},
        {"role": "user", "content": question},
    ]
    try:
        response_json = call_openrouter_chat_completion(
            api_key=runtime_openrouter_api_key,
            model=runtime_openrouter_model,
            messages=messages,
            extra_request_kwargs={"temperature": 0.2},
        )
        text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not text:
            text = response_json.get("choices", [{}])[0].get("text", "").strip()
        return text or question
    except Exception:
        return question

# Глобальные переменные для хранения моделей и индексов
embedder = None
reranker = None
faiss_index = None
chunks_metadata = []

# BM25S глобальные объекты
bm25_retriever = None
bm25_corpus_ids: List[int] = []

# ---------------------------
# Вспомогательное: мульти-GPU энкодинг эмбеддингов
# ---------------------------

def _chunk_iter(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i+size] for i in range(0, len(items), size)]

def _worker_encode(args):
    """Процесс-воркер: кодирует партию на выделенном GPU.

    Параметры: (device_id, model_name, texts)
    Возврат: np.ndarray float32
    """
    device_id, model_name, texts = args
    from sentence_transformers import SentenceTransformer  # локальный импорт в процессе
    import torch  # type: ignore
    torch.cuda.set_device(device_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")
    emb = model.encode(
        sentences=texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return emb

def _encode_multi_gpu(texts: List[str], batch_size: int, gpu_ids: List[int]) -> np.ndarray:
    """Распределённое кодирование: делим тексты на партии и рассылку по GPU.

    Баланс простой round-robin по списку gpu_ids. Собираем эмбеддинги в исходном порядке.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    batches = _chunk_iter(texts, batch_size)
    tasks = []
    for i, b in enumerate(batches):
        device_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((device_id, EMBEDDING_MODEL_NAME, b))

    # Параллельный запуск воркеров
    from multiprocessing import Pool
    with Pool(processes=min(len(tasks), len(gpu_ids))) as pool:
        results = pool.map(_worker_encode, tasks)

    # Конкатенируем в исходном порядке партий
    emb = np.concatenate(results, axis=0)
    return emb

# ---------------------------
# Единый формат кандидата для ретрива
# ---------------------------

# Метки веток ретрива (используются для журналирования и последующего RRF)
RETRIEVAL_DENSE_RU = "dense_ru"
RETRIEVAL_BM25_RU = "bm25_ru"
RETRIEVAL_DENSE_EN = "dense_en"
RETRIEVAL_BM25_EN = "bm25_en"

def build_candidate_dict(chunk_global_id: int, retrieval_label: str, rank: int, score_raw: float) -> Dict[str, Any]:
    """Создает единый словарь для кандидата документа.

    Параметры:
    - chunk_global_id: глобальный индекс чанка в списке chunks_metadata
    - retrieval_label: одна из меток {dense_ru,bm25_ru,dense_en,bm25_en}
    - rank: позиция кандидата в своем списке (начиная с 1)
    - score_raw: сырой скор соответствующего ретривера (без нормализации)
    """
    if not isinstance(chunk_global_id, (int, np.integer)) or chunk_global_id < 0 or chunk_global_id >= len(chunks_metadata):
        raise ValueError("Некорректный chunk_global_id для кандидата.")
    meta = chunks_metadata[chunk_global_id]
    return {
        "chunk_id": int(chunk_global_id),
        "source": meta.get("source"),
        "chunk_index": meta.get("chunk_index"),
        "page": meta.get("page") if isinstance(meta, dict) and "page" in meta else None,
        "text_ref": meta.get("text"),
        "citation": format_citation(meta),
        "retrieval": retrieval_label,
        "rank": int(rank),
        "score_raw": float(score_raw),
    }

def build_candidates_from_arrays(indices: List[int], scores: List[float], retrieval_label: str) -> List[Dict[str, Any]]:
    """Создает список кандидатов в едином формате по массивам индексов и скоров.

    Возвращает список, отсортированный по исходному порядку (рангам) без нормализации.
    """
    candidates: List[Dict[str, Any]] = []
    current_rank = 1
    for idx, s in zip(indices, scores):
        if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < len(chunks_metadata):
            candidates.append(build_candidate_dict(int(idx), retrieval_label, current_rank, float(s)))
            current_rank += 1
    return candidates

# ---------------------------
# Двухветочный ретривал (RU и EN) с четырьмя списками кандидатов
# ---------------------------

def collect_candidates_ru_en(question: str, k_dense: int = TOP_K_DENSE_BRANCH, k_bm25: int = TOP_K_BM25_BRANCH) -> Dict[str, List[Dict[str, Any]]]:
    """Возвращает 4 независимых списка кандидатов: dense_ru, bm25_ru, dense_en, bm25_en.

    Формат каждого кандидата соответствует build_candidate_dict.
    """
    if not question or not question.strip():
        return {"dense_ru": [], "bm25_ru": [], "dense_en": [], "bm25_en": []}

    global faiss_index, chunks_metadata, embedder, bm25_retriever, bm25_corpus_ids

    # RU: Dense
    # Кодируем вопрос на первом доступном GPU из EMBED_GPU_IDS
    q_ru_emb = embedder.encode(
        sentences=[question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    k_dense_eff = min(k_dense, faiss_index.ntotal if faiss_index is not None else 0)
    dense_ru: List[Dict[str, Any]] = []
    if k_dense_eff > 0:
        ru_scores, ru_indices = faiss_index.search(q_ru_emb, k_dense_eff)
        dense_ru = build_candidates_from_arrays(list(ru_indices[0]), list(ru_scores[0]), RETRIEVAL_DENSE_RU)

    # RU: BM25S
    bm25_ru: List[Dict[str, Any]] = []
    if bm25_retriever is not None and bm25_corpus_ids:
        ru_query_tokens = bm25s.tokenize([question])
        ru_results, ru_bm25_scores = bm25_retriever.retrieve(ru_query_tokens, k=k_bm25, corpus=bm25_corpus_ids)
        if len(ru_results) > 0:
            bm25_ru = build_candidates_from_arrays(list(ru_results[0]), list(ru_bm25_scores[0]), RETRIEVAL_BM25_RU)

    # EN translation
    en_question = simple_query_translation(question)

    # EN: Dense
    q_en_emb = embedder.encode(
        sentences=[en_question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    dense_en: List[Dict[str, Any]] = []
    if k_dense_eff > 0:
        en_scores, en_indices = faiss_index.search(q_en_emb, k_dense_eff)
        dense_en = build_candidates_from_arrays(list(en_indices[0]), list(en_scores[0]), RETRIEVAL_DENSE_EN)

    # EN: BM25S
    bm25_en: List[Dict[str, Any]] = []
    if bm25_retriever is not None and bm25_corpus_ids:
        en_query_tokens = bm25s.tokenize([en_question])
        en_results, en_bm25_scores = bm25_retriever.retrieve(en_query_tokens, k=k_bm25, corpus=bm25_corpus_ids)
        if len(en_results) > 0:
            bm25_en = build_candidates_from_arrays(list(en_results[0]), list(en_bm25_scores[0]), RETRIEVAL_BM25_EN)

    result = {
        "dense_ru": dense_ru,
        "bm25_ru": bm25_ru,
        "dense_en": dense_en,
        "bm25_en": bm25_en,
    }
    # Мягкая дедупликация: только логирование пересечений
    log_intersections_debug(result)
    return result

def _compute_intersections_report(cands: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Вычисляет пересечения по chunk_id между всеми парами списков и готовит debug-отчет.

    Возвращает словарь с метриками по парам и примерами дубликатов.
    """
    branches = [RETRIEVAL_DENSE_RU, RETRIEVAL_BM25_RU, RETRIEVAL_DENSE_EN, RETRIEVAL_BM25_EN]
    name_to_list = {
        RETRIEVAL_DENSE_RU: cands.get("dense_ru", []),
        RETRIEVAL_BM25_RU: cands.get("bm25_ru", []),
        RETRIEVAL_DENSE_EN: cands.get("dense_en", []),
        RETRIEVAL_BM25_EN: cands.get("bm25_en", []),
    }
    name_to_idset = {name: {c.get("chunk_id") for c in lst} for name, lst in name_to_list.items()}
    name_to_idmap = {name: {c.get("chunk_id"): c for c in lst} for name, lst in name_to_list.items()}

    pair_stats: List[Dict[str, Any]] = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            a = branches[i]
            b = branches[j]
            set_a = name_to_idset[a]
            set_b = name_to_idset[b]
            inter = list(set_a & set_b)
            size_a = len(set_a)
            size_b = len(set_b)
            union = len(set_a | set_b)
            share_min = (len(inter) / max(1, min(size_a, size_b))) if (size_a > 0 and size_b > 0) else 0.0
            share_union = (len(inter) / max(1, union))
            # Примеры дубликатов (до 5 штук)
            examples = []
            for cid in inter[:5]:
                ca = name_to_idmap[a].get(cid)
                cb = name_to_idmap[b].get(cid)
                if ca and cb:
                    examples.append({
                        "chunk_id": cid,
                        "a": {"retrieval": ca.get("retrieval"), "rank": ca.get("rank"), "score_raw": ca.get("score_raw")},
                        "b": {"retrieval": cb.get("retrieval"), "rank": cb.get("rank"), "score_raw": cb.get("score_raw")},
                    })
            pair_stats.append({
                "pair": [a, b],
                "intersection": len(inter),
                "size_a": size_a,
                "size_b": size_b,
                "share_min": share_min,
                "share_union": share_union,
                "examples": examples,
            })

    return {"pairs": pair_stats}

def log_intersections_debug(cands: Dict[str, List[Dict[str, Any]]]) -> None:
    """Опционально выводит debug-отчет о пересечениях кандидатов при RAG_DEBUG=1.

    Никаких побочных эффектов, если переменная окружения не установлена.
    """
    if os.getenv("RAG_DEBUG") != "1":
        return
    report = _compute_intersections_report(cands)
    try:
        import logging
        logger = logging.getLogger(__name__)
        for p in report.get("pairs", []):
            logger.debug(
                "[DEDUP] pair=%s vs %s | inter=%d | size_a=%d | size_b=%d | share_min=%.3f | share_union=%.3f",
                p.get("pair", [None, None])[0], p.get("pair", [None, None])[1], p.get("intersection", 0),
                p.get("size_a", 0), p.get("size_b", 0), p.get("share_min", 0.0), p.get("share_union", 0.0)
            )
            for ex in p.get("examples", []):
                logger.debug(
                    "[DEDUP_EX] chunk_id=%s | A(retr=%s,rank=%s,score=%.6f) | B(retr=%s,rank=%s,score=%.6f)",
                    str(ex.get("chunk_id")),
                    ex.get("a", {}).get("retrieval"), ex.get("a", {}).get("rank"), float(ex.get("a", {}).get("score_raw", 0.0)),
                    ex.get("b", {}).get("retrieval"), ex.get("b", {}).get("rank"), float(ex.get("b", {}).get("score_raw", 0.0)),
                )
    except Exception:
        # Безопасный no-op в случае проблем с логгером
        pass

# ---------------------------
# RRF-слияние четырех списков кандидатов
# ---------------------------

def fuse_candidates_rrf(cands_by_branch: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Выполняет Reciprocal Rank Fusion над четырьмя ветками результатов.

    Используется классическая формула RRF по рангам: score = weight / (RRF_K + rank).
    Дубликаты по chunk_id объединяются: суммируется score, сохраняется минимальный ранг,
    список веток (hits) и один экземпляр метаданных (source, page, text_ref и др.).
    Итог сортируется по fusion_score (desc), затем по min_rank (asc), затем по приоритету веток.
    """
    if not cands_by_branch:
        return []

    branch_priority = ["dense_ru", "bm25_ru", "dense_en", "bm25_en"]
    def weight_for_branch(branch_key: str) -> float:
        return RRF_WEIGHT_DENSE if branch_key in ("dense_ru", "dense_en") else RRF_WEIGHT_BM25

    aggregated: Dict[int, Dict[str, Any]] = {}

    for branch in branch_priority:
        items = cands_by_branch.get(branch, []) or []
        w = weight_for_branch(branch)
        for item in items:
            chunk_id = item.get("chunk_id")
            rank = item.get("rank")
            if not isinstance(chunk_id, (int, np.integer)):
                continue
            if not isinstance(rank, (int, np.integer)) or int(rank) <= 0:
                continue
            rank_int = int(rank)
            score = float(w) / float(RRF_K + rank_int)
            if chunk_id not in aggregated:
                aggregated[chunk_id] = {
                    "chunk_id": int(chunk_id),
                    "source": item.get("source"),
                    "chunk_index": item.get("chunk_index"),
                    "page": item.get("page"),
                    "text_ref": item.get("text_ref"),
                    "citation": item.get("citation") or format_citation({
                        "source": item.get("source"),
                        "chunk_index": item.get("chunk_index"),
                        "page": item.get("page")
                    }),
                    "fusion_score": score,
                    "min_rank": rank_int,
                    "hits": [branch],
                }
            else:
                agg = aggregated[chunk_id]
                agg["fusion_score"] = float(agg.get("fusion_score", 0.0)) + score
                agg["min_rank"] = min(int(agg.get("min_rank", rank_int)), rank_int)
                if branch not in agg.get("hits", []):
                    agg["hits"].append(branch)

    def branch_priority_index(hits: List[str]) -> int:
        for idx, b in enumerate(branch_priority):
            if b in hits:
                return idx
        return len(branch_priority)

    fused = list(aggregated.values())
    fused.sort(key=lambda x: (
        -float(x.get("fusion_score", 0.0)),
        int(x.get("min_rank", 10**9)),
        branch_priority_index(x.get("hits", [])),
    ))
    return fused[:TOP_K_RERANK_INPUT]

# ---------------------------
# Препроцессор текста для реранка
# ---------------------------

def truncate_candidates_for_rerank(candidates: List[Dict[str, Any]], max_tokens: int = RERANK_MAX_TOKENS) -> List[Dict[str, Any]]:
    """Усекает candidate['text_ref'] до безопасного окна по количеству токенов.

    Токены считаем по пробельному разделению. Возвращает новый список кандидатов
    (копии словарей при усечении), не изменяя исходные объекты.
    """
    if not candidates:
        return []
    truncated = 0
    processed: List[Dict[str, Any]] = []
    for cand in candidates:
        text = cand.get("text_ref")
        if isinstance(text, str):
            tokens = text.split()
            if len(tokens) > max_tokens:
                new_cand = cand.copy()
                new_cand["text_ref"] = " ".join(tokens[:max_tokens])
                processed.append(new_cand)
                truncated += 1
            else:
                processed.append(cand)
        else:
            processed.append(cand)
    if os.getenv("RAG_DEBUG") == "1":
        try:
            import logging
            logging.getLogger(__name__).debug("[RERANK_TRUNCATE] truncated=%d of %d", truncated, len(candidates))
        except Exception:
            pass
    return processed

def initialize_models():
    """Инициализирует и кэширует модели в глобальных переменных.

    GPU-правила:
    - Эмбеддер создаётся один раз; инференс разбивается по EMBED_GPU_IDS через multiprocessing.
    - Реранкер закрепляется за GPU RERANK_GPU_ID.
    """
    global embedder, reranker
    if embedder is None:
        embedder = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=f"cuda:{EMBED_GPU_IDS[0]}",
        )
    if reranker is None:
        # Назначаем конкретный GPU для реранкера
        try:
            import torch  # type: ignore
            has_cuda = bool(getattr(torch.cuda, "is_available", lambda: False)())
            device = f"cuda:{RERANK_GPU_ID}" if has_cuda else "cuda:0"
            use_fp16 = has_cuda
        except Exception:
            device = "cuda:0"
            use_fp16 = False
        reranker = FlagReranker(
            RERANKER_MODEL_NAME,
            use_fp16=use_fp16,
            max_length=RERANK_MAX_LENGTH,
            device=device,
        )

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def format_citation(meta: Dict[str, Any]) -> str:
    """Форматирует ссылку на источник из метаданных чанка.

    Приоритеты:
    - Если заданы page, char_start, char_end: "[source p.page char_start–char_end]"
    - Если задан page: "[source p.page]"
    - Иначе fallback: "[source:chunk_index]"
    """
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
# PDF Loading & Text Extraction
# ---------------------------

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Извлекает текст PDF постранично.

    Возвращает список словарей вида: [{"page": 1, "text": "..."}, ..., {"page": N, "text": "..."}].
    Текст берётся из Docling, нумерация страниц — по порядку элементов (fallback для несоответствий метаданных).
    """
    try:
        loader = DoclingLoader(file_path=[pdf_path])
        docs = loader.load()
        if not docs:
            return []
        page_texts = [d.page_content for d in docs if d.page_content and d.page_content.strip()]
        pages = []
        for i, t in enumerate(page_texts, start=1):
            # Не изменяем содержимое страницы (без strip), только фильтрация пустых
            pages.append({"page": i, "text": t})
        return pages
    except Exception:
        return []


def load_pdfs_from_directory(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Загружает все PDF из директории и извлекает текст постранично.
    Возвращает словарь: {filename: [{"page": i, "text": "..."}, ...]}.
    """
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    docs: Dict[str, List[Dict[str, Any]]] = {}
    for p in pdf_files:
        filename = os.path.basename(p)
        pages = extract_text_from_pdf(p)
        if isinstance(pages, list) and len(pages) > 0:
            docs[filename] = pages
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
    # Не модифицируем содержимое чанков; только фильтрация по длине
    chunks = [c for c in chunks if isinstance(c, str) and len(c.strip()) > 20]
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
        # Обновляем параметр поиска HNSW после загрузки индекса
        try:
            if hasattr(faiss_index, "hnsw"):
                faiss_index.hnsw.efSearch = 128
        except Exception:
            pass
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
    for filename, pages in docs.items():
        if not isinstance(pages, list):
            continue
        for page_entry in pages:
            page_num = page_entry.get("page")
            page_text = page_entry.get("text", "")
            if not isinstance(page_text, str) or not page_text.strip():
                continue
            raw_chunks = chunk_text(page_text)
            # Инкрементальный курсор по странице для корректной привязки смещений
            cursor = 0
            page_len = len(page_text)
            for c_text in raw_chunks:
                if not isinstance(c_text, str):
                    continue
                if len(c_text) == 0:
                    continue
                # Реальные координаты чанка: ищем подстроку, начиная с cursor
                start_idx = page_text.find(c_text, cursor) if page_len > 0 else -1
                if start_idx < 0:
                    # Фолбэк: попробуем найти с начала страницы
                    start_idx = page_text.find(c_text) if page_len > 0 else -1
                if start_idx < 0:
                    # Если не нашли, используем безопасный минимум
                    start_idx = max(0, min(cursor, max(0, page_len - 1)))
                end_idx = start_idx + len(c_text) - 1 if page_len > 0 else 0
                chunks_metadata.append({
                    "source": filename,
                    "chunk_index": doc_counter,
                    "text": c_text,
                    "page": int(page_num) if isinstance(page_num, int) and page_num > 0 else 1,
                    "char_start": int(start_idx),
                    "char_end": int(end_idx),
                })
                all_chunks_text.append(c_text)
                # Шаг курсора: длина чанка минус overlap; гарантируем прогресс минимум на 1
                step = max(1, len(c_text) - CHUNK_OVERLAP)
                cursor = start_idx + step
                doc_counter += 1
            
    # Создание Dense (векторного) индекса с распределением по нескольким GPU
    embeddings = _encode_multi_gpu(all_chunks_text, batch_size=EMBED_BATCH_SIZE, gpu_ids=EMBED_GPU_IDS)
    
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
def hybrid_search_with_rerank(question: str) -> Dict[str, List[Dict[str, Any]]]:
    """Гибридный поиск с RRF-слиянием и реранком.

    Возвращает структуру вида {"fused": fused_top[:20], "reranked": reranked_top}.
    """
    if not question or not question.strip():
        return {"fused": [], "reranked": []}

    # Сбор кандидатов по четырем веткам
    cands_by_branch = collect_candidates_ru_en(question)

    # RRF-слияние
    fused_top = fuse_candidates_rrf(cands_by_branch)
    if not fused_top:
        return {"fused": [], "reranked": []}

    # Подготовка текста для реранка
    fused_for_rerank = truncate_candidates_for_rerank(fused_top)

    # Пары (вопрос, текст)
    pairs = [(question, c.get("text_ref", "") or "") for c in fused_for_rerank]

    # Гарантируем инициализацию модели реранка
    global reranker
    if reranker is None:
        initialize_models()

    scores = reranker.compute_score(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        max_length=RERANK_MAX_LENGTH,
        normalize=True,
    )
    reranked = []
    for cand, scr in zip(fused_for_rerank, scores):
        item = cand.copy()
        item["rerank_score"] = float(scr)
        reranked.append(item)

    reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    reranked_top = reranked[:TOP_K_FINAL]

    return {"fused": fused_top[:20], "reranked": reranked_top}

# --- ГЛАВНАЯ ОБНОВЛЕННАЯ ФУНКЦИЯ ---
def create_rag_chain(openrouter_api_key: str, openrouter_model: str = OPENROUTER_MODEL):
    """
    Создает RAG-цепочку, интегрируя перевод, гибридный поиск и переранжирование.
    """
    
    # Прокидываем рантайм-параметры для simple_query_translation
    global runtime_openrouter_api_key, runtime_openrouter_model
    runtime_openrouter_api_key = openrouter_api_key
    runtime_openrouter_model = openrouter_model
    
    def answer_question(question: str) -> tuple[str, str]:
        # Считаем кандидатов по веткам для статистики
        branch_cands = collect_candidates_ru_en(question)
        # Выполняем гибридный поиск с RRF и реранком
        hybrid_out = hybrid_search_with_rerank(question)
        stats = {
            "branch_counts": {k: len(v) for k, v in branch_cands.items()},
            "fused_size": len(hybrid_out.get("fused", [])),
            "reranked_size": len(hybrid_out.get("reranked", [])),
        }
        stats_str = json.dumps(stats, ensure_ascii=False)
        # Пустой ответ LLM на этом шаге; генерация будет добавлена позже
        return "", stats_str

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
