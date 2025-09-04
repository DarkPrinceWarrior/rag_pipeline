import os
import glob
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import bm25s
import unicodedata

from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from lingua import Language

import rag_core as rc
from rag_models import initialize_models, start_embed_workers, submit_embed, drain_embed, stop_embed_workers
from rag_pipeline import tokenize_text_by_lang  # DRY
from rag_core import debug_log, ensure_dir


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Извлекает текст PDF постранично с помощью Docling."""
    try:
        loader = DoclingLoader(file_path=[pdf_path])
        docs = loader.load()
        if not docs:
            return []
        page_texts = [d.page_content for d in docs if d.page_content and d.page_content.strip()]
        pages = []
        for i, t in enumerate(page_texts, start=1):
            pages.append({"page": i, "text": t})
        return pages
    except Exception:
        return []


def load_pdfs_from_directory(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """Загружает все PDF из директории и извлекает текст постранично."""
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    docs: Dict[str, List[Dict[str, Any]]] = {}
    for p in pdf_files:
        filename = os.path.basename(p)
        pages = extract_text_from_pdf(p)
        if isinstance(pages, list) and len(pages) > 0:
            docs[filename] = pages
    return docs


def chunk_text(text: str, chunk_size: int = rc.CHUNK_SIZE, overlap: int = rc.CHUNK_OVERLAP) -> List[str]:
    """Нарезка текста на чанки с помощью LangChain RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(overlap),
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if isinstance(c, str) and len(c.strip()) > 20]


def build_and_load_knowledge_base(pdf_dir: str, index_dir: str, force_rebuild: bool = False) -> bool:
    """Создаёт или загружает базу знаний: FAISS(HNSW) + BM25, матрицу эмбеддингов и метаданные."""
    initialize_models()

    ensure_dir(index_dir)
    faiss_path = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    bm25_dir = os.path.join(index_dir, "bm25")
    bm25_ids_path = os.path.join(index_dir, "bm25_ids.pkl")
    emb_path = os.path.join(index_dir, "embeddings.npy")

    if force_rebuild:
        for path in [faiss_path, metadata_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(bm25_dir):
            for root, _, files in os.walk(bm25_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            os.rmdir(bm25_dir)
        if os.path.exists(bm25_ids_path):
            os.remove(bm25_ids_path)
        if os.path.exists(emb_path):
            os.remove(emb_path)

    bm25_loaded = False
    if os.path.exists(faiss_path) and os.path.exists(metadata_path) and os.path.isdir(bm25_dir):
        rc.faiss_index = faiss.read_index(faiss_path)
        try:
            if hasattr(rc.faiss_index, "hnsw"):
                rc.faiss_index.hnsw.efSearch = int(rc.HNSW_EF_SEARCH_BASE)
        except Exception:
            pass
        try:
            if rc.FAISS_CPU_THREADS is not None:
                faiss.omp_set_num_threads(int(rc.FAISS_CPU_THREADS))
        except Exception:
            pass
        with open(metadata_path, "rb") as f:
            rc.chunks_metadata = pickle.load(f)
        rc.EMB_MATRIX = None
        try:
            if os.path.exists(emb_path):
                rc.EMB_MATRIX = np.load(emb_path, mmap_mode='r')
        except Exception:
            rc.EMB_MATRIX = None
        try:
            rc.bm25_retriever = bm25s.BM25.load(bm25_dir, load_corpus=False, mmap=False)
            if os.path.exists(bm25_ids_path):
                with open(bm25_ids_path, "rb") as f:
                    rc.bm25_corpus_ids = pickle.load(f)
            else:
                rc.bm25_corpus_ids = []
            bm25_loaded = bool(rc.bm25_corpus_ids)
        except Exception:
            bm25_loaded = False
        try:
            _ntotal = int(getattr(rc.faiss_index, 'ntotal', 0)) if rc.faiss_index is not None else 0
            _meta = len(rc.chunks_metadata)
            _bm25 = len(rc.bm25_corpus_ids)
            _ef = int(getattr(getattr(rc.faiss_index, 'hnsw', object()), 'efSearch', 0)) if rc.faiss_index is not None else 0
            debug_log("[INDEX_LOAD] ntotal=%d, metadata=%d, bm25_ids=%d, efSearch=%d, threads=%s",
                      _ntotal, _meta, _bm25, _ef, (str(rc.FAISS_CPU_THREADS) if rc.FAISS_CPU_THREADS is not None else "default"))
        except Exception:
            pass
        if bm25_loaded:
            return True

    docs = load_pdfs_from_directory(pdf_dir)
    if not docs:
        raise RuntimeError(f"PDF-файлы не найдены в {pdf_dir}.")

    all_chunks_text: List[str] = []
    rc.chunks_metadata = []
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
            cursor = 0
            page_len = len(page_text)
            page_lang = None
            if rc.language_detector is not None:
                try:
                    page_lang_detected = rc.language_detector.detect_language_of(page_text)
                    if page_lang_detected == Language.RUSSIAN:
                        page_lang = 'ru'
                    elif page_lang_detected == Language.ENGLISH:
                        page_lang = 'en'
                except Exception:
                    page_lang = None
            for c_text in raw_chunks:
                if not isinstance(c_text, str) or len(c_text) == 0:
                    continue
                start_idx = page_text.find(c_text, cursor) if page_len > 0 else -1
                if start_idx < 0:
                    start_idx = page_text.find(c_text) if page_len > 0 else -1
                if start_idx < 0:
                    start_idx = max(0, min(cursor, max(0, page_len - 1)))
                end_idx = start_idx + len(c_text) - 1 if page_len > 0 else 0
                lang_code = None
                try:
                    text_for_lang = c_text if len(c_text) >= rc.LANG_DETECT_MIN_CHARS else page_text
                    if rc.language_detector is not None:
                        lang = rc.language_detector.detect_language_of(text_for_lang)
                        if lang == Language.RUSSIAN:
                            lang_code = 'ru'
                        elif lang == Language.ENGLISH:
                            lang_code = 'en'
                except Exception:
                    lang_code = None
                if lang_code is None:
                    lang_code = page_lang or 'unk'
                rc.chunks_metadata.append({
                    "source": filename,
                    "chunk_index": doc_counter,
                    "text": c_text,
                    "page": int(page_num) if isinstance(page_num, int) and page_num > 0 else 1,
                    "char_start": int(start_idx),
                    "char_end": int(end_idx),
                    "lang": lang_code,
                })
                all_chunks_text.append(c_text)
                step = max(1, len(c_text) - rc.CHUNK_OVERLAP)
                cursor = start_idx + step
                doc_counter += 1

    # Эмбеддинги через слой стойких воркеров
    start_embed_workers(rc.EMBED_GPU_IDS, rc.EMBEDDING_MODEL_NAME, rc.EMBED_MAX_LENGTH, rc.EMBED_BATCH_SIZE)
    try:
        # Отправка батчей заданий в воркеры
        global_idx = 0
        for batch in _chunk_iter(all_chunks_text, rc.EMBED_BATCH_SIZE):
            submit_embed((global_idx, batch))
            global_idx += len(batch)
        # Сбор результатов и восстановление порядка по глобальному индексу
        results = drain_embed()
        results.sort(key=lambda x: int(x[0]))
        parts = [emb for _, emb in results if isinstance(emb, np.ndarray) and emb.size > 0]
        if parts:
            embeddings = np.concatenate(parts, axis=0).astype('float32')
        else:
            embeddings = np.zeros((0, 0), dtype='float32')
    finally:
        stop_embed_workers()
    try:
        faiss.normalize_L2(embeddings)
    except Exception:
        pass

    dim = embeddings.shape[1]
    rc.faiss_index = faiss.IndexHNSWFlat(dim, int(rc.HNSW_M), faiss.METRIC_INNER_PRODUCT)
    try:
        if hasattr(rc.faiss_index, "hnsw"):
            rc.faiss_index.hnsw.efConstruction = int(rc.HNSW_EF_CONSTRUCTION)
            rc.faiss_index.hnsw.efSearch = int(rc.HNSW_EF_SEARCH_BASE)
    except Exception:
        pass
    try:
        if rc.FAISS_CPU_THREADS is not None:
            faiss.omp_set_num_threads(int(rc.FAISS_CPU_THREADS))
    except Exception:
        pass

    _t0 = time.perf_counter()
    rc.faiss_index.add(embeddings)
    _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
    try:
        debug_log("[INDEX_BUILD] dim=%d, M=%d, efConstruction=%d, add_time_ms=%.1f, ntotal=%d, threads=%s",
                  int(dim), int(rc.HNSW_M), int(rc.HNSW_EF_CONSTRUCTION), float(_elapsed_ms),
                  int(getattr(rc.faiss_index, 'ntotal', 0)), (str(rc.FAISS_CPU_THREADS) if rc.FAISS_CPU_THREADS is not None else "default"))
    except Exception:
        pass
    faiss.write_index(rc.faiss_index, faiss_path)

    # BM25: токенизация с учётом языка
    texts_for_bm25: List[tuple[str, str | None]] = []
    rc.bm25_corpus_ids = []
    for idx, meta in enumerate(rc.chunks_metadata):
        t = meta.get("text")
        if isinstance(t, str) and t.strip():
            texts_for_bm25.append((t, meta.get("lang")))
            rc.bm25_corpus_ids.append(idx)
    if not texts_for_bm25:
        raise RuntimeError("BM25 корпус пуст после фильтрации.")

    corpus_tokens: List[List[str]] = []
    for text, lang in texts_for_bm25:
        corpus_tokens.append(tokenize_text_by_lang(text, lang if isinstance(lang, str) else None))

    rc.bm25_retriever = bm25s.BM25(method="lucene")
    rc.bm25_retriever.index(corpus_tokens)
    ensure_dir(bm25_dir)
    rc.bm25_retriever.save(bm25_dir)
    with open(bm25_ids_path, "wb") as f:
        pickle.dump(rc.bm25_corpus_ids, f)

    # Сохранение матрицы эмбеддингов
    try:
        np.save(emb_path, embeddings.astype('float32'), allow_pickle=False)
    except Exception:
        pass

    with open(metadata_path, "wb") as f:
        pickle.dump(rc.chunks_metadata, f)

    return True
