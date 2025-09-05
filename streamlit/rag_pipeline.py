import os
import json
import re
import unicodedata
from typing import Any, Dict, List

# --- Ранняя настройка JAX/XLA перед импортом bm25s (во время поиска) ---
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_xla_frac = os.getenv("RAG_XLA_MEM_FRACTION")
if _xla_frac:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = _xla_frac

import bm25s
import faiss
import numpy as np
import requests

import rag_core as rc
from rag_core import debug_log, format_citation
from rag_models import initialize_models, rerank_multi_gpu
from rag_config import (
    RETRIEVAL_DENSE_RU,
    RETRIEVAL_BM25_RU,
    RETRIEVAL_DENSE_EN,
    RETRIEVAL_BM25_EN,
    RETRIEVAL_DENSE_ORIG,
    RETRIEVAL_BM25_ORIG,
    RRF_BRANCH_PRIORITY,
)


# --- Языковые утилиты ---

def _detect_question_lang(question: str) -> str:
    """Определяет язык вопроса: 'en', 'ru', 'uk', 'be' или 'unk'."""
    if not question or not question.strip() or rc.language_detector is None:
        return 'unk'
    try:
        lang = rc.language_detector.detect_language_of(question)
        from lingua import Language  # локальный импорт, чтобы не тянуть при старте
        if lang == Language.ENGLISH:
            return 'en'
        if lang == Language.RUSSIAN:
            return 'ru'
        if lang == Language.UKRAINIAN:
            return 'uk'
        if lang == Language.BELARUSIAN:
            return 'be'
        return 'unk'
    except Exception:
        return 'unk'


def tokenize_text_by_lang(text: str, lang: str | None) -> List[str]:
    """Токенизация текста (RU/EN) для BM25: unicode-разделение, нижний регистр, стемминг, стоп-слова."""
    s = (text or "").lower()
    tokens: List[str] = []
    buf: List[str] = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat and (cat[0] == 'L' or cat[0] == 'N'):
            buf.append(ch)
        else:
            if buf:
                tokens.append(''.join(buf))
                buf = []
    if buf:
        tokens.append(''.join(buf))
    if lang == 'ru':
        if rc.ru_stemmer is not None:
            tokens = rc.ru_stemmer.stemWords(tokens)
        if rc.ru_stopwords:
            tokens = [t for t in tokens if t not in rc.ru_stopwords]
        return tokens
    if lang == 'en':
        if rc.en_stemmer is not None:
            tokens = rc.en_stemmer.stemWords(tokens)
        if rc.en_stopwords:
            tokens = [t for t in tokens if t not in rc.en_stopwords]
        return tokens
    try:
        return bm25s.tokenize([text])[0]
    except Exception:
        return tokens


def simple_query_translation(question: str, q_lang: str | None = None) -> Dict[str, str]:
    """Правило перевода запроса в EN в зависимости от языка."""
    original = question or ""
    if not original.strip():
        return {"original": "", "english": ""}
    if q_lang == 'en':
        return {"original": original, "english": original}
    need_translation = q_lang in {'ru', 'uk', 'be'} if q_lang else True
    if not rc.runtime_openrouter_api_key or not rc.runtime_openrouter_model:
        return {"original": original, "english": original if not need_translation else original}
    messages = [
        {"role": "system", "content": "Translate this query to English. Do not add information."},
        {"role": "user", "content": original},
    ]
    try:
        if need_translation:
            response_json = call_openrouter_chat_completion(
                api_key=rc.runtime_openrouter_api_key,
                model=rc.runtime_openrouter_model,
                messages=messages,
                extra_request_kwargs={"temperature": 0.2},
            )
            text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not text:
                text = response_json.get("choices", [{}])[0].get("text", "").strip()
            english = text or original
        else:
            english = original
        return {"original": original, "english": english}
    except Exception:
        return {"original": original, "english": original}


# --- Кандидаты и ретривал ---


def build_candidate_dict(chunk_global_id: int, retrieval_label: str, rank: int, score_raw: float) -> Dict[str, Any]:
    """Создаёт единый словарь для кандидата документа."""
    if not isinstance(chunk_global_id, (int, np.integer)) or chunk_global_id < 0 or chunk_global_id >= len(rc.chunks_metadata):
        raise ValueError("Некорректный chunk_global_id для кандидата.")
    meta = rc.chunks_metadata[chunk_global_id]
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
    """Создаёт список кандидатов по массивам индексов и скоров."""
    candidates: List[Dict[str, Any]] = []
    current_rank = 1
    for idx, s in zip(indices, scores):
        if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < len(rc.chunks_metadata):
            candidates.append(build_candidate_dict(int(idx), retrieval_label, current_rank, float(s)))
            current_rank += 1
    return candidates


def _compute_intersections_report(cands: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Подсчёт пересечений по chunk_id между ветками."""
    branches = [RETRIEVAL_DENSE_RU, RETRIEVAL_BM25_RU, RETRIEVAL_DENSE_EN, RETRIEVAL_BM25_EN]
    name_to_list = {
        RETRIEVAL_DENSE_RU: cands.get(RETRIEVAL_DENSE_RU, []),
        RETRIEVAL_BM25_RU: cands.get(RETRIEVAL_BM25_RU, []),
        RETRIEVAL_DENSE_EN: cands.get(RETRIEVAL_DENSE_EN, []),
        RETRIEVAL_BM25_EN: cands.get(RETRIEVAL_BM25_EN, []),
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
    """Условный debug-отчёт о пересечениях веток при RAG_DEBUG=1."""
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
        pass


def collect_candidates_ru_en(question: str, k_dense: int = rc.TOP_K_DENSE_BRANCH, k_bm25: int = rc.TOP_K_BM25_BRANCH, apply_lang_quota: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """Сбор кандидатов по 4 веткам (dense/bm25 × RU/EN) с учётом языка вопроса."""
    if not question or not question.strip():
        return {"q_lang": _detect_question_lang(question), "dense_ru": [], "bm25_ru": [], "dense_en": [], "bm25_en": []}

    q_lang = _detect_question_lang(question)
    k_dense_eff = min(k_dense, rc.faiss_index.ntotal if rc.faiss_index is not None else 0)

    def _dense(query_text: str, label: str) -> List[Dict[str, Any]]:
        if k_dense_eff <= 0:
            return []
        try:
            if hasattr(rc.faiss_index, "hnsw"):
                need_topk = int(k_dense_eff)
                ef_val = int(rc.HNSW_EF_SEARCH_BASE)
                if rc.HNSW_USE_DYNAMIC_EF_SEARCH:
                    ef_val = max(int(rc.HNSW_EF_SEARCH_BASE), int(rc.HNSW_EF_SEARCH_PER_TOPK_MULT) * need_topk)
                ef_val = min(int(rc.HNSW_EF_SEARCH_MAX), int(ef_val))
                rc.faiss_index.hnsw.efSearch = int(ef_val)
                debug_log("[SEARCH] label=%s, k=%d, ef=%d", str(label), need_topk, int(ef_val))
        except Exception:
            pass
        if rc.embedder is None:
            from rag_models import initialize_models
            initialize_models()
        q_emb = rc.embedder.encode(
            sentences=[query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indices = rc.faiss_index.search(q_emb, k_dense_eff)
        cands = build_candidates_from_arrays(list(indices[0]), list(scores[0]), label)
        try:
            if hasattr(rc.faiss_index, "hnsw") and len(cands) < int(rc.HNSW_FAILSAFE_RATIO * k_dense_eff):
                retries = 0
                while retries < int(rc.HNSW_FAILSAFE_MAX_RETRY):
                    new_ef = min(int(rc.HNSW_EF_SEARCH_MAX), int(getattr(rc.faiss_index.hnsw, 'efSearch', rc.HNSW_EF_SEARCH_BASE)) * 2)
                    rc.faiss_index.hnsw.efSearch = int(new_ef)
                    debug_log("[SEARCH_RETRY] label=%s, k=%d, retry=%d, ef=%d", str(label), int(k_dense_eff), retries + 1, int(new_ef))
                    scores, indices = rc.faiss_index.search(q_emb, k_dense_eff)
                    cands = build_candidates_from_arrays(list(indices[0]), list(scores[0]), label)
                    retries += 1
                    if len(cands) >= int(rc.HNSW_FAILSAFE_RATIO * k_dense_eff):
                        break
        except Exception:
            pass
        return cands

    def _bm25(query_text: str, label: str, lang_for_query: str | None) -> List[Dict[str, Any]]:
        if rc.bm25_retriever is None or not rc.bm25_corpus_ids:
            return []
        q_tokens = tokenize_text_by_lang(query_text, lang_for_query)
        tokens = [q_tokens]
        res, scr = rc.bm25_retriever.retrieve(tokens, k=k_bm25, corpus=rc.bm25_corpus_ids)
        if len(res) > 0:
            return build_candidates_from_arrays(list(res[0]), list(scr[0]), label)
        return []

    def _apply_lang_quota(items: List[Dict[str, Any]], target_lang: str | None) -> List[Dict[str, Any]]:
        if not items or not target_lang or not apply_lang_quota:
            return items
        same = [c for c in items if (rc.chunks_metadata[c.get("chunk_id")].get("lang") == target_lang)]
        other = [c for c in items if (rc.chunks_metadata[c.get("chunk_id")].get("lang") != target_lang)]
        if not same:
            return items
        n_total = len(items)
        n_same = max(1, int(n_total * rc.SAME_LANG_RATIO))
        n_other = max(0, n_total - n_same)
        return same[:n_same] + other[:n_other]

    dense_ru: List[Dict[str, Any]] = []
    bm25_ru: List[Dict[str, Any]] = []
    dense_en: List[Dict[str, Any]] = []
    bm25_en: List[Dict[str, Any]] = []

    if q_lang == 'en':
        tr = simple_query_translation(question, q_lang='en')
        en_q = tr.get('english') or question
        dense_en = _apply_lang_quota(_dense(en_q, RETRIEVAL_DENSE_EN), 'en')
        bm25_en = _apply_lang_quota(_bm25(en_q, RETRIEVAL_BM25_EN, 'en'), 'en')
    elif q_lang in {'ru', 'uk', 'be'}:
        tr = simple_query_translation(question, q_lang=q_lang)
        orig_q = tr.get('original') or question
        en_q = tr.get('english') or question
        dense_ru = _apply_lang_quota(_dense(orig_q, RETRIEVAL_DENSE_RU), 'ru')
        bm25_ru = _apply_lang_quota(_bm25(orig_q, RETRIEVAL_BM25_RU, q_lang), 'ru')
        dense_en = _apply_lang_quota(_dense(en_q, RETRIEVAL_DENSE_EN), 'en')
        bm25_en = _apply_lang_quota(_bm25(en_q, RETRIEVAL_BM25_EN, 'en'), 'en')
    else:
        tr = simple_query_translation(question, q_lang='unk')
        orig_q = tr.get('original') or question
        en_q = tr.get('english') or question
        dense_ru = _apply_lang_quota(_dense(orig_q, RETRIEVAL_DENSE_ORIG), q_lang)
        bm25_ru = _apply_lang_quota(_bm25(orig_q, RETRIEVAL_BM25_ORIG, q_lang), q_lang)
        dense_en = _apply_lang_quota(_dense(en_q, RETRIEVAL_DENSE_EN), 'en')
        bm25_en = _apply_lang_quota(_bm25(en_q, RETRIEVAL_BM25_EN, 'en'), 'en')

    out = {
        "q_lang": q_lang,
        "dense_ru": dense_ru,
        "bm25_ru": bm25_ru,
        "dense_en": dense_en,
        "bm25_en": bm25_en,
    }
    log_intersections_debug(out)
    return out


def fuse_candidates_rrf(cands_by_branch: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion (по рангам) с агрегацией дубликатов."""
    if not cands_by_branch:
        return []

    branch_priority = RRF_BRANCH_PRIORITY

    def weight_for_branch(branch_key: str) -> float:
        return rc.RRF_WEIGHT_DENSE if branch_key in (RETRIEVAL_DENSE_RU, RETRIEVAL_DENSE_EN) else rc.RRF_WEIGHT_BM25

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
            score = float(w) / float(rc.RRF_K + rank_int)
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
    return fused[:rc.TOP_K_RERANK_INPUT]


def truncate_candidates_for_rerank(candidates: List[Dict[str, Any]], max_tokens: int = rc.RERANK_MAX_TOKENS) -> List[Dict[str, Any]]:
    """Усечение текста кандидатов до безопасного окна по токенам (приближённо по словам)."""
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


def call_openrouter_chat_completion(api_key, model, messages, endpoint=rc.OPENROUTER_ENDPOINT, extra_request_kwargs=None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    if extra_request_kwargs:
        payload.update(extra_request_kwargs)
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=rc.OPENROUTER_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"OpenRouter API request failed: {e}. Response: {resp.text if 'resp' in locals() else 'No response'}")


def hybrid_search_with_rerank(question: str, apply_lang_quota: bool = True) -> Dict[str, Any]:
    """Гибридный поиск (4 ветки) → RRF → реранк → отбор контекста (MMR/квоты/бюджет)."""
    if not question or not question.strip():
        _ql = _detect_question_lang(question)
        _al = _ql if _ql in {"ru", "en"} else "ru"
        return {
            "q_lang": _ql,
            "answer_lang": _al,
            "active_branches": [],
            "fused": [],
            "reranked": [],
            "context_pack": [],
            "context_stats": {},
            "sources_map": {},
        }

    cands_by_branch = collect_candidates_ru_en(question, apply_lang_quota=apply_lang_quota)
    fused_top = fuse_candidates_rrf(cands_by_branch)
    if not fused_top:
        _ql = cands_by_branch.get("q_lang")
        _al = _ql if _ql in {"ru", "en"} else "ru"
        return {
            "q_lang": _ql,
            "answer_lang": _al,
            "active_branches": [k for k in RRF_BRANCH_PRIORITY if cands_by_branch.get(k)],
            "fused": [],
            "reranked": [],
            "context_pack": [],
            "context_stats": {},
            "sources_map": {},
        }

    fused_for_rerank = truncate_candidates_for_rerank(fused_top)
    texts = [c.get("text_ref", "") or "" for c in fused_for_rerank]

    if not rc.reranker_pools and rc.reranker is None:
        initialize_models()

    if rc.reranker_pools:
        scores = rerank_multi_gpu(question, texts, rc.reranker_pools)
    else:
        pairs = [(question, t) for t in texts]
        scores = rc.reranker.compute_score(
            pairs,
            batch_size=rc.RERANK_BATCH_SIZE,
            max_length=rc.RERANK_MAX_LENGTH,
            normalize=True,
        )
    reranked = []
    for cand, scr in zip(fused_for_rerank, scores):
        item = cand.copy()
        item["rerank_score"] = float(scr)
        reranked.append(item)

    reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    reranked_top = reranked[:rc.TOP_K_FINAL]

    context_selection = select_context_for_generation(question, reranked_top, apply_lang_quota=apply_lang_quota)
    context_pack = context_selection.get("selected", [])
    rejected_list = context_selection.get("rejected", [])
    debug_info = context_selection.get("debug", {})

    lang_distribution = dict(debug_info.get("lang_count", {}))
    doc_distribution = dict(debug_info.get("per_doc", {}))
    page_distribution = dict(debug_info.get("per_page", {}))

    rejected_reasons: Dict[str, int] = {}
    for r in rejected_list:
        reason = r.get("reason")
        if isinstance(reason, str) and reason:
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    relax_used = debug_info.get("relax", {}) if isinstance(debug_info.get("relax"), dict) else {}
    context_stats = {
        "selected_count": int(debug_info.get("selected", len(context_pack))),
        "pool_initial": int(debug_info.get("pool_initial", 0)),
        "budget_used_tokens": int(debug_info.get("used_tokens", 0)),
        "budget_limit": int(rc.CONTEXT_TOKENS_BUDGET),
        "lang_distribution": lang_distribution,
        "doc_distribution": doc_distribution,
        "page_distribution": page_distribution,
        "rejected_reasons": rejected_reasons,
        "thresholds": {
            "dup_emb_cos_threshold": float(relax_used.get("dup_cos_threshold", rc.DUP_EMB_COS_THRESHOLD)),
            "dup_char_iou_threshold": float(rc.DUP_CHAR_IOU_THRESHOLD),
            "per_doc_cap": int(rc.PER_DOC_CAP),
            "per_page_cap": int(rc.PER_PAGE_CAP),
            "ignore_per_page": bool(relax_used.get("ignore_per_page", False)),
            "ignore_lang_cover": bool(relax_used.get("ignore_lang_cover", False)),
            "mmr_lambda": float(rc.MMR_LAMBDA),
            "context_top_k": int(rc.CONTEXT_TOP_K),
        },
    }

    sources_map: Dict[str, int] = {}
    try:
        for idx, c in enumerate(context_pack):
            cid = c.get("chunk_id")
            if isinstance(cid, (int, np.integer)):
                sources_map[f"S{idx+1}"] = int(cid)
    except Exception:
        sources_map = {}

    _ql = cands_by_branch.get("q_lang")
    answer_lang = _ql if _ql in {"ru", "en"} else "ru"

    return {
        "q_lang": _ql,
        "answer_lang": answer_lang,
        "active_branches": [k for k in RRF_BRANCH_PRIORITY if cands_by_branch.get(k)],
        "fused": fused_top[:20],
        "reranked": reranked_top,
        "context_pack": context_pack,
        "context_stats": context_stats,
        "sources_map": sources_map,
    }


# --- Пост-обработка, MMR/дедуп/квоты/бюджет ---

def _get_chunk_vec(idx: int) -> np.ndarray:
    """Возвращает L2-нормализованный вектор чанка из rc.EMB_MATRIX."""
    if rc.EMB_MATRIX is None:
        raise RuntimeError("Матрица эмбеддингов не загружена (EMB_MATRIX is None).")
    if not isinstance(idx, (int, np.integer)) or int(idx) < 0 or int(idx) >= rc.EMB_MATRIX.shape[0]:
        raise ValueError("Некорректный индекс чанка для EMB_MATRIX.")
    row = rc.EMB_MATRIX[int(idx)]
    vec = np.asarray(row, dtype="float32")
    try:
        norm2 = float(np.dot(vec, vec))
        if not (0.999 <= norm2 <= 1.001):
            v = vec.reshape(1, -1).copy()
            faiss.normalize_L2(v)
            return v[0]
        return vec
    except Exception:
        return vec


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype="float32")
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return v
    return v / n


def _estimate_tokens(text: str | None, lang: str | None) -> int:
    """Приблизительная оценка числа токенов по длине текста и языку."""
    if not isinstance(text, str) or not text:
        return 0
    length = len(text)
    if lang == 'ru':
        return int(length / 3.5)
    if lang == 'en':
        return int(length / 4.0)
    return int(length / 4.0)


def _compute_char_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    if a0 > a1 or b0 > b1:
        return 0.0
    inter_left = max(a0, b0)
    inter_right = min(a1, b1)
    inter_len = max(0, inter_right - inter_left + 1)
    union_left = min(a0, b0)
    union_right = max(a1, b1)
    union_len = max(1, union_right - union_left + 1)
    return float(inter_len) / float(union_len)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.astype('float32'), b.astype('float32')))


def _candidate_lang(cand: Dict[str, Any]) -> str | None:
    try:
        cid = cand.get("chunk_id")
        if isinstance(cid, (int, np.integer)) and 0 <= int(cid) < len(rc.chunks_metadata):
            return rc.chunks_metadata[int(cid)].get("lang")
    except Exception:
        return None
    return None


def select_context_for_generation(question: str, reranked_top: List[Dict[str, Any]], apply_lang_quota: bool = True) -> Dict[str, Any]:
    """Отбор финального контекста для генерации (MMR + дедуп + квоты и бюджет)."""
    pool = list(reranked_top[: min(rc.MMR_POOL_K, len(reranked_top))])
    selected: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    if not pool:
        return {"selected": [], "rejected": [], "debug": {"reason": "empty_pool"}}

    if rc.embedder is None:
        from rag_models import initialize_models
        initialize_models()
    q_vec = rc.embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype('float32')[0]
    q_vec = _l2_normalize(q_vec)

    per_doc_count: Dict[str, int] = {}
    per_page_count: Dict[tuple, int] = {}
    lang_count: Dict[str, int] = {k: 0 for k in rc.LANG_MIN_COVER.keys()}
    used_tokens = 0

    effective_dup_emb_cos_threshold = float(rc.DUP_EMB_COS_THRESHOLD)
    ignore_per_page_cap = False
    ignore_lang_min_cover = (not bool(apply_lang_quota))

    def _on_accept(cand: Dict[str, Any]) -> None:
        nonlocal used_tokens
        try:
            cid = cand.get("chunk_id")
            meta = rc.chunks_metadata[int(cid)] if isinstance(cid, (int, np.integer)) and 0 <= int(cid) < len(rc.chunks_metadata) else {}
            src = str(meta.get("source")) if meta else None
            pg = int(meta.get("page")) if meta and isinstance(meta.get("page"), int) else None
            lang = meta.get("lang") if meta else None
            if src is not None:
                per_doc_count[src] = per_doc_count.get(src, 0) + 1
            if (not ignore_per_page_cap) and int(rc.PER_PAGE_CAP) > 0 and src is not None and pg is not None:
                per_page_count[(src, pg)] = per_page_count.get((src, pg), 0) + 1
            if isinstance(lang, str) and lang in lang_count:
                lang_count[lang] = lang_count.get(lang, 0) + 1
            used_tokens += _estimate_tokens(meta.get("text") if isinstance(meta, dict) else None, lang)
        except Exception:
            pass

    def passes_dedup(cand: Dict[str, Any]) -> tuple[bool, str]:
        try:
            cid = cand.get("chunk_id")
            if not isinstance(cid, (int, np.integer)):
                return (False, "bad_chunk_id")
            meta = rc.chunks_metadata[int(cid)] if 0 <= int(cid) < len(rc.chunks_metadata) else None
            if not isinstance(meta, dict):
                return (True, "ok")
            src = meta.get("source")
            pg = meta.get("page")
            c0 = int(meta.get("char_start", 0))
            c1 = int(meta.get("char_end", 0))
            for sel in selected:
                sid = sel.get("chunk_id")
                if not isinstance(sid, (int, np.integer)):
                    continue
                sm = rc.chunks_metadata[int(sid)] if 0 <= int(sid) < len(rc.chunks_metadata) else None
                if not isinstance(sm, dict):
                    continue
                if sm.get("source") == src and sm.get("page") == pg:
                    s0 = int(sm.get("char_start", 0))
                    s1 = int(sm.get("char_end", 0))
                    iou = _compute_char_iou(c0, c1, s0, s1)
                    if iou >= float(rc.DUP_CHAR_IOU_THRESHOLD):
                        return (False, f"dup_char_iou={iou:.2f}")
            try:
                cand_vec = _get_chunk_vec(cand.get("chunk_id"))
                max_cos = 0.0
                for sel in selected:
                    sel_vec = _get_chunk_vec(sel.get("chunk_id"))
                    max_cos = max(max_cos, _cosine_sim(cand_vec, sel_vec))
                    if max_cos >= float(effective_dup_emb_cos_threshold):
                        return (False, f"dup_cos={max_cos:.2f}")
            except Exception:
                pass
            return (True, "ok")
        except Exception:
            return (True, "ok")

    def passes_quotas_and_budget(cand: Dict[str, Any]) -> tuple[bool, str]:
        try:
            cid = cand.get("chunk_id")
            meta = rc.chunks_metadata[int(cid)] if isinstance(cid, (int, np.integer)) and 0 <= int(cid) < len(rc.chunks_metadata) else {}
            src = str(meta.get("source")) if meta else None
            pg = int(meta.get("page")) if meta and isinstance(meta.get("page"), int) else None
            lang = meta.get("lang") if meta else None
            if src is not None and per_doc_count.get(src, 0) >= int(rc.PER_DOC_CAP):
                return (False, "doc_cap")
            if (not ignore_per_page_cap) and int(rc.PER_PAGE_CAP) > 0 and src is not None and pg is not None and per_page_count.get((src, pg), 0) >= int(rc.PER_PAGE_CAP):
                return (False, "page_cap")
            text = meta.get("text") if isinstance(meta, dict) else None
            need_tokens = _estimate_tokens(text, lang)
            if used_tokens + need_tokens > int(rc.CONTEXT_TOKENS_BUDGET):
                return (False, "budget")
            remaining_slots = int(rc.CONTEXT_TOP_K) - len(selected)
            remaining_lang_quota: Dict[str, int] = {}
            for lk, min_need in rc.LANG_MIN_COVER.items():
                try:
                    min_need_int = int(min_need)
                except Exception:
                    min_need_int = 0
                covered = lang_count.get(lk, 0)
                if covered < min_need_int:
                    remaining_lang_quota[lk] = (min_need_int - covered)
            total_remaining_quota = sum(remaining_lang_quota.values())
            if (not ignore_lang_min_cover) and total_remaining_quota > 0 and remaining_slots <= total_remaining_quota:
                if not (isinstance(lang, str) and lang in remaining_lang_quota and remaining_lang_quota[lang] > 0):
                    return (False, "lang_quota")
            return (True, "ok")
        except Exception:
            return (True, "ok")

    pool_vecs: Dict[int, np.ndarray] = {}
    for cand in pool:
        try:
            pool_vecs[id(cand)] = _get_chunk_vec(cand.get("chunk_id"))
        except Exception:
            pass

    decline_counter = 0
    while len(selected) < int(rc.CONTEXT_TOP_K) and pool:
        scores_local = []
        for cand in pool:
            cand_vec = pool_vecs.get(id(cand))
            if cand_vec is None:
                try:
                    cand_vec = _get_chunk_vec(cand.get("chunk_id"))
                    pool_vecs[id(cand)] = cand_vec
                except Exception:
                    scores_local.append((-1e9, cand, 0.0, 0.0))
                    continue
            rel = _cosine_sim(q_vec, cand_vec)
            div = 0.0
            for sel in selected:
                sel_vec = pool_vecs.get(id(sel))
                if sel_vec is None:
                    try:
                        sel_vec = _get_chunk_vec(sel.get("chunk_id"))
                        pool_vecs[id(sel)] = sel_vec
                    except Exception:
                        continue
                div = max(div, _cosine_sim(cand_vec, sel_vec))
            score = float(rc.MMR_LAMBDA) * rel - float(1.0 - float(rc.MMR_LAMBDA)) * div
            scores_local.append((score, cand, rel, div))
        scores_local.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cand, best_rel, best_div = scores_local[0]

        ok_dedup, reason_dedup = passes_dedup(best_cand)
        ok_quota, reason_quota = passes_quotas_and_budget(best_cand)
        if ok_dedup and ok_quota:
            try:
                best_cand["mmr_score"] = float(best_score)
                best_cand["mmr_rel"] = float(best_rel)
                best_cand["mmr_div"] = float(best_div)
            except Exception:
                pass
            selected.append(best_cand)
            _on_accept(best_cand)
        else:
            rejected.append({"candidate": best_cand, "reason": (reason_dedup if not ok_dedup else reason_quota), "score": float(best_score)})
            decline_counter += 1

        pool = [c for c in pool if c is not best_cand]
        if decline_counter >= 3 and (len(selected) < int(rc.CONTEXT_TOP_K) // 2):
            try:
                effective_dup_emb_cos_threshold = float(min(float(rc.RELAX_DUP_EMB_COS_MAX), float(effective_dup_emb_cos_threshold) + float(rc.RELAX_DUP_EMB_COS_STEP)))
                decline_counter = 0
            except Exception:
                pass

    relax_info = {
        "dup_cos_threshold": float(effective_dup_emb_cos_threshold),
        "ignore_per_page": bool(False),
        "ignore_lang_cover": bool(ignore_lang_min_cover),
        "fallback_tail_used": bool(False),
        "dup_cos_steps": int(0),
    }

    def _attempt_fill() -> int:
        nonlocal pool
        added = 0
        while len(selected) < int(rc.CONTEXT_TOP_K) and pool:
            scores_local = []
            for cand in pool:
                cand_vec = pool_vecs.get(id(cand))
                if cand_vec is None:
                    try:
                        cand_vec = _get_chunk_vec(cand.get("chunk_id"))
                        pool_vecs[id(cand)] = cand_vec
                    except Exception:
                        scores_local.append((-1e9, cand, 0.0, 0.0))
                        continue
                rel = _cosine_sim(q_vec, cand_vec)
                div = 0.0
                for sel in selected:
                    sel_vec = pool_vecs.get(id(sel))
                    if sel_vec is None:
                        try:
                            sel_vec = _get_chunk_vec(sel.get("chunk_id"))
                            pool_vecs[id(sel)] = sel_vec
                        except Exception:
                            continue
                    div = max(div, _cosine_sim(cand_vec, sel_vec))
                score = float(rc.MMR_LAMBDA) * rel - float(1.0 - float(rc.MMR_LAMBDA)) * div
                scores_local.append((score, cand, rel, div))
            if not scores_local:
                break
            scores_local.sort(key=lambda x: x[0], reverse=True)

            accepted_in_round = False
            for sc, cand, rel, div in scores_local:
                ok_dedup, reason_dedup = passes_dedup(cand)
                ok_quota, reason_quota = passes_quotas_and_budget(cand)
                if ok_dedup and ok_quota:
                    try:
                        cand["mmr_score"] = float(sc)
                        cand["mmr_rel"] = float(rel)
                        cand["mmr_div"] = float(div)
                    except Exception:
                        pass
                    selected.append(cand)
                    _on_accept(cand)
                    pool = [c for c in pool if c is not cand]
                    added += 1
                    accepted_in_round = True
                    break
                else:
                    rejected.append({"candidate": cand, "reason": (reason_dedup if not ok_dedup else reason_quota), "score": float(sc)})
                    pool = [c for c in pool if c is not cand]
            if not accepted_in_round:
                break
        return added

    if len(selected) < int(rc.CONTEXT_TOP_K):
        progressed = True
        while progressed and len(selected) < int(rc.CONTEXT_TOP_K) and effective_dup_emb_cos_threshold < float(rc.RELAX_DUP_EMB_COS_MAX):
            try:
                effective_dup_emb_cos_threshold = float(min(float(rc.RELAX_DUP_EMB_COS_MAX), float(effective_dup_emb_cos_threshold) + float(rc.RELAX_DUP_EMB_COS_STEP)))
                relax_info["dup_cos_threshold"] = float(effective_dup_emb_cos_threshold)
                relax_info["dup_cos_steps"] = int(relax_info.get("dup_cos_steps", 0)) + 1
            except Exception:
                break
            added_now = _attempt_fill()
            progressed = added_now > 0

    if len(selected) < int(rc.CONTEXT_TOP_K):
        ignore_per_page_cap = True
        relax_info["ignore_per_page"] = True
        _attempt_fill()

    if len(selected) < int(rc.CONTEXT_TOP_K):
        ignore_lang_min_cover = True
        relax_info["ignore_lang_cover"] = True
        _attempt_fill()

    if len(selected) < int(rc.CONTEXT_TOP_K):
        seen_ids = {int(c.get("chunk_id")) for c in selected if isinstance(c.get("chunk_id"), (int, np.integer))}
        fallback_added = 0
        for cand in reranked_top:
            if len(selected) >= int(rc.CONTEXT_TOP_K):
                break
            cid = cand.get("chunk_id")
            if isinstance(cid, (int, np.integer)) and int(cid) not in seen_ids:
                selected.append(cand)
                _on_accept(cand)
                seen_ids.add(int(cid))
                fallback_added += 1
        relax_info["fallback_tail_used"] = bool(fallback_added > 0)

    debug = {
        "used_tokens": int(used_tokens),
        "per_doc": per_doc_count,
        "per_page": {f"{k[0]}:p{k[1]}": v for k, v in per_page_count.items()},
        "lang_count": lang_count,
        "pool_initial": min(rc.MMR_POOL_K, len(reranked_top)),
        "selected": len(selected),
        "rejected": len(rejected),
        "relax": relax_info,
    }
    return {"selected": selected, "rejected": rejected, "debug": debug}


# --- Промпт и генерация ---

def build_generation_prompt(question: str, context_pack: List[Dict[str, Any]], sources_map: Dict[str, int], target_lang: str) -> List[Dict[str, str]]:
    """Собирает сообщения для Chat Completions с жёсткими правилами и источниками."""
    tl = target_lang if target_lang in {"ru", "en"} else "ru"

    def _shrink_text(text: Any, limit: int) -> str:
        if not isinstance(text, str):
            return ""
        if len(text) <= int(limit):
            return text
        return (text[: int(limit)].rstrip() + "…")

    sources_lines: List[str] = []
    for idx, item in enumerate(context_pack or []):
        label = f"S{idx+1}"
        cid = item.get("chunk_id")
        src = item.get("source")
        pg = item.get("page")
        c0 = None
        c1 = None
        try:
            if isinstance(cid, (int, np.integer)) and 0 <= int(cid) < len(rc.chunks_metadata):
                meta = rc.chunks_metadata[int(cid)]
                if src is None:
                    src = meta.get("source")
                if pg is None:
                    pg = meta.get("page")
                c0 = meta.get("char_start")
                c1 = meta.get("char_end")
        except Exception:
            pass
        header = f"{label} — {src}, p.{pg}"
        if isinstance(c0, int) and isinstance(c1, int):
            header = f"{header} ({c0}–{c1})"
        snippet = item.get("text_ref")
        if not isinstance(snippet, str):
            try:
                if isinstance(cid, (int, np.integer)) and 0 <= int(cid) < len(rc.chunks_metadata):
                    snippet = rc.chunks_metadata[int(cid)].get("text")
            except Exception:
                snippet = ""
        snippet = _shrink_text(snippet, int(rc.SNIPPET_MAX_CHARS))
        sources_lines.append(f"{header}\n{snippet}")

    system_content = (
        f"Отвечай только на языке: {tl}.\n"
        "Не вставляй текст на других языках.\n"
        "Используй исключительно информацию из раздела «Источники».\n"
        "Каждый ключевой тезис помечай ссылкой [S#] — номер из списка «Источники».\n"
        f"Если сведений недостаточно — начни ответ строкой: \"{rc.INSUFFICIENT_PREFIX}\" и перечисли, чего именно не хватает.\n"
        "Не добавляй внешние сведения и не пытайся угадывать.\n"
        "Если вопрос вне контекста источников — так и скажи, без «общих знаний».\n"
        "Не вставляй длинные прямые цитаты из источников; перефразируй, добавив [S#]. Допускаются короткие цитаты ≤ "
        f"{int(rc.MAX_QUOTE_WORDS)} слов при необходимости.\n\n"
        "Формат ответа:\n"
        "1) Короткий «вывод» (2–4 предложения).\n"
        "2) Структурированный разбор списком; у каждого пункта укажи [S#].\n"
        "3) Раздел «Ссылки» — список S# → {source}, p.{page}.\n"
        "Важно: именно ты должен проставлять [S#] в тексте ответа."
    )

    user_content = (
        f"Вопрос:\n{question}\n\n"
        "Источники:\n" + ("\n\n".join(sources_lines) if sources_lines else "(нет контекста)")
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _language_ratio_simple(text: str, target_lang: str) -> float:
    """Доля символов целевого языка среди всех букв."""
    if not isinstance(text, str) or not text:
        return 0.0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    if target_lang == 'ru':
        target_count = sum(1 for ch in letters if 'А' <= ch <= 'я' or ch in ('Ё', 'ё'))
    elif target_lang == 'en':
        target_count = sum(1 for ch in letters if ('A' <= ch <= 'Z') or ('a' <= ch <= 'z'))
    else:
        target_count = 0
    return float(target_count) / float(len(letters))


def _extract_used_sources(text: str, valid_labels: List[str]) -> List[str]:
    """Извлекает маркеры ссылок вида [S\d+] и фильтрует по списку валидных меток."""
    if not isinstance(text, str) or not text:
        return []
    try:
        ids = re.findall(r"\[S(\d+)\]", text)
        labels = [f"S{int(x)}" for x in ids]
        uniq = sorted({lab for lab in labels if lab in set(valid_labels)}, key=lambda s: int(s[1:]))
        return uniq
    except Exception:
        return []


def create_rag_chain(
    openrouter_api_key: str,
    openrouter_model: str = rc.OPENROUTER_MODEL,
    temperature: float | None = None,
    max_tokens: int | None = None,
    enforce_citations: bool = True,
    language_enforcement: bool = True,
):
    """Создаёт RAG-цепочку, интегрируя перевод, гибридный поиск и реранжирование."""
    rc.runtime_openrouter_api_key = openrouter_api_key
    rc.runtime_openrouter_model = openrouter_model
    rc.runtime_llm_temperature = float(temperature) if isinstance(temperature, (int, float)) else rc.LLM_DEFAULT_TEMPERATURE
    rc.runtime_llm_max_tokens = int(max_tokens) if isinstance(max_tokens, (int, float)) else rc.LLM_DEFAULT_MAX_TOKENS
    rc.runtime_enforce_citations = bool(enforce_citations)
    rc.runtime_language_enforcement = bool(language_enforcement)

    def answer_question(question: str, apply_lang_quota: bool | None = None) -> Dict[str, Any]:
        target_lang = _detect_question_lang(question)
        if target_lang not in {"ru", "en"}:
            target_lang = "ru"

        hybrid_out = hybrid_search_with_rerank(question, apply_lang_quota=bool(apply_lang_quota) if apply_lang_quota is not None else True)
        context_pack = hybrid_out.get("context_pack", []) or []
        reranked = hybrid_out.get("reranked", []) or []
        sources_map = hybrid_out.get("sources_map", {}) or {}

        insufficient = False
        if not context_pack:
            insufficient = True
        else:
            try:
                rerank_sum = float(sum(float(it.get("rerank_score", 0.0)) for it in reranked))
            except Exception:
                rerank_sum = 0.0
            if float(rc.INSUFFICIENT_MIN_RERANK_SUM) > 0.0 and rerank_sum < float(rc.INSUFFICIENT_MIN_RERANK_SUM):
                insufficient = True
            try:
                has_target_lang = False
                for it in context_pack:
                    cid = it.get("chunk_id")
                    if isinstance(cid, (int, np.integer)) and 0 <= int(cid) < len(rc.chunks_metadata):
                        if rc.chunks_metadata[int(cid)].get("lang") == target_lang:
                            has_target_lang = True
                            break
                # Требование совпадения языка контекста с языком вопроса делаем опциональным.
                # По умолчанию не блокируем кросс-языковые ответы (см. rc.REQUIRE_TARGET_LANG_IN_CONTEXT).
                if bool(getattr(rc, "REQUIRE_TARGET_LANG_IN_CONTEXT", False)):
                    if target_lang in {"ru", "en"} and not has_target_lang:
                        insufficient = True
            except Exception:
                pass

        if insufficient:
            # Фолбэки при отсутствии контекста
            # 1) Веб-фолбэк (модель с доступом к интернету через OpenRouter)
            if bool(getattr(rc, "FALLBACK_WEB_ENABLE", False)) and isinstance(getattr(rc, "FALLBACK_WEB_MODEL", None), str):
                web_system = (
                    f"Отвечай только на языке: {target_lang}.\n" +
                    str(getattr(rc, "FALLBACK_WEB_SYSTEM_PROMPT", ""))
                )
                web_messages = [
                    {"role": "system", "content": web_system},
                    {"role": "user", "content": question},
                ]
                try:
                    web_json = call_openrouter_chat_completion(
                        api_key=rc.runtime_openrouter_api_key,
                        model=str(getattr(rc, "FALLBACK_WEB_MODEL")),
                        messages=web_messages,
                        extra_request_kwargs={
                            "temperature": float(rc.runtime_llm_temperature if rc.runtime_llm_temperature is not None else rc.LLM_DEFAULT_TEMPERATURE),
                            "max_tokens": int(rc.runtime_llm_max_tokens if rc.runtime_llm_max_tokens is not None else rc.LLM_DEFAULT_MAX_TOKENS),
                        },
                    )
                    web_answer = web_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or web_json.get("choices", [{}])[0].get("text", "").strip()
                except Exception:
                    web_answer = ""
                if web_answer:
                    ans_lang = _detect_question_lang(web_answer) or target_lang
                    return {
                        "final_answer": web_answer,
                        "used_sources": [],
                        "answer_lang_detected": ans_lang,
                        "flags": {
                            "regenerated_for_lang": False,
                            "regenerated_for_citations": False,
                            "insufficient": False,
                            "fallback_web": True,
                            "fallback_freeform": False,
                        },
                    }

            # 2) Свободный фолбэк (без интернет-поиска)
            if bool(getattr(rc, "FALLBACK_FREEFORM_ENABLE", False)):
                ff_system = (
                    f"Отвечай только на языке: {target_lang}.\n" +
                    str(getattr(rc, "FALLBACK_FREEFORM_SYSTEM_PROMPT", ""))
                )
                ff_messages = [
                    {"role": "system", "content": ff_system},
                    {"role": "user", "content": question},
                ]
                try:
                    ff_json = call_openrouter_chat_completion(
                        api_key=rc.runtime_openrouter_api_key,
                        model=rc.runtime_openrouter_model,
                        messages=ff_messages,
                        extra_request_kwargs={
                            "temperature": float(rc.runtime_llm_temperature if rc.runtime_llm_temperature is not None else rc.LLM_DEFAULT_TEMPERATURE),
                            "max_tokens": int(rc.runtime_llm_max_tokens if rc.runtime_llm_max_tokens is not None else rc.LLM_DEFAULT_MAX_TOKENS),
                        },
                    )
                    ff_answer = ff_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or ff_json.get("choices", [{}])[0].get("text", "").strip()
                except Exception:
                    ff_answer = ""
                if ff_answer:
                    ans_lang = _detect_question_lang(ff_answer) or target_lang
                    return {
                        "final_answer": ff_answer,
                        "used_sources": [],
                        "answer_lang_detected": ans_lang,
                        "flags": {
                            "regenerated_for_lang": False,
                            "regenerated_for_citations": False,
                            "insufficient": False,
                            "fallback_web": False,
                            "fallback_freeform": True,
                        },
                    }

            # 3) Жёсткий ответ об отсутствии знаний (по умолчанию)
            return {
                "final_answer": rc.INSUFFICIENT_ANSWER,
                "used_sources": [],
                "answer_lang_detected": target_lang,
                "flags": {
                    "regenerated_for_lang": False,
                    "regenerated_for_citations": False,
                    "insufficient": True,
                    "fallback_web": False,
                    "fallback_freeform": False,
                },
            }

        messages = build_generation_prompt(
            question=question,
            context_pack=context_pack,
            sources_map=sources_map,
            target_lang=target_lang,
        )

        response_json = call_openrouter_chat_completion(
            api_key=rc.runtime_openrouter_api_key,
            model=rc.runtime_openrouter_model,
            messages=messages,
            extra_request_kwargs={
                "temperature": float(rc.runtime_llm_temperature if rc.runtime_llm_temperature is not None else rc.LLM_DEFAULT_TEMPERATURE),
                "max_tokens": int(rc.runtime_llm_max_tokens if rc.runtime_llm_max_tokens is not None else rc.LLM_DEFAULT_MAX_TOKENS),
            },
        )
        draft = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or response_json.get("choices", [{}])[0].get("text", "").strip()

        regenerated_for_lang = False
        regenerated_for_citations = False

        if rc.runtime_language_enforcement:
            ratio = _language_ratio_simple(draft, target_lang)
            if ratio < float(rc.REGEN_LANG_RATIO_THRESHOLD):
                debug_log("[REGEN_LANG] ratio=%.3f < thr=%.3f, target=%s", float(ratio), float(rc.REGEN_LANG_RATIO_THRESHOLD), str(target_lang))
                regen_messages = [
                    {"role": "system", "content": f"Ты нарушил инструкции. Перепиши ответ строго на {target_lang} без примесей других языков, сохраняя смысл и цитаты [S#]."},
                    {"role": "user", "content": draft},
                ]
                response_json2 = call_openrouter_chat_completion(
                    api_key=rc.runtime_openrouter_api_key,
                    model=rc.runtime_openrouter_model,
                    messages=regen_messages,
                    extra_request_kwargs={
                        "temperature": float(rc.runtime_llm_temperature if rc.runtime_llm_temperature is not None else rc.LLM_DEFAULT_TEMPERATURE),
                        "max_tokens": int(rc.runtime_llm_max_tokens if rc.runtime_llm_max_tokens is not None else rc.LLM_DEFAULT_MAX_TOKENS),
                    },
                )
                draft2 = response_json2.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or response_json2.get("choices", [{}])[0].get("text", "").strip()
                if draft2:
                    draft = draft2
                    regenerated_for_lang = True

        valid_labels = [f"S{i+1}" for i in range(len(context_pack))]
        used_sources = _extract_used_sources(draft, valid_labels)

        if rc.runtime_enforce_citations and context_pack and not used_sources:
            debug_log("[REGEN_CIT] no citations in draft, enforcing")
            msg_sources = []
            for idx, item in enumerate(context_pack):
                label = f"S{idx+1}"
                src = item.get("source")
                pg = item.get("page")
                msg_sources.append(f"{label} — {src}, p.{pg}")
            regen_messages_cit = [
                {"role": "system", "content": (
                    f"Отвечай только на языке: {target_lang}. Добавь явные ссылки [S#] на тезисы. "
                    "Строго используй S# из списка \"Источники\" ниже. "
                    "Не вставляй длинные прямые цитаты; перефразируй."
                )},
                {"role": "user", "content": (
                    f"Вопрос:\n{question}\n\nИсточники:\n" + "\n".join(msg_sources) + "\n\nТекущий ответ:\n" + draft
                )},
            ]
            response_json3 = call_openrouter_chat_completion(
                api_key=rc.runtime_openrouter_api_key,
                model=rc.runtime_openrouter_model,
                messages=regen_messages_cit,
                extra_request_kwargs={
                    "temperature": float(rc.runtime_llm_temperature if rc.runtime_llm_temperature is not None else rc.LLM_DEFAULT_TEMPERATURE),
                    "max_tokens": int(rc.runtime_llm_max_tokens if rc.runtime_llm_max_tokens is not None else rc.LLM_DEFAULT_MAX_TOKENS),
                },
            )
            draft3 = response_json3.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or response_json3.get("choices", [{}])[0].get("text", "").strip()
            if draft3:
                draft = draft3
                regenerated_for_citations = True
                used_sources = _extract_used_sources(draft, valid_labels)

        answer_lang_detected = _detect_question_lang(draft)
        return {
            "final_answer": draft,
            "used_sources": used_sources,
            "answer_lang_detected": answer_lang_detected,
            "flags": {
                "regenerated_for_lang": bool(regenerated_for_lang),
                "regenerated_for_citations": bool(regenerated_for_citations),
                "insufficient": False,
            },
        }

    return {"answer_question": answer_question}
