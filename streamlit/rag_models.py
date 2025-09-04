import os
import multiprocessing as mp
from multiprocessing import queues
import numpy as np
from typing import List, Tuple

# Модели
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

# Язык/токенизация
from lingua import Language, LanguageDetectorBuilder
import Stemmer
import stopwordsiso as stopwords

import rag_core as rc


_embed_in_queue: queues.Queue | None = None
_embed_out_queue: queues.Queue | None = None
_embed_workers: list[mp.Process] = []
_embed_jobs_submitted: int = 0


def _embed_worker(device_id: int, model_name: str, max_length: int, in_q: queues.Queue, out_q: queues.Queue) -> None:
    """Процесс-воркер, загружающий модель один раз на GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch  # type: ignore

    torch.set_grad_enabled(False)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    torch.cuda.set_device(0)

    from sentence_transformers import SentenceTransformer as _ST  # локальный импорт
    model = _ST(model_name, trust_remote_code=True, device="cuda")
    model.max_seq_length = int(max_length)
    model.eval()

    while True:
        job = in_q.get()
        if job is None:
            break
        start_idx, texts = job
        with torch.inference_mode():
            emb = model.encode(
                sentences=texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        out_q.put((start_idx, emb))
        torch.cuda.empty_cache()
    del model


def start_embed_workers(devices: List[int], model_name: str, max_length: int, batch_size: int) -> None:
    """Стартует воркеры энкодера на указанных GPU."""
    global _embed_in_queue, _embed_out_queue, _embed_workers, _embed_jobs_submitted
    _embed_in_queue = mp.Queue()
    _embed_out_queue = mp.Queue()
    _embed_workers = []
    _embed_jobs_submitted = 0
    for dev in devices:
        p = mp.Process(target=_embed_worker, args=(dev, model_name, max_length, _embed_in_queue, _embed_out_queue))
        p.daemon = True
        p.start()
        _embed_workers.append(p)


def submit_embed(job: Tuple[int, list[str]]) -> None:
    """Отправляет задание на кодирование эмбеддингов."""
    global _embed_jobs_submitted
    if _embed_in_queue is None:
        raise RuntimeError("Embed workers not started")
    _embed_in_queue.put(job)
    _embed_jobs_submitted += 1


def drain_embed() -> list[Tuple[int, np.ndarray]]:
    """Ожидает результаты всех отправленных задач."""
    global _embed_jobs_submitted
    results: list[Tuple[int, np.ndarray]] = []
    if _embed_out_queue is None:
        return results
    for _ in range(_embed_jobs_submitted):
        results.append(_embed_out_queue.get())
    _embed_jobs_submitted = 0
    return results


def stop_embed_workers() -> None:
    """Останавливает все воркеры энкодера."""
    global _embed_in_queue, _embed_out_queue, _embed_workers
    if _embed_in_queue is None:
        return
    for _ in _embed_workers:
        _embed_in_queue.put(None)
    for p in _embed_workers:
        p.join()
    _embed_in_queue.close()
    _embed_out_queue.close()
    _embed_in_queue = None
    _embed_out_queue = None
    _embed_workers = []


def initialize_models(load_embedder: bool = True, load_reranker: bool = True) -> None:
    """Инициализация и кэширование моделей/инструментов в глобальном состоянии rc.*"""
    if load_embedder and rc.embedder is None:
        rc.embedder = SentenceTransformer(
            rc.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=rc.EMBED_DEVICES[0],
        )
    if load_reranker and rc.reranker is None:
        try:
            import torch  # type: ignore
            has_cuda = bool(getattr(torch.cuda, "is_available", lambda: False)())
            device = rc.RERANK_DEVICE if has_cuda else "cuda:0"
            use_fp16 = has_cuda
        except Exception:
            device = "cuda:0"
            use_fp16 = False
        rc.reranker = FlagReranker(
            rc.RERANKER_MODEL_NAME,
            use_fp16=use_fp16,
            max_length=rc.RERANK_MAX_LENGTH,
            device=device,
        )
    if rc.language_detector is None:
        try:
            langs = [Language.RUSSIAN, Language.ENGLISH, Language.UKRAINIAN, Language.BELARUSIAN]
            rc.language_detector = LanguageDetectorBuilder.from_languages(*langs).with_low_accuracy_mode().build()
        except Exception:
            rc.language_detector = None
    if rc.ru_stemmer is None:
        try:
            rc.ru_stemmer = Stemmer.Stemmer('russian')
        except Exception:
            rc.ru_stemmer = None
    if rc.en_stemmer is None:
        try:
            rc.en_stemmer = Stemmer.Stemmer('english')
        except Exception:
            rc.en_stemmer = None
    if not rc.ru_stopwords:
        try:
            rc.ru_stopwords = set(stopwords.stopwords('ru'))
        except Exception:
            rc.ru_stopwords = set()
    if not rc.en_stopwords:
        try:
            rc.en_stopwords = set(stopwords.stopwords('en'))
        except Exception:
            rc.en_stopwords = set()

