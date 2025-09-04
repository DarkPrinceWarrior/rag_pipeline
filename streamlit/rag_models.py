import os
import multiprocessing as mp
import numpy as np

# Модели
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

# Язык/токенизация
from lingua import Language, LanguageDetectorBuilder
import Stemmer
import stopwordsiso as stopwords

import rag_core as rc


def _chunk_iter(items: list[str], size: int) -> list[list[str]]:
    """Разбиение списка на батчи фиксированного размера."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def _worker_encode(args):
    """Воркер-процесс для мульти-GPU энкодинга эмбеддингов.

    Параметры: (device_id, model_name, texts)
    Возврат: np.ndarray float32 с L2-нормализацией на уровне модели.
    """
    device_id, model_name, texts = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    from sentence_transformers import SentenceTransformer as _ST  # локальный импорт
    import torch  # type: ignore

    torch.cuda.set_device(0)
    model = _ST(model_name, trust_remote_code=True, device="cuda")
    emb = model.encode(
        sentences=texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return emb


def encode_multi_gpu(texts: list[str], batch_size: int, gpu_ids: list[int]) -> np.ndarray:
    """Распределённое кодирование эмбеддингов по нескольким GPU."""
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    batches = _chunk_iter(texts, batch_size)
    tasks = []
    for i, b in enumerate(batches):
        device_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((device_id, rc.EMBEDDING_MODEL_NAME, b))
    from multiprocessing import Pool
    with Pool(processes=min(len(tasks), len(gpu_ids))) as pool:
        results = pool.map(_worker_encode, tasks)
    return np.concatenate(results, axis=0)


def initialize_models() -> None:
    """Инициализация и кэширование моделей/инструментов в глобальном состоянии rc.*"""
    if rc.embedder is None:
        rc.embedder = SentenceTransformer(
            rc.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=f"cuda:{rc.EMBED_GPU_IDS[0]}",
        )
    if rc.reranker is None:
        try:
            import torch  # type: ignore
            has_cuda = bool(getattr(torch.cuda, "is_available", lambda: False)())
            device = f"cuda:{rc.RERANK_GPU_ID}" if has_cuda else "cuda:0"
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
