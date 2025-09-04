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

    Параметры: (device_id, model_name, texts, batch_size)
    Возврат: np.ndarray float32 с L2-нормализацией на уровне модели.
    """
    device_id, model_name, texts, batch_size = args
    from sentence_transformers import SentenceTransformer as _ST  # локальный импорт

    device = f"cuda:{int(device_id)}"
    model = _ST(model_name, trust_remote_code=True, device=device)
    batches = _chunk_iter(texts, int(batch_size))
    outputs: list[np.ndarray] = []
    for b in batches:
        emb = model.encode(
            sentences=b,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        outputs.append(emb)
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype="float32")


def encode_multi_gpu(texts: list[str], batch_size: int, gpu_ids: list[int]) -> np.ndarray:
    """Распределённое кодирование эмбеддингов по нескольким GPU.

    Алгоритм:
    - Делим вход на равные по количеству чанки по числу доступных GPU.
    - На каждый GPU запускается один процесс, который один раз загружает модель
      на свой `cuda:{id}` и обрабатывает свой срез входа батчами.
    Это гарантирует равномерное распределение памяти и отсутствие лавины на `cuda:0`.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    num_gpus = max(1, len(gpu_ids))
    # Равномерное разбиение входа по GPU
    shard_sizes = [(len(texts) + i) // num_gpus for i in range(num_gpus)]
    shards: list[list[str]] = []
    start = 0
    for sz in shard_sizes:
        end = start + sz
        shards.append(texts[start:end])
        start = end

    tasks = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        device_id = gpu_ids[i % num_gpus]
        # Одна задача на один GPU: внутри задачи выполняется батчинг
        tasks.append((device_id, rc.EMBEDDING_MODEL_NAME, shard, int(batch_size)))

    if not tasks:
        return np.zeros((0, 0), dtype="float32")

    from multiprocessing import Pool
    # Ровно по одному воркеру на задачу/GPU, чтобы каждая модель была загружена один раз
    with Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_encode, tasks)
    return np.concatenate(results, axis=0)


def initialize_models(load_embedder: bool = True, load_reranker: bool = True) -> None:
    """Инициализация и кэширование моделей/инструментов в глобальном состоянии rc.*

    Аргументы управляют отложенной загрузкой крупных моделей для экономии GPU-памяти
    в режимах, где они не требуются (например, при построении индекса).
    """
    if load_embedder and rc.embedder is None:
        rc.embedder = SentenceTransformer(
            rc.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=f"cuda:{rc.EMBED_GPU_IDS[0]}",
        )
    if load_reranker and rc.reranker is None:
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
