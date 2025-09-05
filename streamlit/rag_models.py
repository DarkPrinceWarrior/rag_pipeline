import os
import multiprocessing as mp
import numpy as np
from multiprocessing.queues import Queue as MPQueue
from typing import Any

# Модели
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from concurrent.futures import ThreadPoolExecutor

# Язык/токенизация
from lingua import Language, LanguageDetectorBuilder
import Stemmer
import stopwordsiso as stopwords

import rag_core as rc


def _chunk_iter(items: list[str], size: int) -> list[list[str]]:
    """Разбиение списка на батчи фиксированного размера."""
    return [items[i:i + size] for i in range(0, len(items), size)]


"""
Мульти-GPU энкодинг с «стойкими» воркерами.

Воркеры создаются один раз на GPU, загружают модель и обрабатывают задания из очереди.
encode_multi_gpu отправляет задания по шардированным частям входного списка и собирает
результаты, восстанавливая исходный порядок по позициям.
"""

_EMBED_CTX: mp.context.BaseContext | None = None
_EMBED_TASK_QUEUE: MPQueue | None = None
_EMBED_RESULT_QUEUE: MPQueue | None = None
_EMBED_WORKERS: list[mp.Process] = []


def _embed_worker_main(device_id: int, model_name: str, batch_size: int, task_q: MPQueue, result_q: MPQueue) -> None:
    """Основной цикл воркера энкодинга на конкретном устройстве.

    Ожидает задания вида (job_id, shard_id, positions, texts) и возвращает
    (job_id, shard_id, positions, embeddings: np.ndarray float32).
    """
    try:
        try:
            import torch  # type: ignore
            has_cuda = bool(getattr(torch.cuda, "is_available", lambda: False)())
        except Exception:
            has_cuda = False
        device = f"cuda:{int(device_id)}" if has_cuda else os.getenv("RAG_DEFAULT_CUDA_DEVICE", "cuda:0")
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    except Exception:
        model = SentenceTransformer(model_name, trust_remote_code=True)

    while True:
        task = task_q.get()
        if task is None:
            break
        job_id, shard_id, positions, texts = task
        outputs: list[np.ndarray] = []
        for batch in _chunk_iter(texts, int(batch_size)):
            if not batch:
                continue
            emb = model.encode(
                sentences=batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            outputs.append(emb)
        arr = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype="float32")
        result_q.put((job_id, shard_id, positions, arr))


def start_embed_workers(gpu_ids: list[int], model_name: str, batch_size: int) -> None:
    """Запускает воркеры энкодинга: по одному процессу на GPU.

    Воркеры получают задания через общую очередь и возвращают результаты в общую очередь.
    """
    global _EMBED_CTX, _EMBED_TASK_QUEUE, _EMBED_RESULT_QUEUE, _EMBED_WORKERS
    if _EMBED_WORKERS:
        return
    _EMBED_CTX = mp.get_context("spawn")
    _EMBED_TASK_QUEUE = _EMBED_CTX.Queue()
    _EMBED_RESULT_QUEUE = _EMBED_CTX.Queue()
    _EMBED_WORKERS = []
    for gid in gpu_ids:
        p = _EMBED_CTX.Process(
            target=_embed_worker_main,
            args=(int(gid), model_name, int(batch_size), _EMBED_TASK_QUEUE, _EMBED_RESULT_QUEUE),
            daemon=True,
        )
        p.start()
        _EMBED_WORKERS.append(p)


def submit_embed(job_id: int, shard_id: int, positions: list[int], texts: list[str]) -> None:
    """Отправляет задание на энкодинг в очередь."""
    if _EMBED_TASK_QUEUE is None:
        raise RuntimeError("Очередь заданий не инициализирована. Вызовите start_embed_workers().")
    _EMBED_TASK_QUEUE.put((int(job_id), int(shard_id), positions, texts))


def drain_embed(job_id: int, expected_tasks: int, total_count: int) -> np.ndarray:
    """Собирает результаты для указанного job_id, восстанавливая порядок по positions.

    expected_tasks — число шардов, отправленных в submit_embed.
    total_count — общее число текстов во входе.
    """
    if _EMBED_RESULT_QUEUE is None:
        raise RuntimeError("Очередь результатов не инициализирована. Вызовите start_embed_workers().")
    received = 0
    out: np.ndarray | None = None
    while received < int(expected_tasks):
        r_job, _shard, positions, arr = _EMBED_RESULT_QUEUE.get()
        if r_job != job_id:
            # В текущем дизайне конкурирующих job нет, но проверка остаётся на будущее
            continue
        if out is None:
            dim = int(arr.shape[1]) if isinstance(arr, np.ndarray) and arr.size > 0 else 0
            out = np.zeros((int(total_count), dim), dtype="float32") if dim > 0 else np.zeros((0, 0), dtype="float32")
        if out.size > 0 and isinstance(arr, np.ndarray) and arr.size > 0:
            for i, pos in enumerate(positions):
                if i < arr.shape[0]:
                    out[int(pos)] = arr[i]
        received += 1
    return out if out is not None else np.zeros((0, 0), dtype="float32")


def stop_embed_workers() -> None:
    """Останавливает воркеры и очищает ресурсы."""
    global _EMBED_CTX, _EMBED_TASK_QUEUE, _EMBED_RESULT_QUEUE, _EMBED_WORKERS
    if not _EMBED_WORKERS:
        return
    if _EMBED_TASK_QUEUE is not None:
        for _ in _EMBED_WORKERS:
            _EMBED_TASK_QUEUE.put(None)
    for p in _EMBED_WORKERS:
        try:
            p.join()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    _EMBED_WORKERS = []
    _EMBED_TASK_QUEUE = None
    _EMBED_RESULT_QUEUE = None
    _EMBED_CTX = None


def encode_multi_gpu(texts: list[str], batch_size: int, gpu_ids: list[int]) -> np.ndarray:
    """Кодирование эмбеддингов на нескольких GPU с восстановлением порядка.

    - Делим вход на шардов по числу GPU (равными по количеству элементов).
    - Отправляем задания воркерам с позициями исходных элементов.
    - Собираем результаты и размещаем по соответствующим позициям.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    gids = gpu_ids if gpu_ids else rc.EMBED_GPU_IDS
    num_gpus = max(1, len(gids))

    # План разбиения: равномерно по количеству
    n = len(texts)
    shard_sizes = [(n + i) // num_gpus for i in range(num_gpus)]
    shards_pos: list[list[int]] = []
    shards_txt: list[list[str]] = []
    start = 0
    for sz in shard_sizes:
        end = start + sz
        positions = list(range(start, end))
        texts_slice = texts[start:end]
        shards_pos.append(positions)
        shards_txt.append(texts_slice)
        start = end

    # Запуск воркеров и отправка заданий
    start_embed_workers(gids, rc.EMBEDDING_MODEL_NAME, int(batch_size))
    job_id = 1
    expected = 0
    for shard_id, (pos, txt) in enumerate(zip(shards_pos, shards_txt)):
        if not txt:
            continue
        submit_embed(job_id, shard_id, pos, txt)
        expected += 1

    if expected == 0:
        stop_embed_workers()
        return np.zeros((0, 0), dtype="float32")

    out = drain_embed(job_id, expected, n)
    stop_embed_workers()
    return out


def _optimize_torch_for_ampere() -> None:
    """Включает оптимизации TF32 и autotune cuDNN."""
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def build_reranker_pools() -> list[FlagReranker]:
    """Создаёт экземпляр реранкера на каждый GPU из rc.RERANK_GPU_IDS."""
    rerankers: list[FlagReranker] = []
    for gid in rc.RERANK_GPU_IDS:
        rr = FlagReranker(
            rc.RERANKER_MODEL_NAME,
            use_fp16=True,
            devices=[int(gid)],
        )
        rerankers.append(rr)
    return rerankers


def rerank_multi_gpu(query: str, docs: list[str], rerankers: list[FlagReranker]) -> list[float]:
    """Параллельный реранк кандидатов на нескольких GPU."""
    if not docs:
        return []
    shards = np.array_split(np.arange(len(docs)), len(rerankers))
    results: list[np.ndarray] = []
    with ThreadPoolExecutor(len(rerankers)) as ex:
        futures = []
        for rr, idx in zip(rerankers, shards):
            pairs = [(query, docs[i]) for i in idx]
            futures.append(
                ex.submit(
                    rr.compute_score,
                    pairs,
                    batch_size=rc.RERANK_BATCH_SIZE,
                    max_length=rc.RERANK_MAX_LENGTH,
                )
            )
        for f in futures:
            results.append(np.array(f.result(), dtype=np.float32))
    out = np.empty(len(docs), dtype=np.float32)
    for idx, scores in zip(shards, results):
        out[idx] = scores
    return out.tolist()


def initialize_models(load_embedder: bool = True, load_reranker: bool = True) -> None:
    """Инициализация и кэширование моделей/инструментов в глобальном состоянии rc.*"""

    _optimize_torch_for_ampere()

    if load_embedder and rc.embedder is None:
        rc.embedder = SentenceTransformer(
            rc.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=f"cuda:{rc.EMBED_GPU_IDS[0]}",
        )
    if load_reranker and not rc.reranker_pools:
        rc.reranker_pools = build_reranker_pools()
        if rc.reranker_pools:
            rc.reranker = rc.reranker_pools[0]
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
