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


# ---------------------------
# Слой стойких воркеров эмбеддингов
# ---------------------------

# Глобальное состояние воркеров
_embed_workers: list[mp.Process] = []
_embed_in_queue: mp.Queue | None = None
_embed_out_queue: mp.Queue | None = None
_embed_pending_jobs: int = 0


def _normalize_device_to_visible_token(device: int | str) -> str:
    """Преобразует устройство (int, 'cuda:N' или UUID) в токен для CUDA_VISIBLE_DEVICES."""
    try:
        if isinstance(device, int):
            return str(int(device))
        s = str(device)
        if s.startswith("cuda:"):
            return s.split(":", 1)[1]
        return s
    except Exception:
        return str(device)


def _embed_worker_main(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    device_token: str,
    model_name: str,
    max_length: int,
    inner_batch_size: int,
) -> None:
    """Главная функция GPU-воркера для кодирования эмбеддингов.

    Воркёр загружает модель один раз и обрабатывает задания из своей очереди.
    """
    # Настройки среды и CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_token)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer as _ST  # локальный импорт

        torch.set_grad_enabled(False)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.benchmark = True  # type: ignore
            except Exception:
                pass
        torch.cuda.set_device(0)

        model = _ST(model_name, trust_remote_code=True, device="cuda")
        try:
            model.max_seq_length = int(max_length)
        except Exception:
            pass
        try:
            model.eval()
        except Exception:
            pass

        while True:
            job = in_queue.get()
            if job is None:
                break
            start_idx, texts = job  # job = (global_start_idx, list_of_texts)
            if not isinstance(texts, list) or len(texts) == 0:
                out_queue.put((int(start_idx), np.zeros((0, 0), dtype="float32")))
                continue
            try:
                with torch.inference_mode():  # type: ignore
                    emb = model.encode(
                        sentences=texts,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=int(inner_batch_size),
                    ).astype("float32")
                out_queue.put((int(start_idx), emb))
            except Exception:
                # В случае ошибки возвращаем пустой результат, чтобы не блокировать пайплайн
                out_queue.put((int(start_idx), np.zeros((0, 0), dtype="float32")))
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        # Критическая ошибка инициализации воркера — читаем задания, чтобы не повисли, и завершаем
        while True:
            job = in_queue.get()
            if job is None:
                break
            try:
                start_idx, _ = job
                out_queue.put((int(start_idx), np.zeros((0, 0), dtype="float32")))
            except Exception:
                pass


def start_embed_workers(devices: list[str | int], model_name: str, max_length: int, batch_size: int) -> None:
    """Запускает N GPU-воркеров по числу устройств.

    Каждый воркер:
    - выставляет CUDA_VISIBLE_DEVICES и выбирает cuda:0 внутри процесса;
    - один раз загружает эмбеддер в eval() и работает в режиме inference;
    - читает задания из своей очереди и отправляет результаты в общую out-очередь.
    """
    global _embed_workers, _embed_in_queue, _embed_out_queue, _embed_pending_jobs
    stop_embed_workers()
    if not isinstance(devices, list) or len(devices) == 0:
        _embed_workers = []
        _embed_in_queue = None
        _embed_out_queue = None
        _embed_pending_jobs = 0
        return
    ctx = mp.get_context("spawn")
    _embed_in_queue = ctx.Queue(maxsize=256)
    _embed_out_queue = ctx.Queue(maxsize=256)
    _embed_workers = []
    for i, dev in enumerate(devices):
        token = _normalize_device_to_visible_token(dev)
        p = ctx.Process(
            target=_embed_worker_main,
            args=(
                _embed_in_queue,
                _embed_out_queue,
                str(token),
                str(model_name),
                int(max_length),
                int(batch_size),
            ),
            daemon=True,
        )
        p.start()
        _embed_workers.append(p)
    _embed_pending_jobs = 0


def submit_embed(job: tuple[int, list[str]]) -> None:
    """Ставит задание на кодирование: job=(global_start_idx, list_of_texts)."""
    global _embed_pending_jobs
    if _embed_in_queue is None:
        raise RuntimeError("Воркеры эмбеддингов не запущены: вызовите start_embed_workers().")
    _embed_in_queue.put(job)
    _embed_pending_jobs += 1


def drain_embed() -> list[tuple[int, np.ndarray]]:
    """Ожидает завершения всех поставленных заданий и возвращает результаты."""
    global _embed_pending_jobs
    if _embed_out_queue is None:
        return []
    results: list[tuple[int, np.ndarray]] = []
    for _ in range(int(_embed_pending_jobs)):
        res = _embed_out_queue.get()
        results.append(res)
    _embed_pending_jobs = 0
    return results


def stop_embed_workers() -> None:
    """Останавливает воркеров и очищает ресурсы."""
    global _embed_workers, _embed_in_queue, _embed_out_queue, _embed_pending_jobs
    if _embed_in_queue is not None and _embed_workers:
        for _ in _embed_workers:
            try:
                _embed_in_queue.put(None)
            except Exception:
                pass
    if _embed_workers:
        for p in _embed_workers:
            try:
                p.join(timeout=5)
            except Exception:
                pass
    _embed_workers = []
    _embed_in_queue = None
    _embed_out_queue = None
    _embed_pending_jobs = 0


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
