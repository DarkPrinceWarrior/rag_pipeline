<!-- # Какие GPU видим
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Разводим нагрузки: эмбеддинг на 6 GPU, реранк — на 2 GPU
export RAG_GPU_IDS_EMBED=0,1,2,3,4,5
export RAG_GPU_IDS_RERANK=6,7

# Модели (современные, быстрые, мультиязычные)
export RAG_EMBEDDING_MODEL="google/embeddinggemma-300m"   # 1024-d, MRL/квант.-friendly п
export RAG_RERANK_MODEL="BAAI/bge-reranker-v2-m3"          # быстрый SOTA реранкер v2

# Батчи (стартовые — под A5000 24GB; при длинных текстах можно снизить)
export RAG_EMBED_BATCH=128
export RAG_RERANK_BATCH_SIZE=96

# FAISS: оставляем на CPU (HNSW на GPU не работает)
export RAG_FAISS_USE_GPU=0
export RAG_FAISS_CPU_THREADS=$(nproc)  # 24

# PyTorch: память и TF32 (ускоряет матмулы на Ampere)
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:64,expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=1

# NCCL для одной машины без InfiniBand
export NCCL_IB_DISABLE=1   # не пытаться лезть в IB/RoCE
# P2P NVLink/PCIe оставляем включённым (по умолчанию)

# Токенайзеры: глушим ворнинг про fork/parallelism
export TOKENIZERS_PARALLELISM=false

# CPU/BLAS: избегаем оверсабскрипшена потоков
export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Кэши (на быстрый SSD)
export HF_HOME=/mnt/ssd/hf
export HF_HUB_CACHE=/mnt/ssd/hf/hub
export TRANSFORMERS_CACHE=/mnt/ssd/hf/hub
export HF_DATASETS_CACHE=/mnt/ssd/hf/datasets

# Запуск твоих скриптов
python scripts/build_kb.py
streamlit run streamlit/app.py --server.headless true --server.port 8501
 -->
