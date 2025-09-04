<!-- cd /root/almaservice/rag_pipeline && \
export RAG_EMBED_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6" \
RAG_RERANK_DEVICE="cuda:7" \
RAG_EMBED_BATCH=64 \
RAG_EMBED_MAX_LENGTH_TOKENS=1024 \
RAG_FAISS_CPU_THREADS="$(nproc)" && \
python - <<'PY' && \
streamlit run /root/almaservice/rag_pipeline/streamlit/app.py --server.headless true --server.port 8501
import sys
sys.path.append('/root/almaservice/rag_pipeline/streamlit')
import rag_core as rc
from rag_ingestion import build_and_load_knowledge_base
build_and_load_knowledge_base(rc.PDF_DIR, rc.VECTOR_STORE_PATH, force_rebuild=False)
PY -->