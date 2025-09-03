<!-- export RAG_GPU_IDS_EMBED="0,1"
export RAG_GPU_IDS_RERANK="2"
export RAG_EMBED_BATCH="64"

python3 -c 'from almaservice.rag_pipeline.streamlit import rag_core as rc; rc.build_and_load_knowledge_base(pdf_dir="/root/almaservice/rag_pipeline/pdfs", index_dir="/root/almaservice/rag_pipeline/faiss_index", force_rebuild=True)' -->