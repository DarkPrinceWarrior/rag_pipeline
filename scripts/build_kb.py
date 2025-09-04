from rag_ingestion import build_and_load_knowledge_base
import rag_core as rc

if __name__ == "__main__":
    build_and_load_knowledge_base(
        rc.PDF_DIR,
        rc.VECTOR_STORE_PATH,
        force_rebuild=False,
    )
