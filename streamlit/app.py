# streamlit/app.py
# streamlit/app.py
import streamlit as st
import os
from pathlib import Path
import shutil
# Импортируем обновленные функции
from rag_core import build_and_load_knowledge_base, create_rag_chain, _load_api_key_from_env, hybrid_search_with_rerank
import rag_core as rc

# --- Конфигурация страницы ---
st.set_page_config(
    page_title="RAG-чат с вашими PDF",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG-чат с вашими PDF-документами")
st.markdown("""
Это приложение позволяет вам задавать вопросы к вашим собственным PDF-документам.
**Как это работает:**
1.  **Введите ваш OpenRouter API ключ** в боковой панели.
2.  **Загрузите один или несколько PDF-файлов**.
3.  Нажмите кнопку **"Обработать документы"**, чтобы создать базу знаний.
4.  **Задайте свой вопрос** и получите ответ, основанный на содержании ваших файлов.
""")

# --- Боковая панель ---
with st.sidebar:
    st.header("Настройки")
    # Код для API-ключа остается без изменений...
    try:
        # Пытаемся загрузить ключ из .env, если он существует на уровне выше
        from rag_core import _load_api_key_from_env
        default_key = _load_api_key_from_env()
    except (RuntimeError, ImportError):
        default_key = ""

    api_key_input = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="Введите ваш ключ...",
        value=default_key,
        help="Вы можете получить ключ на openrouter.ai"
    )
    if api_key_input:
        st.session_state.openrouter_api_key = api_key_input
    
    st.divider()

    uploaded_files = st.file_uploader(
        "Загрузите ваши PDF-файлы",
        type="pdf",
        accept_multiple_files=True
    )
    process_button = st.button("Обработать документы", type="primary")
    st.divider()
    lang_filter_flag = st.checkbox("Language filter ON", value=True, help="Включить приоритизацию кандидатов по языку (SAME_LANG_RATIO)")

    # Переключатель Recall level: переопределяем базовый efSearch на сессию
    recall_options = {
        "Fast (ef=64)": 64,
        "Balanced (ef=128)": 128,
        "High (ef=256)": 256,
        "Max (ef=384)": 384,
    }
    recall_labels = list(recall_options.keys())
    default_recall_label = st.session_state.get("recall_label", "Balanced (ef=128)")
    try:
        default_idx = recall_labels.index(default_recall_label)
    except ValueError:
        default_idx = 1
    recall_label = st.selectbox("Recall level", recall_labels, index=default_idx, help="Контроль скорости/точности: базовый efSearch для HNSW")
    st.session_state.recall_label = recall_label
    rc.HNSW_EF_SEARCH_BASE = int(recall_options.get(recall_label, 128))

# --- Основная логика ---
# Базовая директория проекта (корень), независимо от текущей рабочей директории
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "pdfs"
VECTOR_STORE_PATH = BASE_DIR / "faiss_index"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# --- Инициализация состояния сессии ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Обработка кнопки "Обработать документы" ---
if process_button:
    if not uploaded_files:
        st.error("Пожалуйста, загрузите хотя бы один PDF-файл.")
    elif 'openrouter_api_key' not in st.session_state or not st.session_state.openrouter_api_key:
        st.error("Пожалуйста, введите ваш OpenRouter API ключ в боковой панели.")
    else:
        # Очистка старых файлов и сохранение новых
        for file_path in PDF_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
        for uploaded_file in uploaded_files:
            with open(PDF_DIR / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        with st.spinner("Создание базы знаний... Это может занять время."):
            try:
                build_and_load_knowledge_base(
                    pdf_dir=PDF_DIR,
                    index_dir=VECTOR_STORE_PATH,
                    force_rebuild=True
                )
                st.session_state.rag_chain = create_rag_chain(st.session_state.openrouter_api_key)
                st.session_state.messages = []
                st.success("База знаний успешно создана!")
            except Exception as e:
                st.error(f"Произошла ошибка при обработке документов: {e}")
                if VECTOR_STORE_PATH.exists():
                    shutil.rmtree(VECTOR_STORE_PATH)

# --- Загрузка базы знаний при первом запуске, если она уже есть ---
if st.session_state.rag_chain is None and 'openrouter_api_key' in st.session_state:
    try:
        if build_and_load_knowledge_base(
            pdf_dir=PDF_DIR,
            index_dir=VECTOR_STORE_PATH,
            force_rebuild=False
        ):
            st.session_state.rag_chain = create_rag_chain(st.session_state.openrouter_api_key)
            st.toast("✅ Существующая база знаний загружена.", icon="📚")
    except Exception as e:
        st.warning(f"Не удалось загрузить базу знаний: {e}. Пожалуйста, обработайте файлы заново.")

# --- Отображение чата и обработка запросов ---
st.header("Диалог")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
             with st.expander("Показать контекст, переданный в LLM"):
                st.text(message["context"])

if prompt := st.chat_input("Ваш вопрос..."):
    if st.session_state.rag_chain is None:
        st.warning("Пожалуйста, сначала обработайте документы или убедитесь, что API ключ введен.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Идет гибридный поиск, RRF и реранк..."):
                try:
                    # Пробрасываем флаг в цепочку через глобальные настройки (параметр функции)
                    result = hybrid_search_with_rerank(prompt, apply_lang_quota=bool(lang_filter_flag))
                    fused = result.get("fused", [])
                    reranked = result.get("reranked", [])
                    q_lang = result.get("q_lang")
                    active_branches = result.get("active_branches")

                    # Краткая сводка
                    st.markdown(f"**Сводка:** {{'q_lang': '{q_lang}', 'active_branches': {active_branches}, 'fused': {len(fused)}, 'reranked': {len(reranked)}}}")

                    # Вспомогательная функция построения якоря для будущего просмотра PDF
                    def _make_anchor(item: dict) -> str:
                        """Строит якорь вида ?file={source}&page={page} для навигации."""
                        src = item.get("source")
                        pg = item.get("page")
                        if src and isinstance(pg, int):
                            return f"?file={src}&page={pg}"
                        return ""

                    # Блок RRF-слияние (top-20)
                    with st.expander("RRF-слияние (top-20)"):
                        rows_fused = [
                            {
                                "fusion_score": it.get("fusion_score"),
                                "min_rank": it.get("min_rank"),
                                "hits": it.get("hits"),
                                "source": it.get("source"),
                                "page": it.get("page"),
                                "citation": it.get("citation"),
                                "anchor": _make_anchor(it),
                            }
                            for it in fused[:20]
                        ]
                        st.dataframe(rows_fused, use_container_width=True)

                    # Блок Результаты реранка (top-5)
                    with st.expander("Результаты реранка (top-5)"):
                        rows_rerank = [
                            {
                                "rerank_score": it.get("rerank_score"),
                                "source": it.get("source"),
                                "page": it.get("page"),
                                "citation": it.get("citation"),
                                "retrieval_hits": it.get("hits"),
                                "anchor": _make_anchor(it),
                            }
                            for it in reranked[:5]
                        ]
                        st.dataframe(rows_rerank, use_container_width=True)

                    # Отладка efSearch: показываем применённые значения (если включено)
                    with st.expander("Диагностика efSearch"):
                        st.write(f"Базовый efSearch (сессия): {rc.HNSW_EF_SEARCH_BASE}")
                        st.caption("Применённый ef для каждой dense-ветки логируется при RAG_DEBUG=1 в серверный лог.")

                    # Показать языковые доли по веткам
                    with st.expander("Языковые доли по веткам"):
                        if active_branches:
                            from rag_core import chunks_metadata
                            def _share(items, target_lang):
                                if not items:
                                    return (0, 0, 0.0)
                                same = 0
                                for it in items:
                                    cid = it.get("chunk_id")
                                    if isinstance(cid, int) and 0 <= cid < len(chunks_metadata):
                                        if chunks_metadata[cid].get("lang") == target_lang:
                                            same += 1
                                other = len(items) - same
                                ratio = (same / len(items)) if len(items) else 0.0
                                return (same, other, ratio)
                        
                        # Используем reranked как компактное представление top-N
                        branch_map = {
                            "dense_en": ('en', [it for it in fused if 'dense_en' in (it.get('hits') or [])]),
                            "bm25_en": ('en', [it for it in fused if 'bm25_en' in (it.get('hits') or [])]),
                            "dense_ru": ('ru', [it for it in fused if 'dense_ru' in (it.get('hits') or [])]),
                            "bm25_ru": ('ru', [it for it in fused if 'bm25_ru' in (it.get('hits') or [])]),
                        }
                        for br in active_branches:
                            target_lang, items = branch_map.get(br, (None, []))
                            if not target_lang:
                                continue
                            same, other, share = _share(items, target_lang)
                            st.write(f"{br}: same_lang={same}, other_lang={other}, share={share:.2f}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Сводка: {{'fused': {len(fused)}, 'reranked': {len(reranked)}}}",
                    })
                except Exception as e:
                    st.error(f"Произошла ошибка при поиске/слиянии/реранке: {e}")