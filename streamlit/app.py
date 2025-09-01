# streamlit/app.py
# streamlit/app.py
import streamlit as st
import os
from pathlib import Path
import shutil
# Импортируем обновленные функции
from rag_core import build_and_load_knowledge_base, create_rag_chain, _load_api_key_from_env, collect_candidates_ru_en

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
            with st.spinner("Идет поиск кандидатов (RU/EN)..."):
                try:
                    # Получаем кандидатов из четырех веток
                    cands = collect_candidates_ru_en(prompt)

                    # Краткая сводка по количеству
                    summary_counts = {k: len(v) for k, v in cands.items()}
                    st.markdown(f"**Сводка:** {summary_counts}")

                    # Диагностика: показываем первые 10 кандидатов каждой ветки
                    with st.expander("Диагностика ретрива: топ-10 кандидатов по каждой ветке"):
                        for branch_key, branch_title in [
                            ("dense_ru", "dense_ru"),
                            ("bm25_ru", "bm25_ru"),
                            ("dense_en", "dense_en"),
                            ("bm25_en", "bm25_en"),
                        ]:
                            top_items = cands.get(branch_key, [])[:10]
                            if not top_items:
                                st.markdown(f"_{branch_title}: пусто_")
                                continue
                            rows = [
                                {
                                    "retrieval": it.get("retrieval"),
                                    "rank": it.get("rank"),
                                    "score_raw": it.get("score_raw"),
                                    "source": it.get("source"),
                                    "page": it.get("page"),
                                }
                                for it in top_items
                            ]
                            st.markdown(f"**{branch_title}**")
                            st.dataframe(rows, use_container_width=True)

                    # Сохраняем краткий ответ в историю чата (без реранка и генерации)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Диагностика выполнена. Сводка: {summary_counts}",
                    })
                except Exception as e:
                    st.error(f"Произошла ошибка при поиске кандидатов: {e}")