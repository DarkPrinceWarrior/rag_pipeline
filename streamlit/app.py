# streamlit/app.py
import streamlit as st
import os
from pathlib import Path
import shutil
# Важно: Нам нужно получить не только ответ, но и контекст для отладки
from rag_core import get_or_create_vector_store, create_rag_chain

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

if process_button:
    # ... (логика обработки кнопки остается без изменений) ...
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
        
        with st.spinner("Идёт обработка документов... Это может занять некоторое время."):
            try:
                index, metadata, embedder = get_or_create_vector_store(
                    pdf_dir=str(PDF_DIR),
                    index_dir=str(VECTOR_STORE_PATH),
                    force_rebuild=True
                )
                st.session_state.rag_chain = create_rag_chain(
                    index=index,
                    metadata=metadata,
                    embedder=embedder,
                    openrouter_api_key=st.session_state.openrouter_api_key
                )
                # Инициализируем историю чата после успешной обработки
                st.session_state.messages = []
                st.success("Документы успешно обработаны! Теперь вы можете задавать вопросы.")
            except Exception as e:
                st.error(f"Произошла ошибка при обработке документов: {e}")
                if VECTOR_STORE_PATH.exists():
                    shutil.rmtree(VECTOR_STORE_PATH)

# --- Отображение чата ---
st.header("Диалог")

# Инициализация истории чата, если она еще не создана
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение предыдущих сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Отображаем отладочную информацию, если она есть
        if "context" in message and message["context"]:
             with st.expander("Показать контекст, переданный в LLM"):
                st.text(message["context"])


# Поле для ввода нового вопроса
if prompt := st.chat_input("Ваш вопрос..."):
    if 'rag_chain' not in st.session_state:
        st.warning("Пожалуйста, сначала обработайте документы, используя форму в боковой панели.")
    else:
        # 1. Отобразить и сохранить вопрос пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Получить и отобразить ответ модели
        with st.chat_message("assistant"):
            with st.spinner("Думаю над ответом..."):
                try:
                    rag_chain = st.session_state.rag_chain
                    # Получаем ответ и контекст
                    answer, context = rag_chain["answer_question"](prompt)
                    
                    response_placeholder = st.empty()
                    response_placeholder.markdown(answer)

                    # Показываем отладочную информацию
                    with st.expander("Показать контекст, переданный в LLM"):
                        st.text(context)
                    
                    # Сохраняем ответ и контекст в историю
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context # Сохраняем для отладки
                    })

                except Exception as e:
                    st.error(f"Произошла ошибка при получении ответа: {e}")