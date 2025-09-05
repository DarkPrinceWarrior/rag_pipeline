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

    st.divider()
    st.subheader("MMR и контекст")
    mmr_lambda = st.slider("MMR_LAMBDA", min_value=0.2, max_value=0.8, value=float(getattr(rc, "MMR_LAMBDA", 0.5)), step=0.05, help="Баланс релевантность/диверсификация при MMR")
    rc.MMR_LAMBDA = float(mmr_lambda)

    per_doc_cap = st.number_input("PER_DOC_CAP", min_value=1, max_value=10, value=int(getattr(rc, "PER_DOC_CAP", 2)), step=1, help="Ограничение на количество кусков с одного источника")
    rc.PER_DOC_CAP = int(per_doc_cap)

    per_page_cap = st.number_input("PER_PAGE_CAP", min_value=0, max_value=10, value=int(getattr(rc, "PER_PAGE_CAP", 1)), step=1, help="Ограничение на количество кусков с одной страницы (0 = нет лимита)")
    rc.PER_PAGE_CAP = int(per_page_cap)

    st.markdown("LANG_MIN_COVER (минимальные квоты)")
    _ru_default = bool((getattr(rc, "LANG_MIN_COVER", {"ru":1,"en":1}).get("ru", 1)) > 0)
    _en_default = bool((getattr(rc, "LANG_MIN_COVER", {"ru":1,"en":1}).get("en", 1)) > 0)
    lang_ru_checked = st.checkbox("RU", value=_ru_default)
    lang_en_checked = st.checkbox("EN", value=_en_default)
    rc.LANG_MIN_COVER = {"ru": (1 if lang_ru_checked else 0), "en": (1 if lang_en_checked else 0)}

    enable_mmr = st.checkbox("Enable MMR", value=st.session_state.get("enable_mmr", True), help="Отключить MMR для A/B сравнения с top-K после реранка")
    st.session_state.enable_mmr = bool(enable_mmr)

    st.divider()
    st.subheader("Генерация")
    model_options_str = os.getenv("RAG_UI_MODEL_OPTIONS", "")
    default_model_value = getattr(rc, "OPENROUTER_MODEL", "openai/gpt-oss-120b")
    model_options = [s.strip() for s in model_options_str.split(",") if s.strip()] or [default_model_value]
    default_model_sel = st.session_state.get("llm_model", default_model_value)
    if default_model_sel not in model_options:
        model_options = [default_model_sel] + [m for m in model_options if m != default_model_sel]
    llm_model = st.selectbox("LLM_MODEL", model_options, index=0)
    st.session_state.llm_model = llm_model

    llm_max_tokens = st.number_input(
        "LLM_MAX_TOKENS",
        min_value=200,
        max_value=4000,
        value=int(st.session_state.get("llm_max_tokens", int(getattr(rc, "LLM_DEFAULT_MAX_TOKENS", 550)))),
        step=50,
    )
    st.session_state.llm_max_tokens = int(llm_max_tokens)

    llm_temperature = st.slider(
        "LLM_TEMPERATURE",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("llm_temperature", float(getattr(rc, "LLM_DEFAULT_TEMPERATURE", 0.2)))),
        step=0.05,
    )
    st.session_state.llm_temperature = float(llm_temperature)

    enforce_citations = st.checkbox(
        "Enforce citations",
        value=bool(st.session_state.get("enforce_citations", True)),
        help="Принудительная проверка наличия [S#] с однократной перегенерацией",
    )
    st.session_state.enforce_citations = bool(enforce_citations)

    language_enforcement = st.checkbox(
        "Language enforcement",
        value=bool(st.session_state.get("language_enforcement", True)),
        help="Жёсткий контроль языка ответа и перегенерация при нарушении",
    )
    st.session_state.language_enforcement = bool(language_enforcement)

# --- Основная логика ---
# Базовая директория проекта (корень), независимо от текущей рабочей директории
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "pdfs"
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
                    index_dir=rc.VECTOR_STORE_PATH,
                    force_rebuild=True
                )
                st.session_state.rag_chain = create_rag_chain(
                    st.session_state.openrouter_api_key,
                    openrouter_model=st.session_state.get("llm_model", getattr(rc, "OPENROUTER_MODEL", "openai/gpt-oss-120b")),
                    temperature=st.session_state.get("llm_temperature", float(getattr(rc, "LLM_DEFAULT_TEMPERATURE", 0.2))),
                    max_tokens=st.session_state.get("llm_max_tokens", int(getattr(rc, "LLM_DEFAULT_MAX_TOKENS", 550))),
                    enforce_citations=st.session_state.get("enforce_citations", True),
                    language_enforcement=st.session_state.get("language_enforcement", True),
                )
                st.session_state.messages = []
                st.success("База знаний успешно создана!")
            except Exception as e:
                st.error(f"Произошла ошибка при обработке документов: {e}")
                if Path(rc.VECTOR_STORE_PATH).exists():
                    shutil.rmtree(Path(rc.VECTOR_STORE_PATH))

# --- Загрузка базы знаний при первом запуске, если она уже есть ---
if st.session_state.rag_chain is None and 'openrouter_api_key' in st.session_state:
    try:
        if build_and_load_knowledge_base(
            pdf_dir=PDF_DIR,
            index_dir=rc.VECTOR_STORE_PATH,
            force_rebuild=False
        ):
            st.session_state.rag_chain = create_rag_chain(
                st.session_state.openrouter_api_key,
                openrouter_model=st.session_state.get("llm_model", getattr(rc, "OPENROUTER_MODEL", "openai/gpt-oss-120b")),
                temperature=st.session_state.get("llm_temperature", float(getattr(rc, "LLM_DEFAULT_TEMPERATURE", 0.2))),
                max_tokens=st.session_state.get("llm_max_tokens", int(getattr(rc, "LLM_DEFAULT_MAX_TOKENS", 550))),
                enforce_citations=st.session_state.get("enforce_citations", True),
                language_enforcement=st.session_state.get("language_enforcement", True),
            )
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
            with st.spinner("Идёт гибридный поиск, реранк и генерация ответа..."):
                try:
                    # Пробрасываем флаг в цепочку через глобальные настройки (параметр функции)
                    result = hybrid_search_with_rerank(prompt, apply_lang_quota=bool(lang_filter_flag))
                    fused = result.get("fused", [])
                    reranked = result.get("reranked", [])
                    context_pack = result.get("context_pack", [])
                    context_stats = result.get("context_stats", {})
                    q_lang = result.get("q_lang")
                    active_branches = result.get("active_branches")
                    sources_map = result.get("sources_map", {}) or {}

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
                        st.dataframe(rows_fused, width='stretch')

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
                        st.dataframe(rows_rerank, width='stretch')

                    # Панель Context Pack (после MMR)
                    with st.expander("Context Pack (после MMR)", expanded=True):
                        # Таблица: source | page | citation | lang | reason_score | mmr_gain | dup_flags | tokens_est
                        def _tokens_est(meta_text: str | None, lang: str | None) -> int:
                            try:
                                from rag_core import _estimate_tokens
                                return int(_estimate_tokens(meta_text, lang))
                            except Exception:
                                return 0
                        from rag_core import chunks_metadata as _meta
                        rows_ctx = []
                        seen = set()
                        for it in context_pack:
                            cid = it.get("chunk_id")
                            if isinstance(cid, int) and 0 <= cid < len(_meta) and cid not in seen:
                                m = _meta[cid]
                                lang = m.get("lang")
                                anchor = _make_anchor({"source": m.get("source"), "page": m.get("page")})
                                rows_ctx.append({
                                    "source": m.get("source"),
                                    "page": m.get("page"),
                                    "citation": it.get("citation"),
                                    "lang": lang,
                                    "reason_score": it.get("rerank_score"),
                                    "mmr_gain": {
                                        "rel": it.get("mmr_rel"),
                                        "div": it.get("mmr_div"),
                                        "score": it.get("mmr_score"),
                                    },
                                    "dup_flags": None,
                                    "tokens_est": _tokens_est(m.get("text"), lang),
                                    "anchor": anchor,
                                })
                                seen.add(cid)
                        st.dataframe(rows_ctx, width='stretch')

                        # Сводка
                        k = len(context_pack)
                        n = len(reranked)
                        budget_used = int(context_stats.get("budget_used_tokens", 0))
                        budget_limit = int(context_stats.get("budget_limit", 0))
                        lang_dist = context_stats.get("lang_distribution", {}) or {}
                        docs_dist = context_stats.get("doc_distribution", {}) or {}
                        st.markdown(f"Взято {k} из {n}; бюджет {budget_used}/{budget_limit}")
                        st.markdown(f"Языки: {lang_dist}")
                        st.markdown(f"Источники: {docs_dist}")

                        # Пороговые значения и режимы ослабления
                        thresholds = context_stats.get("thresholds", {}) or {}
                        with st.expander("Пороговые значения и режимы ослабления"):
                            st.json(thresholds)

                        # Отфильтрованные кандидаты (rejected reasons)
                        rejected_reasons = context_stats.get("rejected_reasons", {}) or {}
                        with st.expander("Отфильтрованные кандидаты"):
                            rej_rows = [
                                {"reason": k, "count": v}
                                for k, v in sorted(rejected_reasons.items(), key=lambda x: (-int(x[1]), str(x[0])))
                            ]
                            st.dataframe(rej_rows, width='stretch')

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

                    # Генерация ответа
                    try:
                        gen_fn = st.session_state.rag_chain.get("answer_question") if isinstance(st.session_state.rag_chain, dict) else None
                        if callable(gen_fn):
                            gen_out = gen_fn(prompt, apply_lang_quota=bool(lang_filter_flag))
                        else:
                            gen_out = {"final_answer": "", "used_sources": [], "answer_lang_detected": None, "flags": {}}
                    except Exception as _e:
                        gen_out = {"final_answer": f"Ошибка генерации ответа: {_e}", "used_sources": [], "answer_lang_detected": None, "flags": {}}

                    st.subheader("Ответ")
                    answer_text = gen_out.get("final_answer") or ""
                    st.markdown(answer_text)

                    used_labels = gen_out.get("used_sources", []) or []
                    from rag_core import chunks_metadata as _meta
                    rows_sources = []
                    for label in used_labels:
                        cid = sources_map.get(label)
                        if isinstance(cid, int) and 0 <= int(cid) < len(_meta):
                            m = _meta[int(cid)]
                            rows_sources.append({
                                "S#": label,
                                "source": m.get("source"),
                                "page": m.get("page"),
                                "anchor": _make_anchor({"source": m.get("source"), "page": m.get("page")}),
                            })
                    with st.expander("Список источников", expanded=bool(rows_sources)):
                        st.dataframe(rows_sources, width='stretch')

                    flags = gen_out.get("flags", {}) or {}
                    lang_detected = gen_out.get("answer_lang_detected")
                    st.caption(f"Язык ответа: {lang_detected or 'unk'} | Перегенерации: lang={bool(flags.get('regenerated_for_lang'))}, citations={bool(flags.get('regenerated_for_citations'))}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Сводка: {{'fused': {len(fused)}, 'reranked': {len(reranked)}}}",
                    })
                except Exception as e:
                    st.error(f"Произошла ошибка при поиске/слиянии/реранке: {e}")