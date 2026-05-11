"""
AsistIA — Asesor Experto en IA Generativa
Proyecto Final · Módulo de IA Generativa

Interfaz web Streamlit para interactuar con el agente RAG construido en el
notebook asesor_ia.ipynb.

Ejecución local:
    streamlit run streamlit_app.py

Requiere la variable de entorno GEMINI_API_KEY (en .env o, en Streamlit Cloud,
en Settings → Secrets).
"""

import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Cargar .env desde la carpeta del proyecto (busca recursivamente hacia arriba).
load_dotenv(find_dotenv(usecwd=True))


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AsistIA — Asesor en IA Generativa",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    .stat-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        margin: 4px 0;
    }
    .stat-number { font-size: 1.8rem; font-weight: bold; color: #4ade80; }
    .stat-label  { color: #8b8fa8; font-size: 0.8rem; margin-top: 2px; }

    .role-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
        background: #1a2a4a; color: #60a5fa; border: 1px solid #2d5aa0;
        margin-bottom: 6px;
    }

    .rag-panel {
        background: #0f1923;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.82rem;
        color: #7dd3fc;
        font-family: monospace;
        max-height: 200px;
        overflow-y: auto;
        margin-top: 8px;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],          # Historial visible en la UI
        "total_calls": 0,
        "thread_id": f"streamlit-{uuid.uuid4().hex[:8]}",
        "agente": None,
        "retriever": None,
        "ultimo_contexto_rag": "",
        "api_key_ok": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (idéntico al del notebook)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres AsistIA, un asesor experto en Inteligencia Artificial Generativa.
Tu especialidad abarca: ingeniería de prompts, modelos de lenguaje (LLMs), técnicas RAG,
agentes de IA, frameworks como LangChain y LangGraph, y buenas prácticas del ecosistema GenAI.

INSTRUCCIONES:
1. Basa SIEMPRE tus respuestas en el contexto recuperado de la base de conocimiento.
   Si el contexto es relevante, úsalo como fuente principal.
2. Si no encuentras información suficiente en el contexto, indícalo claramente con:
   "Esta pregunta va más allá de mi base de conocimiento actual, pero puedo decirte que..."
   y proporciona una respuesta general desde tu conocimiento previo.
3. Responde SIEMPRE en español, independientemente del idioma de la pregunta.
4. Estructura tus respuestas de forma clara: usa listas cuando sea útil, destaca términos
   técnicos importantes y proporciona ejemplos concretos cuando sea posible.
5. Mantén un tono pedagógico y accesible: eres un experto que enseña, no que abruma.
6. Recuerda el contexto de la conversación: si el usuario hace referencia a algo que
   mencionó antes, úsalo para dar respuestas más personalizadas.
7. Al final de respuestas técnicas, ofrece brevemente un siguiente paso o pregunta
   de seguimiento para guiar el aprendizaje."""


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL AGENTE (cacheada para no recompilar en cada rerun de Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def cargar_agente(api_key: str, modelo: str):
    """Construye el vectorstore, el LLM y el grafo LangGraph.

    Retorna: (agente_compilado, retriever, n_chunks_indexados).
    """
    from typing import Annotated, TypedDict

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    DOCS_DIR = "docs"
    CHROMA_DIR = "chroma_db"
    pdfs = [
        os.path.join(DOCS_DIR, "01_ingenieria_de_prompts.pdf"),
        os.path.join(DOCS_DIR, "02_modelos_ia_generativa.pdf"),
        os.path.join(DOCS_DIR, "03_rag_agentes_arquitecturas.pdf"),
    ]

    for p in pdfs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"No se encontró el documento: {p}")

    # Carga + chunking
    todos = []
    for pdf_path in pdfs:
        todos.extend(PyPDFLoader(pdf_path).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(todos)

    # Embeddings + ChromaDB persistente
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="asesor_ia_generativa",
        persist_directory=CHROMA_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatGoogleGenerativeAI(
        model=modelo,
        temperature=0.3,
        google_api_key=api_key,
    )

    # Estado del grafo
    class EstadoAsesorIA(TypedDict):
        mensajes: Annotated[list[BaseMessage], add_messages]
        contexto_rag: str

    # Nodo 1: recuperación RAG
    def nodo_rag(estado):
        ultimo = None
        for msg in reversed(estado["mensajes"]):
            if isinstance(msg, HumanMessage):
                ultimo = msg.content
                break
        if not ultimo:
            return {"contexto_rag": ""}
        docs = retriever.invoke(ultimo)
        if not docs:
            return {"contexto_rag": "No se encontró información en la base de conocimiento."}
        fragmentos = []
        for i, doc in enumerate(docs, 1):
            fuente = os.path.basename(doc.metadata.get("source", "doc"))
            pag = doc.metadata.get("page", "?")
            fragmentos.append(f"[Fuente {i}: {fuente}, pág. {pag}]\n{doc.page_content}")
        return {"contexto_rag": "\n\n".join(fragmentos)}

    # Nodo 2: generación con Gemini
    def nodo_generacion(estado):
        contexto = estado.get("contexto_rag", "")
        sys_prompt = SYSTEM_PROMPT
        if contexto:
            sys_prompt += (
                f"\n\nCONTEXTO DE LA BASE DE CONOCIMIENTO:\n{'=' * 50}\n"
                f"{contexto}\n{'=' * 50}\n"
                "Usa este contexto como referencia principal."
            )
        mensajes_completos = [SystemMessage(content=sys_prompt)] + estado["mensajes"]
        respuesta = llm.invoke(mensajes_completos)
        return {"mensajes": [respuesta]}

    grafo = StateGraph(EstadoAsesorIA)
    grafo.add_node("recuperar", nodo_rag)
    grafo.add_node("generar", nodo_generacion)
    grafo.add_edge(START, "recuperar")
    grafo.add_edge("recuperar", "generar")
    grafo.add_edge("generar", END)

    agente = grafo.compile(checkpointer=MemorySaver())
    return agente, retriever, len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuración")
    st.divider()

    # API Key — admite valor del .env / secrets como pre-relleno
    api_key_input = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="AIza...",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Obtén una clave gratuita en https://aistudio.google.com/app/apikey",
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

    st.divider()
    st.subheader("🤖 Modelo")
    modelo = st.selectbox(
        "Modelo Gemini",
        [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        index=0,
        help="Flash Lite: rápido y económico · Pro: máxima calidad",
    )

    st.divider()
    st.subheader("📊 Sesión")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-number'>{st.session_state.total_calls}</div>"
            f"<div class='stat-label'>Consultas</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-number'>{len(st.session_state.messages)}</div>"
            f"<div class='stat-label'>Mensajes</div></div>",
            unsafe_allow_html=True,
        )

    st.caption(f"🔑 Thread: `{st.session_state.thread_id[-8:]}`")

    st.divider()
    if st.button("🗑️ Nueva conversación", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.total_calls = 0
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        st.session_state.ultimo_contexto_rag = ""
        st.rerun()

    st.divider()
    st.subheader("💡 Preguntas de ejemplo")
    ejemplos = [
        "¿Qué es few-shot prompting?",
        "Compara GPT-4o con Claude 3.5",
        "¿Cómo funciona ChromaDB?",
        "¿Qué es LangGraph?",
        "Recomiéndame un modelo económico",
    ]
    for ejemplo in ejemplos:
        if st.button(f"→ {ejemplo}", use_container_width=True, key=f"ej_{ejemplo}"):
            st.session_state["_pregunta_rapida"] = ejemplo
            st.rerun()

    with st.expander("🔍 Último contexto RAG"):
        if st.session_state.ultimo_contexto_rag:
            st.markdown(
                f"<div class='rag-panel'>{st.session_state.ultimo_contexto_rag[:800]}...</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Sin consultas aún.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Cabecera
# ─────────────────────────────────────────────────────────────────────────────

st.title("🤖 AsistIA — Asesor Experto en IA Generativa")
st.caption(
    "Proyecto Final · IA Generativa · RAG + Gemini + LangGraph  |  "
    "Base de conocimiento: ingeniería de prompts · modelos LLM · agentes y arquitecturas"
)

st.markdown(
    "<div style='background:#1a1f35;border:1px solid #3d4166;border-radius:10px;"
    "padding:10px 16px;margin-bottom:16px;'>"
    "<span class='role-badge'>🎭 Rol activo</span><br>"
    "<span style='color:#c8cadd;font-size:0.85rem;'>"
    "Asesor experto en IA Generativa · Responde en español · Basado en documentos indexados"
    "</span></div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN DEL AGENTE
# ─────────────────────────────────────────────────────────────────────────────

api_key = os.getenv("GEMINI_API_KEY", "")

if not api_key:
    st.warning(
        "⚠️ Configura tu **Gemini API Key** en el panel lateral para empezar. "
        "Se puede obtener una clave gratuita en "
        "[Google AI Studio](https://aistudio.google.com/app/apikey)."
    )
    st.stop()

try:
    with st.spinner("⚙️ Inicializando agente y base de conocimiento..."):
        agente, retriever, n_chunks = cargar_agente(api_key, modelo)
    st.session_state.agente = agente
    st.session_state.retriever = retriever
    st.session_state.api_key_ok = True
except FileNotFoundError as e:
    st.error(f"❌ {e}\n\nLa carpeta `docs/` debe contener los 3 PDFs.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error inicializando el agente: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HISTORIAL DE CHAT
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 **Hola, soy AsistIA**, asesor experto en IA Generativa.\n\n"
            "Puedo ayudarte con:\n"
            "- 🎯 **Ingeniería de prompts**: zero-shot, few-shot, Chain of Thought…\n"
            "- 🤖 **Modelos LLM**: comparativas GPT-4o, Claude, Gemini, LLaMA…\n"
            "- 🔍 **RAG y vectorstores**: ChromaDB, chunking, embeddings…\n"
            "- 🧩 **Agentes**: LangGraph, ReAct, memoria, herramientas…\n\n"
            "¿Sobre qué tema quieres preguntar?"
        )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tiempo"):
                st.caption(f"⏱️ {msg['tiempo']}ms · 🤖 {modelo}")


# ─────────────────────────────────────────────────────────────────────────────
# INPUT Y PROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

pregunta_rapida = st.session_state.pop("_pregunta_rapida", None)
prompt = st.chat_input("Escribe tu pregunta sobre IA Generativa...") or pregunta_rapida

if prompt:
    from langchain_core.messages import HumanMessage as HMsg

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Buscando en la base de conocimiento y generando respuesta..."):
            try:
                inicio = datetime.now()

                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                entrada = {"mensajes": [HMsg(content=prompt)]}
                resultado = agente.invoke(entrada, config=config)

                tiempo_ms = int((datetime.now() - inicio).total_seconds() * 1000)
                respuesta = resultado["mensajes"][-1].content
                contexto = resultado.get("contexto_rag", "")

                st.session_state.ultimo_contexto_rag = contexto

                st.markdown(respuesta)
                st.caption(
                    f"⏱️ {tiempo_ms}ms · 🤖 {modelo} · 📄 {n_chunks} chunks indexados"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": respuesta,
                    "tiempo": tiempo_ms,
                })
                st.session_state.total_calls += 1

            except Exception as e:
                st.error(f"❌ Error al generar respuesta: {e}")
