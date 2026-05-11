# AsistIA — Asesor Experto en IA Generativa

**Práctica Final · Módulo de IA Generativa · Máster en Data Science e IA**

Agente conversacional construido con Google Gemini, ChromaDB, LangGraph y LangChain, que responde preguntas sobre Inteligencia Artificial Generativa apoyándose en una base de conocimiento vectorial propia y manteniendo memoria entre turnos.

> 🚀 **[Probar la demo en vivo](#)** *(pendiente de desplegar en Streamlit Cloud)*

---

## 1. Dominio elegido

El agente, llamado **AsistIA**, es un asesor experto en **Inteligencia Artificial Generativa**. Cubre cuatro áreas:

- Ingeniería de prompts (zero-shot, few-shot, Chain of Thought, ReAct, role prompting).
- Modelos de lenguaje actuales (GPT-4o, Claude 3.5, Gemini 2.5, LLaMA 3, Mistral) y criterios de selección.
- Pipelines RAG (chunking, embeddings, vectorstores, similitud).
- Agentes y arquitecturas (LangChain, LangGraph, MemorySaver, agentes ReAct).

Se eligió este dominio porque es el propio contenido del módulo y permite construir una base de conocimiento densa con material de calidad. Además, evidencia el valor del RAG: aunque Gemini ya conoce el tema en general, la base vectorial permite acotar las respuestas a un cuerpo concreto de documentos y dar contestaciones reproducibles y trazables a sus fuentes.

---

## 2. Estructura del repositorio

```
EVOLVE_RAG/
├── asesor_ia.ipynb         Notebook principal (agente + demo)
├── streamlit_app.py        Interfaz web (entregable bonus)
├── requirements.txt        Dependencias
├── .env.example            Plantilla de variables de entorno
├── .gitignore
├── README.md
├── docs/                   Base de conocimiento (3 PDFs, ~20 páginas)
│   ├── 01_ingenieria_de_prompts.pdf
│   ├── 02_modelos_ia_generativa.pdf
│   └── 03_rag_agentes_arquitecturas.pdf
└── chroma_db/              Vectorstore persistente (se crea en la 1ª ejecución)
```

---

## 3. Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Google Gemini 2.5 Flash Lite (`gemini-2.5-flash-lite`) |
| Embeddings | `models/gemini-embedding-001` |
| Base vectorial | ChromaDB con persistencia en disco |
| Framework de agente | LangGraph (sobre LangChain) |
| Memoria | `MemorySaver` de LangGraph (por `thread_id`) |
| Entorno | Jupyter Notebook |
| Interfaz web (bonus) | Streamlit |

---

## 4. Instalación y ejecución

### Requisitos

- Python 3.10 o superior.
- Una API key de Gemini ([Google AI Studio](https://aistudio.google.com/app/apikey)).

### Pasos

1. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Configurar la API key. Copiar `.env.example` a `.env` y editar la clave:

   ```bash
   cp .env.example .env
   ```

   Contenido de `.env`:
   ```
   GEMINI_API_KEY=tu_clave_real
   ```

3. Abrir y ejecutar el notebook:

   ```bash
   jupyter notebook asesor_ia.ipynb
   ```

4. (Opcional) Ejecutar la interfaz Streamlit:

   ```bash
   streamlit run streamlit_app.py
   ```

> **Nota sobre el archivo `.env`:** el nombre debe empezar exactamente por un punto. En Windows, el explorador puede tratar los archivos que empiezan por `.` como ocultos; conviene crear el archivo desde la terminal o asegurarse de que el nombre real es `.env` y no `.env.txt`.

---

## 5. Base de conocimiento

| Documento | Contenido |
|---|---|
| `01_ingenieria_de_prompts.pdf` | Zero-shot, few-shot, Chain of Thought, ReAct, role prompting, buenas prácticas, seguridad |
| `02_modelos_ia_generativa.pdf` | GPT-4o, Claude 3.5, Gemini 2.5, LLaMA 3, Mistral, criterios de selección, embeddings |
| `03_rag_agentes_arquitecturas.pdf` | Pipeline RAG, ChromaDB, chunking, agentes, LangChain, LangGraph, MemorySaver |

**Procesado:**

- Chunking con `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=80, separadores `\n\n`, `\n`, `. `, ` `).
- Embeddings con `models/gemini-embedding-001`.
- Indexado en ChromaDB con persistencia en `./chroma_db/` y nombre de colección `asesor_ia_generativa`.
- Recuperación por similitud coseno, top-3 chunks por consulta.

---

## 6. System prompt y justificación de decisiones

El system prompt usado es el siguiente:

```
Eres AsistIA, un asesor experto en Inteligencia Artificial Generativa.
Tu especialidad abarca: ingeniería de prompts, modelos de lenguaje (LLMs),
técnicas RAG, agentes de IA, frameworks como LangChain y LangGraph, y
buenas prácticas del ecosistema GenAI.

INSTRUCCIONES:
1. Basa SIEMPRE tus respuestas en el contexto recuperado de la base de
   conocimiento. Si el contexto es relevante, úsalo como fuente principal.
2. Si no encuentras información suficiente en el contexto, indícalo
   claramente con: "Esta pregunta va más allá de mi base de conocimiento
   actual, pero puedo decirte que..." y proporciona una respuesta general
   desde tu conocimiento previo.
3. Responde SIEMPRE en español, independientemente del idioma de la pregunta.
4. Estructura tus respuestas de forma clara: usa listas cuando sea útil,
   destaca términos técnicos importantes y proporciona ejemplos concretos.
5. Mantén un tono pedagógico y accesible: eres un experto que enseña, no
   que abruma.
6. Recuerda el contexto de la conversación: si el usuario hace referencia
   a algo que mencionó antes, úsalo para dar respuestas más personalizadas.
7. Al final de respuestas técnicas, ofrece brevemente un siguiente paso
   o pregunta de seguimiento para guiar el aprendizaje.
```

**Justificación de cada decisión:**

- **Rol concreto y especializado.** Un rol genérico ("eres un asistente útil") produce respuestas vagas. Al definir exactamente el dominio de expertise, el modelo calibra su nivel de detalle y terminología.
- **Obligación de basarse en el contexto recuperado (instrucción 1).** Es la regla más importante para un sistema RAG. Sin ella, el modelo tiende a ignorar el contexto y a responder desde su preentrenamiento, anulando el propósito del RAG.
- **Fallback explícito ante falta de contexto (instrucción 2).** En lugar de inventar respuestas, el agente reconoce el límite de su base de conocimiento. Esto reduce las alucinaciones y aumenta la confianza del usuario.
- **Idioma fijado en español (instrucción 3).** Garantiza consistencia independientemente del idioma de la pregunta.
- **Estructura y ejemplos concretos (instrucción 4).** Mejora la legibilidad y la utilidad pedagógica de las respuestas.
- **Tono pedagógico (instrucción 5).** El dominio (enseñar IA) demanda un estilo que explique, no que abrume con jerga.
- **Memoria explícita (instrucción 6).** Activa el uso del historial conversacional. Sin esta indicación, el modelo no siempre conecta respuestas anteriores con la pregunta actual.
- **Siguiente paso de aprendizaje (instrucción 7).** Convierte el chat en una herramienta de aprendizaje guiado en lugar de un mero buscador.

**Temperatura del modelo:** `0.3`. Es un valor bajo que prioriza precisión y consistencia (importante para respuestas técnicas) sin caer en la rigidez excesiva del `0.0`.

---

## 7. Arquitectura del agente

```
                ┌──────────────────────────────────┐
                │                                  │
   START ──▶  recuperar  ──▶  generar  ──▶  END   │
                │                  │              │
                ▼                  ▼              │
      Recupera top-3 chunks   Inyecta contexto    │
      desde ChromaDB usando   en system prompt    │
      la última HumanMessage  + historial         │
      del estado.             completo. Llama     │
                              a Gemini.           │
                                                  │
   Estado (TypedDict):                            │
     - mensajes: Annotated[list, add_messages]   ◀┘
       (el reducer add_messages ACUMULA mensajes
        en lugar de reemplazarlos → memoria)
     - contexto_rag: str
       (se sobrescribe en cada turno)

   Memoria: MemorySaver con thread_id
            (cada thread_id = conversación aislada)
```

---

## 8. Ejemplos demostrados en el notebook

El notebook incluye 5 preguntas de ejemplo más una prueba explícita de memoria, ejecutadas todas en threads aislados:

| # | Tema | Thread |
|---|---|---|
| 1 | Zero-shot prompting | `demo-completa` |
| 2 | Comparativa GPT-4o vs Claude 3.5 | `demo-completa` |
| 3 | Pipeline RAG y chunking | `demo-completa` |
| 4 | LangGraph vs LangChain | `demo-completa` |
| 5 | Recomendación de modelo según caso de uso | `demo-completa` |
| 6 | **Prueba de memoria** (3 turnos: contexto → pregunta intermedia → pregunta que referencia el contexto inicial) | `demo-memoria` |

Adicionalmente, la sección 7 del notebook contiene una celda `chat_interactivo()` para conversar libremente con el agente.

---

## 9. Dependencias

`requirements.txt`:

```
langchain>=0.3
langchain-google-genai>=2.0
langchain-community>=0.3
langchain-text-splitters>=0.3
langgraph>=0.2
chromadb>=0.5
pypdf>=4.0
python-dotenv>=1.0
grandalf>=0.8           # visualización ASCII del grafo
streamlit>=1.35         # bonus: interfaz web
```

---

## 10. Bonus: interfaz Streamlit

`streamlit_app.py` ofrece la misma funcionalidad que el notebook con una interfaz web: chat con historial, selección de modelo (Flash Lite / Flash / Pro), panel de inspección del último contexto RAG, contadores de sesión y preguntas de ejemplo.

Ejecución local:

```bash
streamlit run streamlit_app.py
```

Para desplegar en Streamlit Cloud, basta con subir el repositorio (sin el `.env`) y configurar `GEMINI_API_KEY` en *Advanced settings → Secrets*:

```toml
GEMINI_API_KEY = "tu_clave"
```

---

## 11. Buenas prácticas aplicadas

- API key fuera del código, en `.env` (excluido por `.gitignore`).
- Chunking con solapamiento para evitar pérdida de contexto entre fragmentos.
- `MemorySaver` con `thread_id` para aislar conversaciones.
- ChromaDB con persistencia en disco (no se reindexa en cada ejecución).
- Temperatura baja (0.3) para respuestas técnicas precisas.
- Fallback explícito en el system prompt cuando el contexto es insuficiente.
- Caché de recursos en Streamlit (`@st.cache_resource`) para no recompilar el agente en cada rerun.
- Búsqueda robusta del `.env` con `find_dotenv()` (independiente del directorio desde el que se lance Jupyter).

---

## Autor

**Alejandro Rodríguez de la Rosa**
Marketing · Data Science · IA
📧 alejandrorodriguezdelarosa@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/alejandro-rodríguez-de-la-rosa-015956301)
