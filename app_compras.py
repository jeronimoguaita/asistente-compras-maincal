# -*- coding: utf-8 -*-
"""
Interfaz web (Streamlit) para el Asistente de Compras Maincal (LLM + RAG).

Como ejecutar (local):
    1) python -m pip install streamlit openai pypdf pandas openpyxl scikit-learn numpy
    2) streamlit run app_compras.py
    3) Se abre en el navegador. Pega tu API key en la barra lateral.

Archivos necesarios en la misma carpeta:
    - Cerco_informacion_Maincal.pdf
    - BOM_Cronos-N04.xlsx
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

PDF_PATH = "Cerco_informacion_Maincal.pdf"
BOM_PATH = "BOM_Cronos-N04.xlsx"
MODELO_EMBEDDING = "text-embedding-3-small"
MODELO_CHAT = "gpt-4o-mini"
TOP_K = 3
TEMPERATURA = 0.1
MAX_TOKENS = 800

SYSTEM_PROMPT = """
Eres un asistente de compras de la empresa Maincal (fabrica de calzado de seguridad industrial).
Tu funcion es ayudar a decidir prioridades de compra del producto Cronos-N04, basandote
UNICAMENTE en la informacion del contexto (fichas de proveedores + BOM + stock actual).

REGLAS GENERALES:
- Responde solo con la informacion del contexto. Si un dato no esta, responde:
  "No tengo esa informacion disponible". Nunca inventes datos ni supongas valores.
- NO inventes cantidades de orden. Si la pregunta no menciona una cantidad de pares, no asumas ninguna.

COMO EVALUAR PRIORIDAD (cuando se pregunta que insumo priorizar):
- Para cada insumo critico, compara su stock actual contra su stock minimo (de la ficha).
  Si el stock actual esta por debajo del minimo, ese insumo debe priorizarse.
- Entre los que esten por debajo del minimo, ordena por riesgo: mayor lead time,
  proveedor unico, importacion o paralizacion de linea = mas urgente.
- Si el stock actual esta por encima del minimo, no requiere atencion inmediata.

COMO EVALUAR UNA ORDEN PUNTUAL (solo si la pregunta indica una cantidad de pares):
- Calcula necesidad = cantidad_de_pares x consumo_por_par (de la BOM) para cada insumo.
- Compara la necesidad contra el stock actual e indica si alcanza o si hay riesgo de quiebre.

Se claro y concreto, y explica el porque (stock vs minimo, lead time, proveedor unico, importacion, etc.).
"""


# ---------- Carga de datos (cacheada) ----------
@st.cache_data(show_spinner=False)
def cargar_chunks():
    reader = PdfReader(PDF_PATH)
    paginas = [p.extract_text().strip() for p in reader.pages
               if p.extract_text() and p.extract_text().strip()]
    return paginas


@st.cache_data(show_spinner=False)
def cargar_bom():
    df = pd.read_excel(BOM_PATH, header=2)
    return df.dropna(subset=["insumo", "cantidad_por_par"])


@st.cache_data(show_spinner="Generando embeddings (una sola vez)...")
def generar_embeddings(_client, chunks):
    resp = _client.embeddings.create(model=MODELO_EMBEDDING, input=chunks)
    return np.array([d.embedding for d in resp.data])


def construir_datos_numericos(bom_df, stock_actual):
    lineas = ["=== BOM (consumo por par del producto Cronos-N04) ==="]
    for _, fila in bom_df.iterrows():
        crit = " [CRITICO]" if str(fila.get("critico", "")).upper() == "SI" else ""
        lineas.append(f"- {fila['insumo']}: {fila['cantidad_por_par']} {fila['unidad']} por par{crit}")
    lineas.append("\n=== STOCK ACTUAL de insumos criticos (fuente: ERP/CNN) ===")
    for insumo, cant in stock_actual.items():
        lineas.append(f"- {insumo}: {cant}")
    lineas.append("\nNota: los insumos no listados se consideran con disponibilidad suficiente.")
    return "\n".join(lineas)


def rag_answer(client, question, chunks, chunk_embeddings, datos_numericos, historial):
    q_emb = client.embeddings.create(model=MODELO_EMBEDDING, input=[question]).data[0].embedding
    sims = cosine_similarity([q_emb], chunk_embeddings)[0]
    idx = sims.argsort()[::-1][:TOP_K]
    contexto_prov = "\n\n".join(chunks[i] for i in idx)
    contexto = (f"--- FICHAS DE PROVEEDORES ---\n{contexto_prov}\n\n"
                f"--- DATOS NUMERICOS (BOM + STOCK) ---\n{datos_numericos}")
    user_prompt = f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {question}"

    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}] + historial
    mensajes.append({"role": "user", "content": user_prompt})
    resp = client.chat.completions.create(
        model=MODELO_CHAT, messages=mensajes, temperature=TEMPERATURA, max_tokens=MAX_TOKENS)
    return resp.choices[0].message.content


# ---------- Interfaz ----------
st.set_page_config(page_title="Asistente de Compras Maincal", page_icon="📦", layout="centered")
st.title("📦 Asistente de Compras Maincal")
st.caption("LLM + RAG · Producto Cronos-N04")

with st.sidebar:
    st.header("Configuracion")
    api_key = st.text_input("API key de OpenAI", type="password",
                            value=os.environ.get("OPENAI_API_KEY", ""))
    st.divider()
    st.header("Stock actual (demo)")
    st.caption("Cambia los valores y volve a preguntar para simular escenarios.")
    stock_pu = st.number_input("Conjunto Sistema PU (kg)", min_value=0, value=1200, step=100)
    stock_punt = st.number_input("Puntera de Acero 59 Normal (u)", min_value=0, value=18000, step=500)
    stock_caja = st.number_input("Caja de empaque (u)", min_value=0, value=5000, step=500)
    if st.button("Reiniciar conversacion"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("Ingresa tu API key de OpenAI en la barra lateral para empezar.")
    st.stop()

if not (os.path.exists(PDF_PATH) and os.path.exists(BOM_PATH)):
    st.error(f"Faltan archivos. Asegurate de tener '{PDF_PATH}' y '{BOM_PATH}' en esta carpeta.")
    st.stop()

client = OpenAI(api_key=api_key)
chunks = cargar_chunks()
bom_df = cargar_bom()
chunk_embeddings = generar_embeddings(client, chunks)

stock_actual = {
    "Conjunto Sistema PU": stock_pu,
    "Puntera de Acero 59 Normal": stock_punt,
    "Caja de empaque": stock_caja,
}
datos_numericos = construir_datos_numericos(bom_df, stock_actual)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrada de chat
if pregunta := st.chat_input("Escribi tu consulta (ej: que insumo priorizar?)"):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta = rag_answer(client, pregunta, chunks, chunk_embeddings,
                                   datos_numericos, st.session_state.messages[:-1])
        st.markdown(respuesta)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
