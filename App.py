"""
App de Analisis de Sentimiento
Lee un conjunto de comentarios (desde un archivo subido o desde un set
interno de ejemplo), clasifica cada uno como Positivo / Negativo / Neutral
y muestra graficas resumen.
"""

import io
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from textblob import TextBlob
from deep_translator import GoogleTranslator

# ── Configuracion de la pagina ──────────────────────────────
st.set_page_config(
    page_title="Analisis de Sentimiento",
    page_icon="🧠",
    layout="wide",
)

# Colores semanticos consistentes en toda la app, calibrados para fondo
# oscuro: verde = positivo, rojo = negativo, gris = neutral.
COLOR_POSITIVO = "#34D399"
COLOR_NEGATIVO = "#F87171"
COLOR_NEUTRAL = "#9CA3AF"
COLOR_ACENTO = "#818CF8"
COLORES = {"Positivo": COLOR_POSITIVO, "Negativo": COLOR_NEGATIVO, "Neutral": COLOR_NEUTRAL}

FONDO_APP = "#0E1117"
FONDO_TARJETA = "#161B22"
BORDE_TARJETA = "#262C36"
TEXTO_MUTED = "#9CA3AF"
TEXTO_PRINCIPAL = "#E6E8EB"

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at 15% 0%, rgba(99,102,241,0.12), transparent 40%),
            radial-gradient(circle at 85% 10%, rgba(52,211,153,0.08), transparent 35%),
            {FONDO_APP};
    }}

    .sa-hero {{
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(52,211,153,0.10));
        border: 1px solid {BORDE_TARJETA};
        margin-bottom: 1.4rem;
    }}
    .sa-hero h1 {{
        margin: 0;
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A5B4FC, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .sa-hero p {{
        margin: 0.35rem 0 0 0;
        color: {TEXTO_MUTED};
        font-size: 0.95rem;
    }}

    .sa-card {{
        background: {FONDO_TARJETA};
        border: 1px solid {BORDE_TARJETA};
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        text-align: left;
    }}
    .sa-card .sa-label {{
        color: {TEXTO_MUTED};
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .sa-card .sa-value {{
        font-size: 2.1rem;
        font-weight: 800;
        margin-top: 0.2rem;
        color: {TEXTO_PRINCIPAL};
    }}
    .sa-card .sa-sub {{
        color: {TEXTO_MUTED};
        font-size: 0.82rem;
        margin-top: 0.15rem;
    }}

    section[data-testid="stSidebar"] {{
        background: {FONDO_TARJETA};
        border-right: 1px solid {BORDE_TARJETA};
    }}

    div[data-testid="stExpander"] {{
        background: {FONDO_TARJETA};
        border: 1px solid {BORDE_TARJETA};
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def tarjeta_metrica(icono: str, etiqueta: str, valor, sub: str, color_valor: str) -> str:
    return f"""
    <div class="sa-card" style="border-top: 3px solid {color_valor};">
        <div class="sa-label">{icono} {etiqueta}</div>
        <div class="sa-value" style="color:{color_valor};">{valor}</div>
        <div class="sa-sub">{sub}</div>
    </div>
    """

# ── Set interno de comentarios de ejemplo ───────────────────
COMENTARIOS_EJEMPLO = [
    "El producto llego a tiempo y funciona excelente, muy contento con la compra.",
    "Pesimo servicio al cliente, nadie respondio mis mensajes en una semana.",
    "La calidad es aceptable, cumple lo basico pero nada sorprendente.",
    "Estoy encantado con el soporte tecnico, resolvieron mi problema en minutos.",
    "El envio llego tarde y la caja venia dañada, muy decepcionante.",
    "Precio justo por lo que ofrece, cumple sus funciones sin problema.",
    "La app se cierra sola constantemente, es muy frustrante de usar.",
    "Excelente atencion, el equipo fue muy amable y profesional.",
    "No es lo que esperaba, la descripcion no coincide con el producto real.",
    "Facil de instalar y configurar, todo funciono a la primera.",
    "El material se siente barato y se rompio a los pocos dias de uso.",
    "Buena relacion calidad-precio, lo recomendaria a otras personas.",
    "Tuve que esperar mas de una hora en la llamada de soporte, terrible.",
    "Superó mis expectativas, definitivamente volveria a comprar aqui.",
    "Es un producto normal, ni bueno ni malo, cumple lo minimo.",
    "La entrega fue rapidisima y el empaque estaba perfecto.",
    "Cobraron de mas y aun no me han dado explicacion ni reembolso.",
    "Interfaz intuitiva y muy facil de aprender a usar desde el primer dia.",
    "El manual de instrucciones esta incompleto y confuso.",
    "Gran experiencia de compra, todo el proceso fue muy claro y rapido.",
]


# ── Motor de analisis ───────────────────────────────────────
def clasificar_polaridad(polaridad: float) -> str:
    if polaridad > 0.1:
        return "Positivo"
    if polaridad < -0.1:
        return "Negativo"
    return "Neutral"


# Lexico de respaldo en español: se usa solo si el servicio de traduccion
# no esta disponible (limite de solicitudes, sin conexion, etc.), para que
# el analisis no dependa por completo de un servicio externo.
PALABRAS_POSITIVAS = {
    "excelente", "bueno", "buena", "buenos", "buenas", "genial", "encantado",
    "encantada", "contento", "contenta", "rapido", "rapida", "facil",
    "recomiendo", "recomendaria", "perfecto", "perfecta", "satisfecho",
    "satisfecha", "agradable", "profesional", "amable", "super", "gran",
    "mejor", "feliz", "comodo", "comoda", "confiable", "eficiente", "claro",
    "clara", "resolvieron", "increible", "fantastico", "fantastica",
}
PALABRAS_NEGATIVAS = {
    "pesimo", "pesima", "terrible", "malo", "mala", "malos", "malas",
    "decepcionante", "decepcionado", "decepcionada", "roto", "rota",
    "dañado", "dañada", "lento", "lenta", "confuso", "confusa",
    "frustrante", "tarde", "nunca", "nadie", "problema", "problemas",
    "reembolso", "incompleto", "incompleta", "defectuoso", "defectuosa",
    "caro", "cara", "error", "fallo", "falla", "queja", "horrible",
}


def analizar_con_lexico_local(texto: str) -> float:
    palabras = re.findall(r"[a-záéíóúñ]+", texto.lower())
    positivas = sum(1 for p in palabras if p in PALABRAS_POSITIVAS)
    negativas = sum(1 for p in palabras if p in PALABRAS_NEGATIVAS)
    if positivas == 0 and negativas == 0:
        return 0.0
    return max(-1.0, min(1.0, (positivas - negativas) / max(positivas + negativas, 1)))


def analizar_comentario(texto: str, traducir: bool) -> dict:
    metodo = "TextBlob (original)"
    texto_analizado = texto

    if traducir:
        try:
            texto_analizado = GoogleTranslator(source="auto", target="en").translate(texto)
            metodo = "TextBlob (traducido)"
        except Exception:
            texto_analizado = None  # marca que la traduccion no estuvo disponible

    if texto_analizado is not None:
        blob = TextBlob(texto_analizado)
        polaridad = blob.sentiment.polarity
        subjetividad = blob.sentiment.subjectivity
    else:
        # Respaldo sin conexion: lexico local en español
        polaridad = analizar_con_lexico_local(texto)
        subjetividad = None
        metodo = "Lexico local (sin traduccion)"

    return {
        "comentario": texto,
        "sentimiento": clasificar_polaridad(polaridad),
        "polaridad": round(polaridad, 3),
        "subjetividad": round(subjetividad, 3) if subjetividad is not None else None,
        "metodo": metodo,
    }


@st.cache_data(show_spinner=False)
def analizar_comentarios(comentarios: tuple, traducir: bool) -> pd.DataFrame:
    filas = [analizar_comentario(c, traducir) for c in comentarios if c.strip()]
    return pd.DataFrame(filas)


def leer_comentarios_de_archivo(archivo) -> list:
    nombre = archivo.name.lower()
    if nombre.endswith(".csv"):
        df = pd.read_csv(archivo)
        columna = df.columns[0]
        if "coment" in "".join(df.columns).lower():
            for col in df.columns:
                if "coment" in col.lower():
                    columna = col
                    break
        return df[columna].astype(str).tolist()

    contenido = io.TextIOWrapper(archivo, encoding="utf-8", errors="ignore").read()
    return [linea.strip() for linea in contenido.splitlines() if linea.strip()]


# ── Interfaz ─────────────────────────────────────────────────
st.markdown(
    """
    <div class="sa-hero">
        <h1>🧠 Analisis de Sentimiento de Comentarios</h1>
        <p>Clasifica un conjunto de comentarios como Positivo, Negativo o Neutral y muestra un resumen grafico.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Fuente de datos")
    fuente = st.radio(
        "¿De donde vienen los comentarios?",
        ["Set de ejemplo (interno)", "Subir archivo (.csv o .txt)"],
    )

    archivo = None
    if fuente == "Subir archivo (.csv o .txt)":
        archivo = st.file_uploader("Archivo con un comentario por linea o columna", type=["csv", "txt"])

    traducir = st.checkbox(
        "Traducir automaticamente antes de analizar (recomendado para comentarios en español)",
        value=True,
    )

    analizar = st.button("Analizar comentarios", type="primary")

if fuente == "Set de ejemplo (interno)":
    comentarios = COMENTARIOS_EJEMPLO
else:
    comentarios = leer_comentarios_de_archivo(archivo) if archivo else []

st.subheader("Comentarios a analizar")
st.write(f"{len(comentarios)} comentario(s) cargado(s).")
with st.expander("Ver comentarios"):
    for c in comentarios:
        st.write(f"- {c}")

if analizar:
    if not comentarios:
        st.warning("⚠️ No hay comentarios para analizar. Sube un archivo o usa el set de ejemplo.")
    else:
        with st.spinner("Analizando comentarios..."):
            resultados = analizar_comentarios(tuple(comentarios), traducir)

        usos_lexico = (resultados["metodo"] == "Lexico local (sin traduccion)").sum()
        if usos_lexico:
            st.info(
                f"ℹ️ El servicio de traduccion no estuvo disponible para {usos_lexico} "
                "comentario(s); se uso un analisis de respaldo por palabras clave en español."
            )

        conteo = resultados["sentimiento"].value_counts().reindex(
            ["Positivo", "Neutral", "Negativo"]
        ).fillna(0).astype(int)
        total = int(conteo.sum())

        st.subheader("Resumen")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(tarjeta_metrica("📊", "Total", total, "comentarios analizados", COLOR_ACENTO), unsafe_allow_html=True)
        with m2:
            st.markdown(tarjeta_metrica("😊", "Positivos", int(conteo["Positivo"]), f"{conteo['Positivo'] / total:.0%} del total", COLOR_POSITIVO), unsafe_allow_html=True)
        with m3:
            st.markdown(tarjeta_metrica("😐", "Neutrales", int(conteo["Neutral"]), f"{conteo['Neutral'] / total:.0%} del total", COLOR_NEUTRAL), unsafe_allow_html=True)
        with m4:
            st.markdown(tarjeta_metrica("😞", "Negativos", int(conteo["Negativo"]), f"{conteo['Negativo'] / total:.0%} del total", COLOR_NEGATIVO), unsafe_allow_html=True)

        st.write("")

        # Estilo oscuro compartido por ambas graficas de matplotlib
        plt.rcParams.update({
            "figure.facecolor": FONDO_TARJETA,
            "axes.facecolor": FONDO_TARJETA,
            "axes.edgecolor": BORDE_TARJETA,
            "axes.labelcolor": TEXTO_MUTED,
            "xtick.color": TEXTO_PRINCIPAL,
            "ytick.color": TEXTO_MUTED,
            "text.color": TEXTO_PRINCIPAL,
            "font.size": 10,
        })

        col_bar, col_pie = st.columns(2)

        with col_bar:
            fig_bar, ax_bar = plt.subplots(figsize=(4, 3.2))
            categorias = ["Positivo", "Neutral", "Negativo"]
            valores = [conteo[c] for c in categorias]
            colores = [COLORES[c] for c in categorias]
            ax_bar.bar(categorias, valores, color=colores, width=0.55)
            for i, v in enumerate(valores):
                ax_bar.text(i, v, str(v), ha="center", va="bottom", fontsize=10, color=TEXTO_PRINCIPAL)
            ax_bar.set_ylabel("Cantidad de comentarios")
            ax_bar.set_title("Comentarios por sentimiento", color=TEXTO_PRINCIPAL, fontweight="bold")
            ax_bar.spines[["top", "right"]].set_visible(False)
            ax_bar.spines[["left", "bottom"]].set_color(BORDE_TARJETA)
            ax_bar.grid(axis="y", color=BORDE_TARJETA, linewidth=0.6, alpha=0.6)
            ax_bar.set_axisbelow(True)
            fig_bar.tight_layout()
            st.pyplot(fig_bar)

        with col_pie:
            categorias_no_cero = [c for c in categorias if conteo[c] > 0]
            valores_no_cero = [conteo[c] for c in categorias_no_cero]
            colores_no_cero = [COLORES[c] for c in categorias_no_cero]
            fig_pie, ax_pie = plt.subplots(figsize=(4, 3.2))
            ax_pie.pie(
                valores_no_cero,
                labels=categorias_no_cero,
                autopct="%1.0f%%",
                colors=colores_no_cero,
                startangle=90,
                wedgeprops={"edgecolor": FONDO_TARJETA, "linewidth": 2},
                textprops={"color": TEXTO_PRINCIPAL},
            )
            ax_pie.set_title("Distribucion de sentimiento", color=TEXTO_PRINCIPAL, fontweight="bold")
            fig_pie.tight_layout()
            st.pyplot(fig_pie)

        st.subheader("Detalle por comentario")
        st.dataframe(resultados, width="stretch", hide_index=True)

        csv_resultado = resultados.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar resultados (CSV)",
            data=csv_resultado,
            file_name="resultados_sentimiento.csv",
            mime="text/csv",
        )
