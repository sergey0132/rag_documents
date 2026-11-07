from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="RAG de documentos", page_icon="📄", layout="wide")

# st.navigation dibuja el menú en el punto en que se llama; colocar la marca
# antes de esa llamada la deja arriba de los enlaces laterales.
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def cambiar_tema() -> None:
    """Actualiza los colores del tema según el estado del interruptor."""
    if st.session_state.switch_theme:
        theme_options = {
            "theme.backgroundColor": "#0E1117",
            "theme.secondaryBackgroundColor": "#262730",
            "theme.textColor": "#FAFAFA",
            "theme.primaryColor": "#FF4B4B",
        }
        st.session_state.dark_mode = True
    else:
        theme_options = {
            "theme.backgroundColor": "#FFFFFF",
            "theme.secondaryBackgroundColor": "#F0F2F6",
            "theme.textColor": "#31333F",
            "theme.primaryColor": "#FF4B4B",
        }
        st.session_state.dark_mode = False

    # Streamlit no ofrece cambiar el tema de una sesión en caliente como API
    # pública; esta llamada interna se prueba al cambiar el interruptor.
    for option, value in theme_options.items():
        st._config.set_option(option, value)
    st.rerun()


st.sidebar.toggle(
    "Modo oscuro 🌙",
    value=st.session_state.dark_mode,
    key="switch_theme",
    on_change=cambiar_tema,
)

if st.session_state.dark_mode:
    background_color = "#0E1117"
    secondary_background_color = "#262730"
    text_color = "#FAFAFA"
else:
    background_color = "#FFFFFF"
    secondary_background_color = "#F0F2F6"
    text_color = "#31333F"

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: {background_color} !important;
        color: {text_color} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {secondary_background_color} !important;
        color: {text_color} !important;
    }}
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {{
        color: {text_color} !important;
    }}
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div {{
        background-color: {secondary_background_color} !important;
        color: {text_color} !important;
    }}
    [data-testid="stBaseButton-primary"] {{
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

navigation = st.navigation(
    [
        st.Page(
            "pages/0_Gestion_Y_Subida.py",
            title="Inicio · Subida y etiquetas",
            icon="📂",
            default=True,
        ),
        st.Page(
            "pages/1_Busqueda_Global.py",
            title="Búsqueda global",
            icon="🌐",
        ),
        st.Page(
            "pages/2_Busqueda_Etiqueta.py",
            title="Búsqueda por etiqueta",
            icon="🏷️",
        ),
        st.Page(
            "pages/3_Busqueda_Documento.py",
            title="Búsqueda por documento",
            icon="📄",
        ),
    ],
    position="sidebar",
)
navigation.run()
