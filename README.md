## RAG de documentos

Aplicación local con Streamlit para cargar documentos PDF, DOCX y Excel,
indexarlos y hacer preguntas sobre su contenido. Cada formato tiene un lector
independiente en `src/rag_documentos/readers/`.

**Problema que soluciona:**
Permite facilitar la busqueda de informacion por documentos, global, por etiqueta o en un documento concreto.
Cada posibilidad de busqueda dara resultados explicados en la propia pagina con ejemplos dados.
Permite ahorra tiempo de busqueda de documento en caso de olvido del nombre del archivo/documento  buscado
y facilitar buscar informacion exacta sin tener que leer todo el documento.


## Requisitos

- Python 3.10 o posterior

## Instalación y ejecución

Desde esta carpeta, crea y activa un entorno virtual e instala las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Instala [Ollama](https://ollama.com/) y descarga los modelos usados por defecto:

```powershell
ollama pull nomic-embed-text
ollama pull llama3
```

Ollama debe estar en ejecución antes de indexar documentos o hacer preguntas.
El modelo de embeddings convierte texto en vectores para la búsqueda; el modelo
de chat redacta las respuestas con los fragmentos recuperados.

La sección **Búsqueda global** busca en los fragmentos y verifica cada extracto
con el modelo de chat. Solo muestra documentos cuyo contenido corresponde a la
consulta, con un máximo de cinco resultados; si no encuentra coincidencias,
indica que no hay documentos relacionados.

La sección **Búsqueda por documento** permite filtrar los nombres indexados,
seleccionar un archivo en el desplegable y hacer una pregunta sobre él. La
recuperación vectorial se limita al documento seleccionado, así que los
fragmentos de otros archivos no se incluyen en el contexto de la respuesta.

Al subir archivos, primero pulsa **Clasificar documentos con IA**. Ollama
clasifica el contenido (no el nombre del archivo) en una categoría fija —Libro,
CV, Factura, Contrato, Manual, Informe, Artículo, Presentación, Hoja de cálculo
u Otro— y propone una subcategoría. Revisa el resultado y pulsa **Indexar
documentos clasificados**. Puedes corregir manualmente la categoría o la
subcategoría antes de indexar. Por ejemplo, un PDF de una novela debe quedar
como `Libro` con subcategoría `Novela`, independientemente del nombre del
archivo. Si eliges **Otro**, puedes añadir una etiqueta personalizada opcional;
si la dejas vacía, la etiqueta será `Otro`. La categoría, subcategoría y
etiqueta se guardan en los metadatos de cada fragmento.

La página principal reúne la subida y la gestión de etiquetas: permite elegir
un documento indexado y sustituir su etiqueta, actualizar todos sus fragmentos
y conservar la categoría y subcategoría originales.

**Busqueda global** permite buscar fragmentos por todos los documentos subidos 
despues de realizar la busqueda se mostraran de 2 a 5  archivos en los que aparecen las
coincidencias y el chunk/bloque de texto donde aparece el fragmento.

**Búsqueda por etiqueta** interpreta una petición en lenguaje natural, la
relaciona con las etiquetas existentes y muestra todos los documentos
coincidentes en paneles desplegables. Los documentos indexados antes de esta
función se pueden etiquetar desde la opción **Etiquetar documentos indexados
anteriormente**.

**Busqueda por documento** interpreta una busqueda de informacion en el documento concreto.
Dispone de un menu plegable con los documentos existentes y la busqueda del documento escribiendo su nombre.


La aplicación usa `st.navigation` para controlar la navegación multipágina
nativa de Streamlit. El logo de la aplicacion aparece arriba en la pagina principal.
La página principal contiene la subida y el cambio
de etiquetas; las búsquedas global, por etiqueta y por documento están en el
menú lateral. Al final de la página principal se indica cómo continuar con una
búsqueda. Los documentos que ya estaban indexados permanecen disponibles.

Inicia la interfaz desde la carpeta del proyecto:

```powershell
streamlit run app.py
```

## Estructura

```text
rag_documentos/
├── app.py
├── pages/
│   ├── 0_Gestion_Y_Subida.py
│   ├── 1_Busqueda_Global.py
│   ├── 2_Busqueda_Etiqueta.py
│   └── 3_Busqueda_Documento.py
├── assets/
│   └── rag_logo_banner.png
├── requirements.txt
├── src/
│   └── rag_documentos/
│       ├── __init__.py
│       ├── document_loader.py
│       ├── rag_service.py
│       ├── searching/
│       │   ├── __init__.py
│       │   └── global_search.py
│       ├── tagging/
│       │   ├── __init__.py
│       │   └── document_tags.py
│       └── readers/
│           ├── __init__.py
│           ├── docx_reader.py
│           ├── excel_reader.py
│           └── pdf_reader.py
└── tests/
    ├── test_document_loader.py
    ├── test_document_tags.py
    └── test_rag_service.py
```

La extracción de PDF trabaja con texto seleccionable; los documentos escaneados
necesitarán una etapa OCR. Los fragmentos tienen un tamaño de 500 caracteres
y un solapamiento de 50. ChromaDB conserva el índice en la carpeta local
`documentos_subidos/`, en una colección del mismo nombre; no se envían los
documentos a servicios externos.

Puedes configurar `OLLAMA_HOST`, `OLLAMA_CHAT_MODEL`,
`OLLAMA_EMBEDDING_MODEL` y `CHROMA_PERSIST_DIRECTORY` como variables de entorno.


