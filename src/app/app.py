import streamlit as st
from PIL import Image
import numpy as np
import io
from scanner_api.app_models import Scanner, ScannerResult, OrderModel, YOLO
import pandas as pd
from pathlib import Path

# Define la ruta raíz del proyecto de manera robusta
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Rutas a los archivos
MODEL_PATH = PROJECT_ROOT / "models" / "my_model.pt"
SHELF_CSV_PATH = PROJECT_ROOT / "src" / "app" / "scanner_api" / "shelves" / "BATERIAS (1F) 0,36M.csv"

# --- Cargar modelos y datos una sola vez con @st.cache_resource ---
@st.cache_resource
def load_models():
    """
    Función para cargar el modelo YOLO y el OrderModel.
    Usa la caché de Streamlit para no recargarlos.
    """
    try:
        yolo_model = YOLO(str(MODEL_PATH))
        order_model = OrderModel(str(SHELF_CSV_PATH))
        scanner = Scanner(yolo_model, order_model)
        return scanner
    except FileNotFoundError as e:
        st.error(f"Error al cargar modelos o datos: {e}. Asegúrate de que las rutas son correctas.")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar modelos: {e}")
        return None

# Cargar la instancia del scanner al inicio de la app
scanner_instance = load_models()

# --- Interfaz de usuario de Streamlit ---
st.title("Analizador de Anaquel")

uploaded_file = st.file_uploader("Carga una imagen del anaquel...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and scanner_instance:
    # Mostrar la imagen en el frontend
    st.image(uploaded_file, caption="Imagen Subida.", use_container_width=True)
    st.write("")
    st.write("Procesando imagen...")

    image_bytes = uploaded_file.read()

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # --- Ejecutar el pipeline de escaneo ---
        scanner_result = scanner_instance.predict(pil_image)
        
        # Extraer los DataFrames de los resultados
        comparison_df, detection_df = scanner_result.order_result

        # Productos en su lugar y fuera de lugar
        productos_en_lugar = comparison_df[comparison_df['detected'] == True]
        productos_fuera_de_lugar = comparison_df[comparison_df['detected'] == False]
        
        # Cantidades
        total_productos_detectados = len(detection_df)
        total_en_lugar = len(productos_en_lugar)
        total_fuera_de_lugar = len(productos_fuera_de_lugar)

        # --- Sección 1: Resumen del Escaneo ---
        st.subheader("Resumen del Escaneo")
        st.write(f"🔍 Se detectaron **{total_productos_detectados}** productos en total.")
        st.write(f"✅ **{total_en_lugar}** productos están en su lugar.")
        st.write(f"❌ **{total_fuera_de_lugar}** productos no están en su lugar.")
        
        st.markdown("---") # Separador visual
        
        # --- Sección 2: Productos Detectados ---
        st.subheader("Productos detectados en la imagen")
        if not detection_df.empty:
            st.write("Se han encontrado los siguientes productos:")
            for index, row in detection_df.iterrows():
                st.write(f"- {row['detected_SKU']} (Posición {int(row['pos'])})")
        else:
            st.write("No se detectaron productos en la imagen.")

        st.markdown("---") # Separador visual

        # --- Sección 3: Estado del Anaquel ---
        st.subheader("Estado del Anaquel")
        
        # Productos en su lugar
        st.write(f"✅ **Productos en su lugar ({total_en_lugar}):**")
        if not productos_en_lugar.empty:
            for index, row in productos_en_lugar.iterrows():
                st.write(f"- {row['expected_SKU']} (Posición {int(row['pos'])})")
        else:
            st.write("¡Todos los productos esperados fueron detectados en su lugar!")

        # Productos que no están en su lugar
        st.write("---")
        st.write(f"❌ **Productos que no están en su lugar ({total_fuera_de_lugar}):**")
        if not productos_fuera_de_lugar.empty:
            for index, row in productos_fuera_de_lugar.iterrows():
                st.write(f"- {row['expected_SKU']} (Posición {int(row['pos'])})")
        else:
            st.write("¡No se encontraron productos fuera de lugar!")

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

elif uploaded_file is not None and not scanner_instance:
    st.error("La aplicación no pudo cargar los modelos. Por favor, revise los mensajes de error para más detalles.")
