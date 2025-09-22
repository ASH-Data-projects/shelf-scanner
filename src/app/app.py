import streamlit as st
from PIL import Image
import numpy as np
import io
from scanner_api.app_models import Scanner, OrderModel
from ultralytics import YOLO
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "my_model.pt"
SHELF_CSV_PATH = PROJECT_ROOT / "src" / "app" / "scanner_api" / "shelves" / "BATERIAS (1F) 0,36M.csv"

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

# Interfaz de usuario de Streamlit
st.title("Analizador de Anaquel")

uploaded_file = st.file_uploader("Carga una imagen del anaquel...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and scanner_instance:
    st.write("Procesando imagen...")

    image_bytes = uploaded_file.read()

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    
        MAX_WIDTH = 400
        width, height = pil_image.size
        
        if width > MAX_WIDTH:
            new_height = int(height * (MAX_WIDTH / width))
            display_image = pil_image.resize((MAX_WIDTH, new_height))
        else:
            display_image = pil_image

        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(display_image, caption="Imagen Subida.", use_container_width=True)

    
        
        # Usa la imagen original para el modelo para no perder calidad
        scanner_result = scanner_instance.predict(pil_image)
        
        
        
        # Ejecutar el pipeline de escaneo
        scanner_result = scanner_instance.predict(pil_image)
        
        # Extraer los DataFrames de los resultados
        comparison_df, detection_df = scanner_result.order_result

        # Productos en su lugar y fuera de lugar
        productos_en_lugar = comparison_df[comparison_df['detected'] == True]
        productos_fuera_de_lugar = comparison_df[comparison_df['detected'] == False]
        
        # Cantidades
        total_productos_esperados = len(comparison_df)
        total_productos_detectados = len(detection_df)
        total_en_lugar = len(productos_en_lugar)
        total_fuera_de_lugar = len(productos_fuera_de_lugar)
        
        # --- Sección 1: Resumen del Escaneo (con porcentajes) ---
        st.subheader("Resumen del Escaneo")
        
        if total_productos_esperados > 0:
            porcentaje_en_lugar = (total_en_lugar / total_productos_esperados) * 100
            porcentaje_fuera_de_lugar = (total_fuera_de_lugar / total_productos_esperados) * 100

            st.write(f"🔍 **{total_productos_detectados}** productos detectados de un total de **{total_productos_esperados}** esperados.")
            st.write(f"✅ **{porcentaje_en_lugar:.1f}%** de los productos están en su lugar.")
            st.write(f"❌ **{porcentaje_fuera_de_lugar:.1f}%** de los productos no están en su lugar.")
        else:
            st.write("No hay productos esperados en la configuración del anaquel. No se puede calcular el porcentaje.")
        
        

        # --- Botón expandible para detalles ---
        with st.expander("Ver detalles del escaneo"):
            # NUEVA SECCIÓN: Imagen con elementos escaneados
            st.subheader("Imagen del Anaquel Escaneado")
            
            # Convierte el resultado de YOLO a un formato compatible con st.image
            scanned_image = scanner_result.yolo_result.plot()
            st.image(scanned_image, caption="Anaquel con elementos detectados.", use_container_width=True)
            
            st.markdown("---") # Separador visual
            # Productos detectados
            st.subheader("Productos detectados en la imagen")
            if not detection_df.empty:
                st.write("Se han encontrado los siguientes productos:")
                for index, row in detection_df.iterrows():
                    st.write(f"- {row['detected_SKU']} (Posición {int(row['pos'])})")
            else:
                st.write("No se detectaron productos en la imagen.")
            
            st.markdown("---")

            # Contenido de la sección de estado del anaquel
            st.subheader("Estado del Anaquel")
            
            # Productos en su lugar
            st.write(f"✅ **Productos en su lugar ({total_en_lugar}):**")
            if not productos_en_lugar.empty:
                for index, row in productos_en_lugar.iterrows():
                    st.write(f"- {row['expected_SKU']} (Posición esperada: {int(row['pos'])})")
            else:
                st.write("¡Todos los productos esperados fueron detectados en su lugar!")

            # Productos que no están en su lugar
            st.write("---")
            st.write(f"❌ **Productos que no están en su lugar ({total_fuera_de_lugar}):**")
            if not productos_fuera_de_lugar.empty:
                for index, row in productos_fuera_de_lugar.iterrows():
                    st.write(f"- {row['expected_SKU']} (Posición esperada: {int(row['pos'])})")
            else:
                st.write("¡No se encontraron productos fuera de lugar!")

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

elif uploaded_file is not None and not scanner_instance:
    st.error("La aplicación no pudo cargar los modelos. Por favor, revise los mensajes de error para más detalles.")