import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import io
from ..scanner_api.app_models import Scanner, OrderModel
from ..scanner_api.scanner_classes import ScannerResult
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import cv2
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


st.markdown("""
    <style>
    /* Oculta elementos nativos de Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
            
    h2, h3, h4, h5, h6, 
    p, 
    div[data-testid="stText"], 
    div[data-testid="stExpander"] > div > div > div > p,
    .stMarkdown,
    .stAlert {
        color: #FAFAFA !important; 
    }
    
    a {
        color: #6C757D !important; 
    }
    
    div.stApp {
        background-color: #0E1117; 
    }
    div[data-testid="stAppViewContainer"] {
        background-color: #0E1117; 
    }
    section.main {
        background-color: #0E1117 !important; 
    }
    div[data-testid="stBlockContainer"] {
        background-color: #0E1117;
    }
    
    body {
        text-align: center;

    }
            
    h1 {
        text-align: center;
        font-size: 4em !important; 
        margin-top: -20px;
    }
    
    .st-emotion-cache-18jrb99 > img, .st-emotion-cache-h5rgay > img, 
    div.st-emotion-cache-vk2m4n > img, 
    div.st-emotion-cache-10qj09c > img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
            
    
    </style>
    """, unsafe_allow_html=True)

        
    

MODEL_PATH = PROJECT_ROOT / "models" / "my_model.pt"
SHELF_CSV_PATH = PROJECT_ROOT / "src" / "scanner_api" / "shelves" / "BATERIAS (1F) 0,36M.csv"
LOGO_PATH = PROJECT_ROOT / "src" / "app"/ "assets" / "images" / "logo.png"

@st.cache_resource
def load_models():

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
st.image(str(LOGO_PATH), use_container_width=True) 

st.markdown("---")

uploaded_file = st.file_uploader("Carga una imagen del anaquel...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and scanner_instance:
    st.write("Procesando imagen...")

    image_bytes = uploaded_file.read()

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Lógica de rotación EXIF
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(pil_image._getexif().items())
            
            if exif[orientation] == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                pil_image = pil_image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
            
        # --- Redimensionar la imagen original para la visualización ---
        MAX_WIDTH_UPLOADED = 400
        width, height = pil_image.size
        
        if width > MAX_WIDTH_UPLOADED:
            new_height = int(height * (MAX_WIDTH_UPLOADED / width))
            display_image = pil_image.resize((MAX_WIDTH_UPLOADED, new_height))
        else:
            display_image = pil_image

        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(display_image, caption="Imagen Subida.", use_container_width=True)
            
        # Usa la imagen original para el modelo
        scanner_result: ScannerResult = scanner_instance.predict(pil_image)
        
        # Extraer los DataFrames de los resultados
        comparison_df = scanner_result.order_result.comparison_df
        detection_df = scanner_result.order_result.detection_df

        # Lógica de procesamiento de DataFrames
        def get_brand_from_sku(sku):
            sku_parts = str(sku).split(' ')
            if len(sku_parts) > 1:
                return sku_parts[1]
            return 'Sin Marca'

        if 'detected_SKU' in detection_df.columns:
            detection_df['brand'] = detection_df['detected_SKU'].apply(get_brand_from_sku)
        else:
            st.warning("La columna 'detected_SKU' no se encontró en los resultados de la detección.")
            detection_df['brand'] = 'Desconocida'
            
        productos_en_lugar = comparison_df[comparison_df['detected'] == True]
        productos_fuera_de_lugar = comparison_df[comparison_df['detected'] == False]
        
        total_productos_esperados = len(comparison_df)
        total_productos_detectados = len(detection_df)
        total_en_lugar = len(productos_en_lugar)
        total_fuera_de_lugar = len(productos_fuera_de_lugar)

        
        st.markdown("---") 
        st.subheader("Resumen de Caras de SKU")
        
        if total_productos_detectados > 0:
            st.write(f"🔍 Se encontraron un total de **{total_productos_detectados}** caras de SKU.")
        else:
            st.write("🔍 No se detectó ninguna cara de SKU en la imagen.")
            
        with st.expander("Ver lista de productos encontrados"):
            if not detection_df.empty:
                marcas_encontradas = detection_df.groupby('brand')
                for marca, group in marcas_encontradas:
                    with st.expander(f"**{marca}** ({len(group)} productos)"):
                        for index, row in group.iterrows():
                            st.write(f"- {row['detected_SKU']} (Posición: {int(row['pos'])})")
            else:
                st.write("No se detectaron productos en la imagen.")
        
        scanned_image_arr = scanner_result.yolo_result.plot()
        scanned_image_arr_rgb = cv2.cvtColor(scanned_image_arr, cv2.COLOR_BGR2RGB)
        scanned_pil_image = Image.fromarray(scanned_image_arr_rgb) 
        
        MAX_WIDTH_SCANNED = 400
        scanned_width, scanned_height = scanned_pil_image.size
        
        if scanned_width > MAX_WIDTH_SCANNED:
            new_scanned_height = int(scanned_height * (MAX_WIDTH_SCANNED / scanned_width))
            display_scanned_image_sku = scanned_pil_image.resize((MAX_WIDTH_SCANNED, new_scanned_height))
        else:
            display_scanned_image_sku = scanned_pil_image

        st.markdown("---")
        st.subheader("Anaquel Escaneado (Detección de SKU)")
        
        col_scan1, col_scan2, col_scan3 = st.columns([1, 4, 1])
        with col_scan2:
            st.image(display_scanned_image_sku, caption="Anaquel con SKU detectados.", use_container_width=True)
        
        
        st.markdown("---") 
        st.subheader("Estado de Inventario")

        if total_productos_esperados > 0:
            porcentaje_en_lugar = (total_en_lugar / total_productos_esperados) * 100
            st.write(f"✅ **{porcentaje_en_lugar:.1f}%** de los productos están en su lugar.")
        else:
            st.write("No hay productos esperados en la configuración del anaquel. No se puede calcular el porcentaje.")

        with st.expander("Ver productos que no están en orden"):
            if not productos_fuera_de_lugar.empty:
                for index, row in productos_fuera_de_lugar.iterrows():
                    st.write(f"- {row['expected_SKU']} (Posición esperada: {int(row['pos'])})")
            else:
                st.write("¡Todos los productos esperados fueron detectados en su lugar!")


       
        
        # Obtener la imagen con las cajas de color VERDE/ROJO (ya es RGB)
        highlighted_pil_image = scanner_result.highlighted_boxes()

        # Redimensionar la imagen
        MAX_WIDTH_SCANNED = 400
        scanned_width, scanned_height = highlighted_pil_image.size
        
        if scanned_width > MAX_WIDTH_SCANNED:
            new_scanned_height = int(scanned_height * (MAX_WIDTH_SCANNED / scanned_width))
            display_scanned_image_order = highlighted_pil_image.resize((MAX_WIDTH_SCANNED, new_scanned_height))
        else:
            display_scanned_image_order = highlighted_pil_image

        st.markdown("---")
        st.subheader("Verificación de Orden del Anaquel")
        
        col_order1, col_order2, col_order3 = st.columns([1, 4, 1])
        with col_order2:
            st.image(display_scanned_image_order, caption="Verde: Producto correcto en posición. Rojo: Producto incorrecto o ausente.", use_container_width=True)
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

elif uploaded_file is not None and not scanner_instance:
    st.error("La aplicación no pudo cargar los modelos. Por favor, revise los mensajes de error para más detalles.")