import streamlit as st
from PIL import Image
import numpy as np
import io

# Título de la aplicación
st.title("Subir y procesar una imagen")

# Widget para subir archivos
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen en el frontend
    st.image(uploaded_file, caption="Imagen Subida.", use_container_width=True)
    st.write("")
    st.write("Procesando imagen...")

    # Acceder a los datos del archivo en el backend
    # Leer el archivo como bytes
    image_bytes = uploaded_file.read()
    
    # Ejemplo de procesamiento: Convertir a un objeto de imagen de PIL o a un arreglo de NumPy
    try:
        # Usando PIL (Pillow)
        pil_image = Image.open(io.BytesIO(image_bytes))
        st.write("La imagen se ha cargado correctamente en el backend (como objeto PIL).")
        
        # Aquí puedes llamar a tus funciones de backend.
        # Por ejemplo, una función que reciba una imagen PIL.
        # procesar_imagen(pil_image)
        
        # Si tu backend usa OpenCV, puedes convertirla a un arreglo de NumPy
        opencv_image = np.array(pil_image)
        st.write("La imagen se ha convertido a un arreglo de NumPy para OpenCV.")
        # Por ejemplo, una función que reciba un arreglo de NumPy.
        # analizar_anaquel(opencv_image)
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")