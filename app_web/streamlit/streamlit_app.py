"""
Aplicación web (Streamlit) — DenseNet201 multitarea sobre HAM10000

Cabeza A: triaje binario (benigno / maligno)
Cabeza B: clasificación de 7 clases de lesión cutánea

Archivos necesarios en la misma carpeta:
  - densenet201_multitarea.pth  (archivo único generado por torch.save)
  - config_modelo.json          (generado por densenet201_ham10000_balanceado_peso.py)
"""

import json
import os

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Clasificación de lesiones cutáneas", page_icon="🩺", layout="centered")

AQUI = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "archivo_pesos": "densenet201_multitarea.pth",
    "img_size": [224, 224],
    "normalizacion": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "clases": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
}
ruta_config = os.path.join(AQUI, "config_modelo.json")
if os.path.isfile(ruta_config):
    with open(ruta_config, encoding="utf-8") as f:
        CONFIG.update(json.load(f))

CLASES = CONFIG["clases"]
RUTA_PESOS = os.path.join(AQUI, CONFIG["archivo_pesos"])

NOMBRES_ES = {
    "akiec": "Queratosis actínica / enfermedad de Bowen (akiec)",
    "bcc": "Carcinoma basocelular (bcc)",
    "bkl": "Lesión benigna tipo queratosis (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "Melanoma (mel)",
    "nv": "Nevus melanocítico (nv)",
    "vasc": "Lesión vascular (vasc)",
}


class DenseNetMultitarea(nn.Module):
    """Misma arquitectura que en el entrenamiento, sin descargar pesos de ImageNet."""

    def __init__(self, num_clases_especificas=7):
        super().__init__()
        self.densenet = models.densenet201(weights=None)
        num_caracteristicas = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        self.cabeza_binaria = nn.Linear(num_caracteristicas, 2)
        self.cabeza_especifica = nn.Linear(num_caracteristicas, num_clases_especificas)

    def forward(self, x):
        caracteristicas = self.densenet(x)
        return self.cabeza_binaria(caracteristicas), self.cabeza_especifica(caracteristicas)


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo():
    torch.set_num_threads(2)
    modelo = DenseNetMultitarea(num_clases_especificas=len(CLASES))
    estado = torch.load(RUTA_PESOS, map_location="cpu", weights_only=True)
    modelo.load_state_dict(estado)
    modelo.eval()
    return modelo


transformacion = transforms.Compose([
    transforms.Resize(tuple(CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG["normalizacion"]["mean"], std=CONFIG["normalizacion"]["std"]),
])


def evaluar(modelo, imagen: Image.Image):
    tensor = transformacion(imagen.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits_bin, logits_esp = modelo(tensor)
        prob_bin = torch.softmax(logits_bin, dim=1)[0]
        prob_esp = torch.softmax(logits_esp, dim=1)[0]
    return prob_bin.tolist(), prob_esp.tolist()


st.title("Clasificación de lesiones cutáneas con DenseNet201")
st.warning(
    "**Aviso importante.** Esta herramienta es un prototipo académico desarrollado en el marco de "
    "un Trabajo Fin de Máster. **No es un dispositivo médico ni sustituye el diagnóstico de un "
    "dermatólogo.** El modelo se entrenó con imágenes dermatoscópicas del conjunto HAM10000; su "
    "comportamiento con fotografías tomadas con un teléfono móvil u otras cámaras no ha sido "
    "validado. Ante cualquier lesión cutánea que le preocupe, consulte a un profesional sanitario."
)
st.write(
    "Sube una imagen dermatoscópica de la lesión. El modelo devuelve el triaje benigno/maligno "
    "(Cabeza A) y la probabilidad de cada una de las 7 clases (Cabeza B)."
)

archivo = st.file_uploader("Imagen de la lesión", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    if st.button("Evaluar imagen", type="primary"):
        modelo = cargar_modelo()
        with st.spinner("Evaluando..."):
            prob_bin, prob_esp = evaluar(modelo, imagen)

        st.subheader("Triaje (Cabeza A)")
        st.write(f"Benigno: {prob_bin[0]:.1%}")
        st.progress(prob_bin[0])
        st.write(f"Maligno: {prob_bin[1]:.1%}")
        st.progress(prob_bin[1])

        st.subheader("Clasificación detallada (Cabeza B)")
        orden = sorted(zip(CLASES, prob_esp), key=lambda t: t[1], reverse=True)
        for clase, p in orden:
            st.write(f"{NOMBRES_ES.get(clase, clase)}: {p:.1%}")
            st.progress(p)
