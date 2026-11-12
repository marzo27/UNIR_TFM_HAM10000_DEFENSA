"""
Aplicación web de inferencia — DenseNet201 multitarea (HAM10000)

Cabeza A: triaje binario (benigno / maligno)
Cabeza B: clasificación de 7 clases de lesión cutánea

Requiere el archivo de pesos 'densenet201_multitarea.pth' en la misma carpeta
(se genera con el bloque de guardado añadido al script de entrenamiento).
"""

import os

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

RUTA_PESOS = os.environ.get("RUTA_PESOS", "densenet201_multitarea.pth")

# Mismo orden que MAPEO_FIJO_CLASES del script de entrenamiento
CLASES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NOMBRES_ES = {
    "akiec": "Queratosis actínica / enfermedad de Bowen (akiec)",
    "bcc": "Carcinoma basocelular (bcc)",
    "bkl": "Lesión benigna tipo queratosis (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "Melanoma (mel)",
    "nv": "Nevus melanocítico (nv)",
    "vasc": "Lesión vascular (vasc)",
}

torch.set_num_threads(max(1, os.cpu_count() or 1))
device = torch.device("cpu")


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


def cargar_modelo():
    if not os.path.exists(RUTA_PESOS):
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos '{RUTA_PESOS}'. "
            "Debe generarse con torch.save(model.state_dict(), ...) tras el entrenamiento "
            "y subirse junto a app.py."
        )
    modelo = DenseNetMultitarea(num_clases_especificas=len(CLASES))
    estado = torch.load(RUTA_PESOS, map_location=device)
    modelo.load_state_dict(estado)
    modelo.to(device).eval()
    return modelo


# Misma transformación que 'transformacion_val' del entrenamiento
transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

modelo = cargar_modelo()


def evaluar(imagen: Image.Image):
    if imagen is None:
        raise gr.Error("Sube una imagen para evaluarla.")

    tensor = transformacion(imagen.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_bin, logits_esp = modelo(tensor)
        prob_bin = torch.softmax(logits_bin, dim=1)[0]
        prob_esp = torch.softmax(logits_esp, dim=1)[0]

    triaje = {"Benigno": float(prob_bin[0]), "Maligno": float(prob_bin[1])}
    detalle = {NOMBRES_ES[c]: float(p) for c, p in zip(CLASES, prob_esp)}
    return triaje, detalle


AVISO = (
    "**Aviso importante.** Esta herramienta es un prototipo académico desarrollado en el marco de "
    "un Trabajo Fin de Máster. **No es un dispositivo médico ni sustituye el diagnóstico de un "
    "dermatólogo.** El modelo se entrenó con imágenes dermatoscópicas del conjunto HAM10000; su "
    "comportamiento con fotografías tomadas con un teléfono móvil u otras cámaras no ha sido "
    "validado. Ante cualquier lesión cutánea que le preocupe, consulte a un profesional sanitario."
)

with gr.Blocks(title="Clasificación de lesiones cutáneas — DenseNet201") as demo:
    gr.Markdown("# Clasificación de lesiones cutáneas con DenseNet201")
    gr.Markdown("# TFM: Evaluación comparativa de SVM y DenseNet201 para la clasificación automática de enfermedades dermatológicas")
    gr.Markdown("# Clasificación de lesiones cutáneas con DenseNet201")
    gr.Markdown(
        "Sube una imagen dermatoscópica de la lesión. El modelo devuelve el triaje "
        "benigno/maligno (Cabeza A) y la probabilidad de cada una de las 7 clases (Cabeza B)."
    )
    gr.Markdown(AVISO)
    with gr.Row():
        entrada = gr.Image(type="pil", label="Imagen de la lesión")
        with gr.Column():
            salida_triaje = gr.Label(label="Triaje (Cabeza A)", num_top_classes=2)
            salida_detalle = gr.Label(label="Clasificación detallada (Cabeza B)", num_top_classes=7)
    boton = gr.Button("Evaluar imagen", variant="primary")
    boton.click(evaluar, inputs=entrada, outputs=[salida_triaje, salida_detalle])

if __name__ == "__main__":
    demo.launch()
