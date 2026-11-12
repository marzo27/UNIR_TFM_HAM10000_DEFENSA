"""
DenseNet201 Multitarea — HAM10000 (Cabeza A: triaje binario | Cabeza B: 7 clases)

Mismo algoritmo, mismos hiperparámetros y mismo procedimiento de
entrenamiento que la versión original de este script. Cambios respecto a
esa versión:

  1) Se eliminan salidas de consola sin valor evaluativo: prints
     decorativos ("Fase X completada", emojis de éxito), la vista previa
     inicial del dataset (df.head(), value_counts() en crudo), el
     diccionario de nombres de lesión que no se usaba en ningún punto del
     script, la clase Dataset y el bloque de prueba iniciales (sustituidos
     más adelante por HAM10000DatasetCorregido; nunca llegaban a usarse) y
     el grid de 5 imágenes de muestra.

  2) Se amplía la evaluación final para cubrir las DOS cabezas del modelo
     (antes solo se evaluaba la Cabeza B; la Cabeza A de triaje binario no
     se llegaba a medir en ningún punto) con las mismas familias de
     métricas calculadas en svm_ham10000_v1.py: Accuracy, Balanced
     Accuracy, F1 (macro/weighted/micro), Precision/Recall (macro/weighted
     + por clase), AUC-ROC y Average Precision (por clase + macro/weighted,
     one-vs-rest), Cohen's Kappa, Matthews Correlation Coefficient, matriz
     de confusión (absoluta + normalizada), curvas ROC/PR y tiempos de
     entrenamiento e inferencia. Se añade un JSON con todas las métricas
     para poder comparar después con el SVM.

  3) VERSIÓN CON PESOS: al terminar el entrenamiento (antes de la evaluación
     y de las gráficas) se guardan los pesos finales del modelo y un JSON con
     la configuración de inferencia necesaria para la aplicación web. Los
     archivos se escriben en 'resultados_densenet_peso/' (carpeta distinta a
     la de la versión anterior, para no sobrescribir sus métricas). El resto
     del script (datos, arquitectura, hiperparámetros, bucle de entrenamiento
     y evaluación) no se modifica.

  NOTA: no se incluyen validación cruzada k-fold ni análisis PCA. Ninguno
  de los dos forma parte del procedimiento de esta red (entrenamiento por
  épocas sobre un único split train/val, sin reducción de dimensionalidad),
  así que añadirlos habría cambiado el algoritmo en sí, no solo sus salidas.
"""

import pandas as pd
import os

# 1. Cargar el archivo de metadatos
# Cambia 'dataset/HAM10000_metadata.csv' por tu ruta si es diferente
df = pd.read_csv('dataset/HAM10000_metadata_balanceo.csv', sep=";")


import os

# 1. Definir las rutas a las carpetas donde descomprimiste las imágenes
# Ajusta estos nombres si tus carpetas se llaman diferente
folder_part1 = 'dataset/HAM10000_images_part_1'
folder_part2 = 'dataset/HAM10000_images_part_2'

# 2. Crear un diccionario de mapeo: { 'nombre_imagen': 'ruta_completa_al_archivo' }
image_path_mapping = {}

# Escanear parte 1
if os.path.exists(folder_part1):
    for filename in os.listdir(folder_part1):
        if filename.endswith('.jpg'):
            id_img = os.path.splitext(filename)[0]
            image_path_mapping[id_img] = os.path.join(folder_part1, filename)

# Escanear parte 2
if os.path.exists(folder_part2):
    for filename in os.listdir(folder_part2):
        if filename.endswith('.jpg'):
            id_img = os.path.splitext(filename)[0]
            image_path_mapping[id_img] = os.path.join(folder_part2, filename)

print(f"Imágenes indexadas en disco: {len(image_path_mapping)}")


from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================================
# 1. DICCIONARIO GLOBAL E INMUTABLE (Plantilla Unificada)
# ==========================================================
MAPEO_FIJO_CLASES = {
    'akiec': 0, # Queratosis actínica
    'bcc': 1,   # Carcinoma basocelular
    'bkl': 2,   # Lesión benigna tipo queratosis
    'df': 3,    # Dermatofibroma
    'mel': 4,   # Melanoma (Maligno crítico)
    'nv': 5,    # Nevus melanocítico (Lunar común)
    'vasc': 6   # Lesión vascular
}

# ==========================================================
# 2. ESTRATEGIAS DE PROCESAMIENTO (DATA AUGMENTATION)
# ==========================================================
# El set de Train recibe aumentos geométricos homogéneos para evitar overfitting
transformacion_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # Volteo espejo horizontal
    transforms.RandomVerticalFlip(p=0.5),   # Volteo espejo vertical
    transforms.RandomRotation(degrees=15),   # Rotación leve (máx 15° para cuidar el eje clínico)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# El set de Validación se queda estricto (Refleja la realidad médica pura)
transformacion_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================================
# 3. CLASE DATASET ADAPTADA
# ==========================================================
class HAM10000DatasetCorregido(Dataset):
    def __init__(self, dataframe, image_paths, transform=None):
        self.df = dataframe
        self.image_paths = image_paths
        self.transform = transform
        self.label_mapping = MAPEO_FIJO_CLASES
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        label = self.label_mapping[row['dx']] # Asignación exacta
        
        img_path = self.image_paths[img_id]
        imagen = Image.open(img_path).convert('RGB')
        
        if self.transform:
            imagen = self.transform(imagen)
            
        return imagen, label

# ==========================================================
# 4. DIVISIÓN PROPORCIONAL Y ESTRATIFICADA (80% Train, 20% Val)
# ==========================================================
df_train, df_val = train_test_split(
    df, 
    test_size=0.20, 
    random_state=42, 
    stratify=df['dx'] # Mantiene los mismos porcentajes de enfermedades en ambos sets
)

# ==========================================================
# 5. INSTANCIAR LOS CARGADORES OFICIALES
# ==========================================================
# Pasamos 'transformacion_train' al entrenamiento y 'transformacion_val' a la validación
dataset_train = HAM10000DatasetCorregido(dataframe=df_train, image_paths=image_path_mapping, transform=transformacion_train)
dataset_val = HAM10000DatasetCorregido(dataframe=df_val, image_paths=image_path_mapping, transform=transformacion_val)

BATCH_SIZE = 32
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Guardamos la lista oficial de nombres basada en el mapeo inmutable para sklearn
nombres_clases = list(MAPEO_FIJO_CLASES.keys())

# ==========================================================
# RESUMEN DE CONFIGURACIÓN DE DATOS (para comparar después con el SVM)
# ==========================================================
dist_original = df['dx'].value_counts().to_dict()
dist_train    = df_train['dx'].value_counts().to_dict()
dist_val      = df_val['dx'].value_counts().to_dict()

print("=" * 60)
print("CONFIGURACIÓN DE DATOS")
print("=" * 60)
print(f"Clases ({len(nombres_clases)})   : {nombres_clases}")
print(f"Total muestras       : {len(df)}")
print(f"Train / Val          : {len(df_train)} / {len(df_val)}")
print(f"Distribución original: {dist_original}")
print(f"Distribución train   : {dist_train}")
print(f"Distribución val     : {dist_val}")
print("Tamaño de imagen     : 224x224 px RGB")
print(f"Batch size           : {BATCH_SIZE}")
print("=" * 60)


import torch
import torch.nn as nn
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado: {device}")
class DenseNetMultitarea(nn.Module):
    def __init__(self, num_clases_especificas=7):
        super(DenseNetMultitarea, self).__init__()
        # 1. Cargar el cuerpo base de la DenseNet201
        pesos_preentrenados = models.DenseNet201_Weights.DEFAULT
        self.densenet = models.densenet201(weights=pesos_preentrenados)
        
        # Extraer los canales de entrada del clasificador original
        num_caracteristicas = self.densenet.classifier.in_features
        
        # 2. Convertir el clasificador original en un puente neutro
        self.densenet.classifier = nn.Identity()
        
        # 3. CABEZA A: Clasificación Binaria de Triaje (0: Benigno, 1: Maligno)
        self.cabeza_binaria = nn.Linear(num_caracteristicas, 2)
        
        # 4. CABEZA B: Clasificación Detallada (Las 7 patologías del TFM)
        self.cabeza_especifica = nn.Linear(num_caracteristicas, num_clases_especificas)
        
    def forward(self, x):
        # Extraer mapas de características globales
        caracteristicas = self.densenet(x)
        
        # Bifurcar el flujo en paralelo hacia ambas subtareas
        salida_binaria = self.cabeza_binaria(caracteristicas)
        salida_especifica = self.cabeza_especifica(caracteristicas)
        
        return salida_binaria, salida_especifica

# Inicializar y mover a tu GPU RTX 3060
model = DenseNetMultitarea(num_clases_especificas=7)
model = model.to(device)

n_parametros_entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros entrenables: {n_parametros_entrenables:,}")


import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================
# 1. BALANCEO MATEMÁTICO DE DATOS
# ==========================================================
conteo_clases = df_train['dx'].value_counts().to_dict()
muestras_por_clase = [conteo_clases[clase] for clase in nombres_clases]
total_muestras = sum(muestras_por_clase)
pesos_clases = [total_muestras / (len(muestras_por_clase) * cantidad) for cantidad in muestras_por_clase]

# Convertir a Tensor para la GPU
pesos_tensor = torch.FloatTensor(pesos_clases).to(device)

print("=== PESOS DE BALANCEO SINCRONIZADOS ===")
for clase, peso, cant in zip(nombres_clases, pesos_clases, muestras_por_clase):
    print(f"Enfermedad: {clase:<6} | Cantidad: {cant:<5} | Peso: {peso:.4f}")
print("\n")

# ==========================================================
# 2. CONFIGURACIÓN DE CRITERIOS Y SCHEDULER
# ==========================================================
# Pérdida ponderada para las 7 clases específicas
criterion_especifico = nn.CrossEntropyLoss(weight=pesos_tensor)

# Pérdida estándar para la decisión binaria (Benigno vs Maligno)
criterion_binario = nn.CrossEntropyLoss()

# Optimizador Adam acoplado a todo el nuevo grafo del modelo
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Scheduler para reducir el paso si la pérdida combinada se estanca
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)


import time

# Mapeo numérico automático basado en tus clases de entrenamiento:
# Modifica los índices si en tu MAPEO_FIJO_CLASES las posiciones cambiaron.
# Por defecto asume que akiec, bcc y mel son las patologías malignas.
def generar_etiquetas_binarias(etiquetas_especificas):
    # Condición booleana: 1 si es maligno, 0 si es benigno
    # Evaluamos basándonos en los índices estándar del array nombres_clases
    idx_akiec = nombres_clases.index('akiec')
    idx_bcc = nombres_clases.index('bcc')
    idx_mel = nombres_clases.index('mel')
    
    es_maligno = (etiquetas_especificas == idx_akiec) | (etiquetas_especificas == idx_bcc) | (etiquetas_especificas == idx_mel)
    return es_maligno.long().to(device)

EPOCHS = 30
print(f"Iniciando entrenamiento jerárquico por {EPOCHS} épocas...\n")

historial_train_loss = []
historial_train_acc = []
historial_val_loss = []
historial_val_acc = []

t_train_inicio = time.time()

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # ------------------------------------------------------
    # MODO ENTRENAMIENTO
    # ------------------------------------------------------
    model.train()
    running_loss = 0.0
    correct_especifico = 0
    total_muestras_train = 0
    
    for imagenes, etiquetas in train_loader:
        imagenes = imagenes.to(device)
        etiquetas_especificas = etiquetas.to(device)
        
        # Generar la máscara binaria clínica al vuelo
        etiquetas_binarias = generar_etiquetas_binarias(etiquetas_especificas)
        
        optimizer.zero_grad()
        
        # Doble salida en paralelo de la red
        pred_binaria, pred_especifica = model(imagenes)
        
        # Calcular pérdidas individuales
        loss_b = criterion_binario(pred_binaria, etiquetas_binarias)
        loss_e = criterion_especifico(pred_especifica, etiquetas_especificas)
        
        # Pérdida Conjunta Híbrida (50% de peso a cada tarea)
        loss_total = 0.5 * loss_b + 0.5 * loss_e
        
        loss_total.backward()
        optimizer.step()
        
        # Métricas de entrenamiento (enfocadas en la clasificación final de 7 clases)
        running_loss += loss_total.item() * imagenes.size(0)
        _, predicted = torch.max(pred_especifica, 1)
        total_muestras_train += etiquetas_especificas.size(0)
        correct_especifico += (predicted == etiquetas_especificas).sum().item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = (correct_especifico / total_muestras_train) * 100
    
    # ------------------------------------------------------
    # MODO VALIDACIÓN
    # ------------------------------------------------------
    model.eval()
    running_val_loss = 0.0
    correct_val_especifico = 0
    total_muestras_val = 0
    
    with torch.no_grad():
        for imagenes, etiquetas in val_loader:
            imagenes = imagenes.to(device)
            etiquetas_especificas = etiquetas.to(device)
            etiquetas_binarias = generar_etiquetas_binarias(etiquetas_especificas)
            
            pred_binaria, pred_especifica = model(imagenes)
            
            loss_b = criterion_binario(pred_binaria, etiquetas_binarias)
            loss_e = criterion_especifico(pred_especifica, etiquetas_especificas)
            loss_total = 0.5 * loss_b + 0.5 * loss_e
            
            running_val_loss += loss_total.item() * imagenes.size(0)
            _, predicted = torch.max(pred_especifica, 1)
            total_muestras_val += etiquetas_especificas.size(0)
            correct_val_especifico += (predicted == etiquetas_especificas).sum().item()
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_acc = (correct_val_especifico / total_muestras_val) * 100
    epoch_time = time.time() - start_time

    historial_train_loss.append(epoch_loss)
    historial_train_acc.append(epoch_acc)
    historial_val_loss.append(epoch_val_loss)
    historial_val_acc.append(epoch_val_acc)
    
    # Imprimir Reporte de la Época
    print(f"Época [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s)")
    print(f"  Train -> Loss: {epoch_loss:.4f} | Accuracy (7 Clases): {epoch_acc:.2f}%")
    print(f"  Val   -> Loss: {epoch_val_loss:.4f} | Accuracy (7 Clases): {epoch_val_acc:.2f}%")
    print("-" * 50)
    
    # Actualizar el Scheduler basándose en la pérdida combinada de validación
    scheduler.step(epoch_val_loss)

t_train_total = time.time() - t_train_inicio
print(f"Entrenamiento finalizado: {EPOCHS} épocas en {t_train_total/60:.1f} min ({t_train_total:.1f} s)")


# ==========================================================
# GUARDADO DE PESOS Y CONFIGURACIÓN PARA LA APLICACIÓN WEB
# ==========================================================
# Se guarda ahora (antes de la evaluación y de plt.show()) para no perder el
# entrenamiento si el proceso se interrumpe más adelante.
import json as _json
import datetime

dir_salida_peso = 'resultados_densenet_peso'
os.makedirs(dir_salida_peso, exist_ok=True)

ruta_pesos = os.path.join(dir_salida_peso, 'densenet201_multitarea.pth')
torch.save(model.state_dict(), ruta_pesos)   # pesos de la última época (los que se evalúan abajo)

config_app = {
    "arquitectura": "DenseNetMultitarea (DenseNet201 + cabeza_binaria + cabeza_especifica)",
    "archivo_pesos": "densenet201_multitarea.pth",
    "img_size": [224, 224],
    "normalizacion": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "clases": nombres_clases,                       # orden = índice de la cabeza B
    "mapeo_clases": MAPEO_FIJO_CLASES,
    "clases_malignas": ["akiec", "bcc", "mel"],     # definición usada para la cabeza A
    "cabeza_a_etiquetas": ["benigno", "maligno"],   # índice 0 / 1
    "epochs": EPOCHS,
    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}
with open(os.path.join(dir_salida_peso, 'config_modelo.json'), 'w', encoding='utf-8') as f:
    _json.dump(config_app, f, indent=2, ensure_ascii=False)

print(f"Pesos guardados en: {ruta_pesos} ({os.path.getsize(ruta_pesos)/1e6:.1f} MB)")


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

# ==========================================================
# 1. RECOLECTAR PREDICCIONES Y PROBABILIDADES DE AMBAS CABEZAS
# ==========================================================
model.eval()

etiquetas_reales_especificas = []
predicciones_especificas = []
probabilidades_especificas = []   # softmax, para AUC / PR (Cabeza B)

etiquetas_reales_binarias = []
predicciones_binarias = []
probabilidades_binarias = []      # prob. de la clase "maligno" (Cabeza A)

t_infer_inicio = time.time()
with torch.no_grad():
    for imagenes, etiquetas in val_loader:
        imagenes = imagenes.to(device)
        etiquetas_especificas = etiquetas.to(device)
        etiquetas_binarias = generar_etiquetas_binarias(etiquetas_especificas)

        pred_binaria, pred_especifica = model(imagenes)

        probas_especificas = torch.softmax(pred_especifica, dim=1)
        probas_binarias = torch.softmax(pred_binaria, dim=1)

        _, pred_clase_especifica = torch.max(pred_especifica, 1)
        _, pred_clase_binaria = torch.max(pred_binaria, 1)

        etiquetas_reales_especificas.extend(etiquetas_especificas.cpu().numpy())
        predicciones_especificas.extend(pred_clase_especifica.cpu().numpy())
        probabilidades_especificas.extend(probas_especificas.cpu().numpy())

        etiquetas_reales_binarias.extend(etiquetas_binarias.cpu().numpy())
        predicciones_binarias.extend(pred_clase_binaria.cpu().numpy())
        probabilidades_binarias.extend(probas_binarias[:, 1].cpu().numpy())

t_infer_total = time.time() - t_infer_inicio
t_infer_por_imagen_ms = t_infer_total / len(val_loader.dataset) * 1000

etiquetas_reales_especificas = np.array(etiquetas_reales_especificas)
predicciones_especificas     = np.array(predicciones_especificas)
probabilidades_especificas   = np.array(probabilidades_especificas)

etiquetas_reales_binarias = np.array(etiquetas_reales_binarias)
predicciones_binarias     = np.array(predicciones_binarias)
probabilidades_binarias   = np.array(probabilidades_binarias)

nombres_clases_binarias = ['benigno', 'maligno']


# ==========================================================
# 2. CABEZA B — DIAGNÓSTICO ESPECÍFICO (7 clases)
# ==========================================================
acc_b         = accuracy_score(etiquetas_reales_especificas, predicciones_especificas)
bal_acc_b     = balanced_accuracy_score(etiquetas_reales_especificas, predicciones_especificas)
f1_macro_b    = f1_score(etiquetas_reales_especificas, predicciones_especificas, average='macro', zero_division=0)
f1_weighted_b = f1_score(etiquetas_reales_especificas, predicciones_especificas, average='weighted', zero_division=0)
f1_micro_b    = f1_score(etiquetas_reales_especificas, predicciones_especificas, average='micro', zero_division=0)
prec_macro_b  = precision_score(etiquetas_reales_especificas, predicciones_especificas, average='macro', zero_division=0)
prec_w_b      = precision_score(etiquetas_reales_especificas, predicciones_especificas, average='weighted', zero_division=0)
rec_macro_b   = recall_score(etiquetas_reales_especificas, predicciones_especificas, average='macro', zero_division=0)
rec_w_b       = recall_score(etiquetas_reales_especificas, predicciones_especificas, average='weighted', zero_division=0)
kappa_b       = cohen_kappa_score(etiquetas_reales_especificas, predicciones_especificas)
mcc_b         = matthews_corrcoef(etiquetas_reales_especificas, predicciones_especificas)

y_bin_b = label_binarize(etiquetas_reales_especificas, classes=list(range(len(nombres_clases))))
auc_por_clase_b = {}
ap_por_clase_b = {}
for i, cls in enumerate(nombres_clases):
    try:
        auc_por_clase_b[cls] = roc_auc_score(y_bin_b[:, i], probabilidades_especificas[:, i])
    except Exception:
        auc_por_clase_b[cls] = float('nan')
    try:
        ap_por_clase_b[cls] = average_precision_score(y_bin_b[:, i], probabilidades_especificas[:, i])
    except Exception:
        ap_por_clase_b[cls] = float('nan')

try:
    auc_macro_b    = roc_auc_score(y_bin_b, probabilidades_especificas, average='macro', multi_class='ovr')
    auc_weighted_b = roc_auc_score(y_bin_b, probabilidades_especificas, average='weighted', multi_class='ovr')
except Exception:
    auc_macro_b = auc_weighted_b = float('nan')

reporte_b = classification_report(
    etiquetas_reales_especificas, predicciones_especificas,
    target_names=nombres_clases, output_dict=True, zero_division=0
)

cm_b = confusion_matrix(etiquetas_reales_especificas, predicciones_especificas)
cm_b_norm = cm_b.astype(float) / cm_b.sum(axis=1, keepdims=True)

print("\n" + "=" * 60)
print("CABEZA B — DIAGNÓSTICO ESPECÍFICO (7 clases)")
print("=" * 60)
print(classification_report(etiquetas_reales_especificas, predicciones_especificas,
                             target_names=nombres_clases, zero_division=0))
print(f"Accuracy             : {acc_b:.4f}")
print(f"Balanced Accuracy    : {bal_acc_b:.4f}")
print(f"Recall Macro         : {rec_macro_b:.4f}")
print(f"Recall Weighted      : {rec_w_b:.4f}")
print(f"Precision Macro      : {prec_macro_b:.4f}")
print(f"Precision Weighted   : {prec_w_b:.4f}")
print(f"F1 Macro             : {f1_macro_b:.4f}")
print(f"F1 Weighted          : {f1_weighted_b:.4f}")
print(f"F1 Micro             : {f1_micro_b:.4f}")
print(f"AUC Macro (OvR)      : {auc_macro_b:.4f}")
print(f"AUC Weighted (OvR)   : {auc_weighted_b:.4f}")
print(f"Cohen's Kappa        : {kappa_b:.4f}")
print(f"MCC                  : {mcc_b:.4f}")
print("=" * 60)


# ==========================================================
# 3. CABEZA A — TRIAJE BINARIO (Benigno vs. Maligno)
# ==========================================================
acc_a           = accuracy_score(etiquetas_reales_binarias, predicciones_binarias)
bal_acc_a       = balanced_accuracy_score(etiquetas_reales_binarias, predicciones_binarias)
f1_macro_a      = f1_score(etiquetas_reales_binarias, predicciones_binarias, average='macro', zero_division=0)
f1_weighted_a   = f1_score(etiquetas_reales_binarias, predicciones_binarias, average='weighted', zero_division=0)
prec_maligno_a  = precision_score(etiquetas_reales_binarias, predicciones_binarias, pos_label=1, zero_division=0)
rec_maligno_a   = recall_score(etiquetas_reales_binarias, predicciones_binarias, pos_label=1, zero_division=0)   # Sensibilidad
especificidad_a = recall_score(etiquetas_reales_binarias, predicciones_binarias, pos_label=0, zero_division=0)   # Especificidad
kappa_a         = cohen_kappa_score(etiquetas_reales_binarias, predicciones_binarias)
mcc_a           = matthews_corrcoef(etiquetas_reales_binarias, predicciones_binarias)

try:
    auc_a = roc_auc_score(etiquetas_reales_binarias, probabilidades_binarias)
except Exception:
    auc_a = float('nan')
try:
    ap_a = average_precision_score(etiquetas_reales_binarias, probabilidades_binarias)
except Exception:
    ap_a = float('nan')

reporte_a = classification_report(
    etiquetas_reales_binarias, predicciones_binarias,
    target_names=nombres_clases_binarias, output_dict=True, zero_division=0
)

cm_a = confusion_matrix(etiquetas_reales_binarias, predicciones_binarias)
cm_a_norm = cm_a.astype(float) / cm_a.sum(axis=1, keepdims=True)

print("\n" + "=" * 60)
print("CABEZA A — TRIAJE BINARIO (Benigno vs. Maligno)")
print("=" * 60)
print(classification_report(etiquetas_reales_binarias, predicciones_binarias,
                             target_names=nombres_clases_binarias, zero_division=0))
print(f"Sensibilidad (Recall clase Maligno)  : {rec_maligno_a:.4f}")
print(f"Especificidad (Recall clase Benigno) : {especificidad_a:.4f}")
print(f"Accuracy                             : {acc_a:.4f}")
print(f"Balanced Accuracy                    : {bal_acc_a:.4f}")
print(f"Precision (clase Maligno)            : {prec_maligno_a:.4f}")
print(f"F1 Macro                             : {f1_macro_a:.4f}")
print(f"F1 Weighted                          : {f1_weighted_a:.4f}")
print(f"AUC-ROC                              : {auc_a:.4f}")
print(f"Average Precision                    : {ap_a:.4f}")
print(f"Cohen's Kappa                        : {kappa_a:.4f}")
print(f"MCC                                  : {mcc_a:.4f}")
print("=" * 60)


# ==========================================================
# 4. TIEMPOS
# ==========================================================
print("\n" + "=" * 60)
print("TIEMPOS")
print("=" * 60)
print(f"Entrenamiento total    : {t_train_total:.1f} s ({t_train_total/60:.1f} min)")
print(f"Inferencia total (val) : {t_infer_total:.4f} s")
print(f"Inferencia por imagen  : {t_infer_por_imagen_ms:.4f} ms")
print("=" * 60)


# ==========================================================
# 5. GRÁFICOS
# ==========================================================

# --- 5.1 Matriz de confusión — Cabeza B (absoluta + normalizada) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Matriz de Confusión — Cabeza B: Diagnóstico Específico (7 clases)', fontsize=14, fontweight='bold')
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues',
            xticklabels=nombres_clases, yticklabels=nombres_clases, ax=ax1)
ax1.set_title('Conteos absolutos')
ax1.set_ylabel('Diagnóstico Real (Dermatólogo)')
ax1.set_xlabel('Predicción del Modelo (DenseNet201)')
sns.heatmap(cm_b_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=nombres_clases, yticklabels=nombres_clases, ax=ax2)
ax2.set_title('Normalizada (recall por clase)')
ax2.set_ylabel('Diagnóstico Real (Dermatólogo)')
ax2.set_xlabel('Predicción del Modelo (DenseNet201)')
plt.tight_layout()
plt.show()

# --- 5.2 Matriz de confusión — Cabeza A (absoluta + normalizada) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle('Matriz de Confusión — Cabeza A: Triaje Binario', fontsize=14, fontweight='bold')
sns.heatmap(cm_a, annot=True, fmt='d', cmap='Reds',
            xticklabels=nombres_clases_binarias, yticklabels=nombres_clases_binarias, ax=ax1)
ax1.set_title('Conteos absolutos'); ax1.set_ylabel('Real'); ax1.set_xlabel('Predicción')
sns.heatmap(cm_a_norm, annot=True, fmt='.2f', cmap='Reds',
            xticklabels=nombres_clases_binarias, yticklabels=nombres_clases_binarias, ax=ax2)
ax2.set_title('Normalizada'); ax2.set_ylabel('Real'); ax2.set_xlabel('Predicción')
plt.tight_layout()
plt.show()

# --- 5.3 Curvas ROC — Cabeza B (por clase + macro) ---
n_clases_b = len(nombres_clases)
paleta_b = sns.color_palette('husl', n_clases_b)
ncols = 4
nrows = (n_clases_b + 1 + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
fig.suptitle('Curvas ROC — Cabeza B (One-vs-Rest)', fontsize=14, fontweight='bold')
axes_flat = axes.flatten() if nrows > 1 else axes

for i, cls in enumerate(nombres_clases):
    fpr, tpr, _ = roc_curve(y_bin_b[:, i], probabilidades_especificas[:, i])
    ax = axes_flat[i]
    ax.plot(fpr, tpr, color=paleta_b[i], lw=2, label=f'AUC = {auc_por_clase_b[cls]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_title(cls, fontsize=10)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(loc='lower right', fontsize=8)

all_ax = axes_flat[n_clases_b]
for i, cls in enumerate(nombres_clases):
    fpr, tpr, _ = roc_curve(y_bin_b[:, i], probabilidades_especificas[:, i])
    all_ax.plot(fpr, tpr, color=paleta_b[i], lw=1.5, label=f'{cls} ({auc_por_clase_b[cls]:.3f})')
all_ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
all_ax.set_title(f'Todas las clases (AUC macro={auc_macro_b:.3f})', fontsize=10)
all_ax.legend(loc='lower right', fontsize=7)

for j in range(n_clases_b + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
plt.show()

# --- 5.4 Curvas Precision-Recall — Cabeza B (por clase) ---
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
fig.suptitle('Curvas Precision-Recall — Cabeza B', fontsize=14, fontweight='bold')
axes_flat = axes.flatten() if nrows > 1 else axes

for i, cls in enumerate(nombres_clases):
    prec, rec, _ = precision_recall_curve(y_bin_b[:, i], probabilidades_especificas[:, i])
    ax = axes_flat[i]
    ax.plot(rec, prec, color=paleta_b[i], lw=2, label=f'AP = {ap_por_clase_b[cls]:.3f}')
    baseline = y_bin_b[:, i].mean()
    ax.axhline(baseline, color='gray', linestyle='--', lw=0.8, label=f'Baseline={baseline:.2f}')
    ax.set_title(cls, fontsize=10)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.legend(loc='upper right', fontsize=8)

all_ax = axes_flat[n_clases_b]
for i, cls in enumerate(nombres_clases):
    prec, rec, _ = precision_recall_curve(y_bin_b[:, i], probabilidades_especificas[:, i])
    all_ax.plot(rec, prec, color=paleta_b[i], lw=1.5, label=f'{cls} ({ap_por_clase_b[cls]:.3f})')
all_ax.set_title('Todas las clases', fontsize=10)
all_ax.legend(loc='upper right', fontsize=7)

for j in range(n_clases_b + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.tight_layout()
plt.show()

# --- 5.5 Curva ROC + PR — Cabeza A (binaria) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Cabeza A — Triaje Binario (Benigno vs. Maligno)', fontsize=14, fontweight='bold')

fpr_a, tpr_a, _ = roc_curve(etiquetas_reales_binarias, probabilidades_binarias)
ax1.plot(fpr_a, tpr_a, color='crimson', lw=2, label=f'AUC = {auc_a:.3f}')
ax1.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax1.set_title('Curva ROC'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR (Sensibilidad)')
ax1.legend(loc='lower right')

prec_a_curva, rec_a_curva, _ = precision_recall_curve(etiquetas_reales_binarias, probabilidades_binarias)
ax2.plot(rec_a_curva, prec_a_curva, color='crimson', lw=2, label=f'AP = {ap_a:.3f}')
baseline_a = etiquetas_reales_binarias.mean()
ax2.axhline(baseline_a, color='gray', linestyle='--', lw=0.8, label=f'Baseline={baseline_a:.2f}')
ax2.set_title('Curva Precision-Recall'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()

# --- 5.6 Métricas por clase — Cabeza B (barras) ---
metricas_por_clase_df = pd.DataFrame({
    'Precision': [reporte_b[c]['precision'] for c in nombres_clases],
    'Recall'   : [reporte_b[c]['recall']    for c in nombres_clases],
    'F1-Score' : [reporte_b[c]['f1-score']  for c in nombres_clases],
    'AUC'      : [auc_por_clase_b.get(c, 0) for c in nombres_clases],
}, index=nombres_clases)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Métricas por Clase — Cabeza B (DenseNet201)', fontsize=14, fontweight='bold')
axes_flat = axes.flatten()
for ax, metric in zip(axes_flat, ['Precision', 'Recall', 'F1-Score', 'AUC']):
    vals = metricas_por_clase_df[metric]
    bars = ax.bar(nombres_clases, vals, color=paleta_b)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis='x', rotation=45)
    mu = vals.mean()
    ax.axhline(mu, color='red', linestyle='--', alpha=0.6, label=f'Media: {mu:.3f}')
    ax.legend(fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.show()

# --- 5.7 Curvas de entrenamiento (loss y accuracy por época) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Curvas de Entrenamiento — DenseNet201', fontsize=14, fontweight='bold')
rango_epocas = range(1, EPOCHS + 1)
ax1.plot(rango_epocas, historial_train_loss, label='Train', marker='o', ms=3)
ax1.plot(rango_epocas, historial_val_loss, label='Val', marker='o', ms=3)
ax1.set_title('Pérdida combinada (Loss)'); ax1.set_xlabel('Época'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(rango_epocas, historial_train_acc, label='Train', marker='o', ms=3)
ax2.plot(rango_epocas, historial_val_acc, label='Val', marker='o', ms=3)
ax2.set_title('Accuracy (7 clases)'); ax2.set_xlabel('Época'); ax2.set_ylabel('Accuracy (%)')
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================================
# 6. EXPORTAR MÉTRICAS A JSON (para comparar después con el SVM)
# ==========================================================
resultados_json = {
    "modelo": "DenseNet201_Multitask",
    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "dataset": "HAM10000_metadata_balanceo.csv",
    "img_size": [224, 224],
    "n_train": int(len(df_train)),
    "n_val": int(len(df_val)),
    "n_clases": len(nombres_clases),
    "clases": nombres_clases,
    "epochs": EPOCHS,
    "parametros_entrenables": int(n_parametros_entrenables),
    "cabeza_b_7_clases": {
        "accuracy": round(float(acc_b), 6),
        "balanced_accuracy": round(float(bal_acc_b), 6),
        "f1_macro": round(float(f1_macro_b), 6),
        "f1_weighted": round(float(f1_weighted_b), 6),
        "f1_micro": round(float(f1_micro_b), 6),
        "precision_macro": round(float(prec_macro_b), 6),
        "precision_weighted": round(float(prec_w_b), 6),
        "recall_macro": round(float(rec_macro_b), 6),
        "recall_weighted": round(float(rec_w_b), 6),
        "auc_macro": round(float(auc_macro_b), 6),
        "auc_weighted": round(float(auc_weighted_b), 6),
        "cohen_kappa": round(float(kappa_b), 6),
        "mcc": round(float(mcc_b), 6),
        "por_clase": {
            cls: {
                "precision": round(float(reporte_b[cls]['precision']), 6),
                "recall": round(float(reporte_b[cls]['recall']), 6),
                "f1_score": round(float(reporte_b[cls]['f1-score']), 6),
                "support": int(reporte_b[cls]['support']),
                "auc": round(float(auc_por_clase_b.get(cls, float('nan'))), 6),
                "avg_precision": round(float(ap_por_clase_b.get(cls, float('nan'))), 6),
            }
            for cls in nombres_clases
        },
    },
    "cabeza_a_binaria": {
        "accuracy": round(float(acc_a), 6),
        "balanced_accuracy": round(float(bal_acc_a), 6),
        "sensibilidad_maligno": round(float(rec_maligno_a), 6),
        "especificidad_benigno": round(float(especificidad_a), 6),
        "precision_maligno": round(float(prec_maligno_a), 6),
        "f1_macro": round(float(f1_macro_a), 6),
        "f1_weighted": round(float(f1_weighted_a), 6),
        "auc_roc": round(float(auc_a), 6),
        "average_precision": round(float(ap_a), 6),
        "cohen_kappa": round(float(kappa_a), 6),
        "mcc": round(float(mcc_a), 6),
        "por_clase": {
            cls: {
                "precision": round(float(reporte_a[cls]['precision']), 6),
                "recall": round(float(reporte_a[cls]['recall']), 6),
                "f1_score": round(float(reporte_a[cls]['f1-score']), 6),
                "support": int(reporte_a[cls]['support']),
            }
            for cls in nombres_clases_binarias
        },
    },
    "tiempos_segundos": {
        "entrenamiento_total": round(float(t_train_total), 3),
        "inferencia_total_val": round(float(t_infer_total), 4),
        "inferencia_por_imagen_ms": round(float(t_infer_por_imagen_ms), 4),
    },
    "configuracion_modelo": {
        "arquitectura": "DenseNet201 (ImageNet, pesos por defecto) + 2 cabezas lineales",
        "optimizador": "Adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "scheduler": "ReduceLROnPlateau(factor=0.1, patience=3)",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "peso_perdida_binaria": 0.5,
        "peso_perdida_especifica": 0.5,
        "device": str(device),
    },
    "historial_entrenamiento": {
        "train_loss": [round(float(v), 6) for v in historial_train_loss],
        "train_acc": [round(float(v), 4) for v in historial_train_acc],
        "val_loss": [round(float(v), 6) for v in historial_val_loss],
        "val_acc": [round(float(v), 4) for v in historial_val_acc],
    },
}

output_dir = 'resultados_densenet_peso'
os.makedirs(output_dir, exist_ok=True)
json_path = os.path.join(output_dir, 'metricas_densenet201.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(resultados_json, f, indent=2, ensure_ascii=False)


# ==========================================================
# 7. RESUMEN FINAL
# ==========================================================
print("\n" + "=" * 60)
print("RESUMEN FINAL (Recall/Sensibilidad como criterio principal)")
print("=" * 60)
print(f"Cabeza A (binario)  -> Sensibilidad: {rec_maligno_a:.4f} | Especificidad: {especificidad_a:.4f} | "
      f"Accuracy: {acc_a:.4f} | AUC: {auc_a:.4f}")
print(f"Cabeza B (7 clases) -> Recall Macro: {rec_macro_b:.4f} | Recall Weighted: {rec_w_b:.4f} | "
      f"Accuracy: {acc_b:.4f} | F1 Macro: {f1_macro_b:.4f} | AUC Macro: {auc_macro_b:.4f}")
print(f"Entrenamiento: {t_train_total/60:.1f} min | Inferencia/imagen: {t_infer_por_imagen_ms:.3f} ms")
print(f"Métricas exportadas a: {json_path}")
print("=" * 60)
