# Añadir al FINAL del script de entrenamiento (tras el bucle de épocas),
# o ejecutar en la misma sesión donde sigue vivo el objeto `model`.
import os, torch

os.makedirs('resultados_densenet', exist_ok=True)
torch.save(model.state_dict(), 'resultados_densenet/densenet201_multitarea.pth')
print("Pesos guardados.")
