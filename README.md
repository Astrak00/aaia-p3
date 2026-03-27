# Practica 3 - IA en Salud

Autores:
- Edardo Alarcón Navarro - 100472175@alumnos.uc3m.es
- Gabriel Gómez García - 100566646@alumnos.uc3m.es
- Eduardo Polo Peyres - 100567004@alumnos.uc3m.es
- Maria Isabel Ruiz Martínez - 100541735@alumnos.uc3m.es

Este repositorio contiene cuatro notebooks de trabajo, dos para clasificacion, uno para interpretabilidad y otro para segmentacion de imagenes medicas.


# 1. Requisitos y preparacion del entorno

### Opcion A (recomendada): usar `uv`

```bash
uv sync
uv run python -m ipykernel install --user --name aaia-p3 --display-name "Python (aaia-p3)"
uv run jupyter lab
```

### Opcion B: `venv` + `pip`

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install ipykernel jupyterlab
pip install torch torchvision scikit-learn tqdm
pip install albumentations opencv-python matplotlib seaborn pandas grad-cam ipywidgets
```

# 2. Estructura esperada de datos

Cada notebook espera una estructura de datos concreta en el directorio raiz del proyecto.
Las carpetas para clasificacion deben tener subcarpetas por clase, mientras que para segmentacion se espera una estructura con imagenes y mascaras en la misma carpeta.

### Clasificacion e interpretabilidad

- `4864-train/` y `1216-test/` con subcarpetas por clase:
	- `0-noDR`
	- `1-mild`
	- `2-moderate`
	- `3-severe`
	- `4-proliferativeDR`

### Clasificacion revisada

- En fases iniciales utiliza rutas tipo:
	- `fotos_raw_train/`
	- `fotos_raw_test/`
- y cada una con subcarpetas por clase (0-4):
	- `0-noDR`
	- `1-mild`
	- `2-moderate`
	- `3-severe`
	- `4-proliferativeDR`
- En fases posteriores usa train/test ya procesados (el propio notebook documenta la ruta exacta segun la fase).

### Segmentacion

- Dataset en:
	- `dataset/data1` o `dataset/data2`
- Cada carpeta debe contener:
	- imagenes `.bmp`
	- mascaras `.png`

# 3. Guia de uso por notebook

## Clasificación: `p3_clasificación.ipynb`

### ¿Qué hace?

- Pipeline de clasificacion de retinopatia diabetica.
- Incluye clasificacion binaria y multiclase.
- Entrena modelos basados en EfficientNet y guarda checkpoints.

### ¿Cómo usarlo?

1. Ejecuta todas las celdas de instalacion/imports.
2. Revisa la celda de configuracion (`Config`) y ajusta:
	 - rutas de train/test
	 - `TASK` (`binary` o `multiclass`)
	 - `SMOKE_TEST` (True para pruebas rapidas)
3. Ejecuta el preprocesamiento y construccion de datasets.
4. Entrena el modelo.
5. Ejecuta celdas de evaluacion (metricas, curvas, matrices de confusion).

### Salidas esperadas

- Modelos en `checkpoints/`.
- Figuras y metricas de evaluacion en salida del notebook.

## Clasificación v2: `p3_clasificación_revisado.ipynb`

### ¿Qué hace?

- Version ampliada/revisada del pipeline de clasificacion.
- Incluye varias fases y comparativa de arquitecturas.
- Contiene ajustes para GPU/XPU y refinamientos de entrenamiento.

### ¿Cómo usarlo?

1. Ejecuta primero la seccion de entorno y deteccion de dispositivo.
2. Verifica rutas de datos de la fase activa.
3. Revisa hiperparametros por fase antes de lanzar entrenamientos largos.
4. Ejecuta por bloques (Fase 1 -> Fase 2 -> Fase 3), no todo de golpe.
5. Al final, ejecuta las celdas de evaluacion global y guardado de modelo.

### Salidas esperadas

- Checkpoints por experimento/fase.
- Graficas comparativas entre modelos.
- Tablas resumen de rendimiento.

## Interpretabilidad: `p3_interpretabilidad.ipynb`

### ¿Qué hace?

- Analiza la interpretabilidad de los modelos de clasificacion.
- Compara metodos:
	- Grad-CAM
	- Grad-CAM++
	- Integrated Gradients

### Requisito previo

- Tener checkpoints entrenados disponibles en `checkpoints/`:
	- `best_model_binary.pth`
	- `best_model_multiclass.pth`

### ¿Cómo usarlo?

1. Ejecuta instalacion/imports.
2. Ajusta rutas (`TRAIN_DIR`, `TEST_DIR`, `CHECKPOINT_DIR`).
3. Carga modelos.
4. Ejecuta secciones de comparacion CAM y luego Integrated Gradients.
5. Prueba diferentes carpetas (`0-noDR`, `1-mild`, etc.) para comparar explicaciones.

### Salidas esperadas

- Overlays de activacion sobre imagenes de retina.
- Comparativas visuales por metodo y por clase.
- Conclusiones cualitativas sobre sesgos y regiones relevantes.

## Segmentacion: `p3_segmentación.ipynb`

### ¿Qué hace?

- Segmentacion de celulas sanguineas.
- Entrena y compara 5 arquitecturas:
	- UNet
	- Custom ResNet-based
	- U-Net++
	- UNetV2 (Attention U-Net)
	- ResUNet

### ¿Cómo usarlo?

1. Configura `CONFIG` al inicio:
	 - dataset (`1` o `2`)
	 - `img_size`, `batch_size`, `epochs`
2. Ejecuta carga de datos en memoria.
3. Verifica visualizaciones iniciales de imagen/mascara.
4. Ejecuta entrenamiento de todos los modelos (bloque largo).
5. Ejecuta evaluacion final y comparativas (ROC, curvas, resultados visuales).

### Salidas esperadas

- Pesos de mejor modelo por arquitectura (`*_best.pth`).
- Metricas (Accuracy, IoU, Dice, AUC).
- Curvas y comparativas visuales de prediccion.

# 4. Orden recomendado de ejecucion

Si quieres reproducir el flujo completo del proyecto:

1. `p3_clasificación.ipynb` (entrenamiento principal)
2. `p3_interpretabilidad.ipynb` (explicabilidad sobre checkpoints)

De manera independiente, podemos ejecutar la segmentacion sin necesidad de pasar por clasificacion:
1. `p3_segmentación.ipynb` (linea independiente de segmentacion)

`p3_clasificación_revisado.ipynb` puede usarse como pipeline alternativo o referencia de la version inicial, en la que se obtienen mejores resultados.