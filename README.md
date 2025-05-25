
# Segmentación de Imágenes con K-Means y Comparación con Máscaras

Este proyecto aplica segmentación de imágenes usando el algoritmo de clustering **K-Means** y compara los resultados con **máscaras reales** de segmentación. El objetivo es identificar, por cada imagen, el **cluster que más se asemeje a la región esperada** (por ejemplo, una zona afectada por enfermedad).

---

## 📂 Estructura esperada del proyecto

```
project/
├── images/                  # Contiene imágenes .jpg
│   ├── 00001_0.jpg
│   ├── ...
├── masks/                   # Contiene máscaras reales .png
│   ├── 00001_0.png
│   ├── ...
├── output_best_clusters/    # Salida con segmentaciones binarias
│   ├── 00001_0_cluster2_iou0.74.png
│   ├── ...
├── resumen_resultados.csv   # Archivo resumen con métricas IoU
└── project.py               # Script principal
```

---

## 🚀 Cómo ejecutar

1. Asegúrate de tener instaladas las dependencias:

```bash
pip install opencv-python numpy scikit-learn matplotlib
```

2. Estructura tus carpetas `images/` y `masks/` como se describe arriba.

3. Ejecuta el script:

```bash
python project.py
```

---

## 🧠 ¿Qué hace el script?

- Aplica **K-Means (k=10)** a cada imagen RGB.
- Compara cada uno de los clusters con su **máscara binaria** usando **IoU (Jaccard Score)**.
- Selecciona el **cluster más similar a la región segmentada real**.
- Guarda una **máscara binaria del cluster seleccionado** en `output_best_clusters/`.
- Genera un archivo `resumen_resultados.csv` con:

| Imagen    | Cluster | IoU    | Ruta de salida                             |
|-----------|---------|--------|--------------------------------------------|
| 00001_0   | 2       | 0.7451 | output_best_clusters/00001_0_cluster2_... |

---

## 📊 Métrica usada

- **IoU (Intersection over Union)** entre la máscara real y la predicha para determinar la mejor coincidencia de cluster.
- Los valores de la máscara esperada deben ser **binarios** (`0` y `valor_objetivo`, por defecto `38`).

---

## ✏️ Notas

- Puedes cambiar el valor objetivo (por defecto `38`) si tus máscaras usan otros niveles de gris.
- El script guarda **solo la mejor máscara generada por imagen**, no todos los clusters.
- Funciona en lote para múltiples imágenes.

---

## 📬 Contacto

Dierick Salvador Brochero  
`dierickbr@gmail.com`  
