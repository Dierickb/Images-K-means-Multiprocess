import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm # Para una barra de progreso visual

# --- CONFIGURACIÓN ---
images_dir = os.path.join(os.getcwd(), "images")
masks_dir = os.path.join(os.getcwd(), "masks")
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

k = 10  # Número de clusters para K-Means
target_value = 38  # Valor de interés en la máscara (enfermedad, hoja, etc.)

def process_single_image(file_name_and_paths):
    img_id, img_path, mask_path = file_name_and_paths

    if not os.path.exists(mask_path):
        return None

    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    binary_mask = np.equal(mask, target_value).astype(np.uint8).flatten()
    pixel_values = image_rgb.reshape((-1, 3)).astype(np.float32)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixel_values)
    image_shape = image_rgb.shape

    best_score = 0
    best_cluster = -1

    for cluster_id in range(k):
        cluster_mask = np.equal(labels, cluster_id).astype(np.uint8)
        score = jaccard_score(binary_mask, cluster_mask)

        if score > best_score:
            best_score = score
            best_cluster = cluster_id

    best_mask_bool = np.equal(labels, best_cluster).reshape(image_shape[:2])
    best_cluster_image = np.zeros_like(image_rgb)
    best_cluster_image[best_mask_bool] = image_rgb[best_mask_bool]

    output_path = os.path.join(output_dir, f"{img_id}_cluster{best_cluster}_iou{best_score:.2f}.png")
    cv2.imwrite(output_path, cv2.cvtColor(best_cluster_image, cv2.COLOR_RGB2BGR))

    return (img_id, best_cluster, round(best_score, 4), output_path, True) # True indica éxito

if __name__ == "__main__":

    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    total_images_in_directory = len(image_files)

    tasks = []
    for file_name in image_files:
        img_id = os.path.splitext(file_name)[0]
        img_path = os.path.join(images_dir, file_name)
        mask_path = os.path.join(masks_dir, f"{img_id}.png")
        tasks.append((img_id, img_path, mask_path))

    results = []
    total_score = 0
    processed_count = 0

    print("Iniciando procesamiento paralelo...")
    num_processes = cpu_count()
    print(f"Usando {num_processes} procesos.")

    with Pool(processes=num_processes) as pool:
        # Usamos tqdm para tener una barra de progreso mientras se procesan las imágenes
        for res in tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks), desc="Procesando imágenes"):
            if res is not None: # Si el procesamiento fue exitoso
                img_id, best_cluster, score, output_path, success = res
                results.append((img_id, best_cluster, score, output_path))
                total_score += score
                processed_count += 1
            # else: # Aquí podríamos manejar los errores retornados por process_single_image

    average_score = total_score / processed_count if processed_count > 0 else 0
    percentage_processed = (processed_count / total_images_in_directory) * 100 if total_images_in_directory > 0 else 0

    # Mostrar resumen
    print("\n✅ PROCESAMIENTO COMPLETADO")
    print(f"📸 Total de imágenes JPG encontradas: {total_images_in_directory}")
    print(f"📸 Total de imágenes procesadas exitosamente: {processed_count}")
    print(f"📊 IoU promedio: {average_score:.4f}")
    print(f"📈 Porcentaje total de imágenes procesadas: {percentage_processed:.2f}%")

    # Guardar resumen en CSV
    summary_csv = os.path.join(os.getcwd(), "resumen_resultados.csv")
    with open(summary_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Imagen", "Cluster", "IoU", "Ruta de salida"])
        writer.writerows(results)

    print(f"📁 Resumen guardado en: {summary_csv}")