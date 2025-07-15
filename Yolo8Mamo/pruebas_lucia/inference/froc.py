import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


detections_csv = '/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/inference/detections.csv' # Ruta del archivo CSV con las detecciones
ground_truth_csv = '/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/inference/bounding_boxes_good.csv' # Ruta del archivo CSV con las ground truth
output_plot = 'froc_curve.png' # Ruta de la carpeta donde se guardará el gráfico FROC de salida

# La columna 'cancer' indica si el maligno o benigno. Asumimos cancer = 1 como maligno y cancer = 0 como benigno
MALIGNANCY_INDICATOR = 1

# Función para calcular el centro de la caja predicha
def calculate_center(xmin, ymin, xmax, ymax):
    # Suponemos que fastercnn devuelve cajas en formato [xmin, ymin, xmax, ymax], de manera que si no es así  habría que 
    # cambiar esta parte
    return (xmin + xmax) / 2, (ymin + ymax) / 2


# Función para comprobar si el centro de la caja predicha está dentro de la caja de ground truth 
# (suponiendo que las cajas están en formato [xmin, ymin, xmax, ymax])
def is_center_inside_box(center, xmin, ymin, xmax, ymax):
    cx, cy = center
    return xmin <= cx <= xmax and ymin <= cy <= ymax


# Función para obtener todas las regiones malignas verdaderas del dataset (y que sean únicas)
def get_gt_regions(ground_truth_df, malignancy_indicator_value):

    """
    Identifica todas las regiones malignas de ground truth únicas en el dataset completo.
    Retorna un diccionario mapeando 'image_id' a una lista de cajas GT malignas para esa imagen.
    Y también el número total de regiones malignas únicas.
    """

    # Filtrar solo las regiones de Ground Truth donde 'cancer' es igual a MALIGNANCY_INDICATOR
    malignant_gt = ground_truth_df[
        ground_truth_df['cancer'] == malignancy_indicator_value
    ].copy()
    
    gt_regions = {} # Para almacenar las regiones malignas agrupadas por id de imagen
    total_gt = 0 # Para contar el númer total de regiones malignas


    malignant_gt['gt_id'] = malignant_gt.groupby(['id', 'minx', 'miny', 'maxx', 'maxy']).ngroup()

    for image_id in malignant_gt['id'].unique():
        # Selecciona todas las ground truth malignas para la imagen actual
        image_gts = malignant_gt[malignant_gt['id'] == image_id]
        # Almacena las ground truths malignas en el diccionario
        gt_regions[image_id] = image_gts.to_dict(orient='records')
        # Incrementa el contador de regiones malignas
        total_gt += len(image_gts)
        
    return gt_regions, total_gt


# EJECUCIÓN PRINCIPAL DEL PROGRAMA

if __name__ == '__main__':
    all_threshold_data = [] # Para almacenar los valores de falsos positivos y sensitividad para cada umbral

    # 1. Cargar el Ground Truth una vez
    print("Cargando Ground Truth desde:", ground_truth_csv)
    try:
        ground_truth_df = pd.read_csv(ground_truth_csv)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de Ground Truth en '{ground_truth_csv}'.")
        exit()
    
    # Asegurarse de que las columnas de GT existen: 'id', 'minx', 'miny', 'maxx', 'maxy', 'cancer'
    required_gt_cols = ['id', 'minx', 'miny', 'maxx', 'maxy', 'cancer']
    if not all(col in ground_truth_df.columns for col in required_gt_cols):
        print(f"Error: El archivo '{ground_truth_csv}' no contiene todas las columnas GT requeridas ({required_gt_cols}).")
        exit()

    unique_malignant_gt_by_image, total_malignant_gt_regions = get_gt_regions(
        ground_truth_df, MALIGNANCY_INDICATOR
    )
    
    # 2. Cargar todas las detecciones una vez
    print("Cargando Detecciones desde:", detections_csv)
    try:
        all_detections_df = pd.read_csv(detections_csv)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de Detecciones en '{detections_csv}'.")
        exit()

    # Asegurarse de que las columnas de Detecciones existen: 'id', 'minx', 'miny', 'maxx', 'maxy', 'cancer', 'confidence'
    required_det_cols = ['id', 'box_x1', 'box_y1', 'box_x2', 'box_y2', 'score']
    if not all(col in all_detections_df.columns for col in required_det_cols):
        print(f"Error: El archivo '{detections_csv}' no contiene todas las columnas de detección requeridas ({required_det_cols}).\n Asegúrate de que 'confidence' esté presente.")
        exit()


    
    # Obtener el número total de imágenes evaluadas (asumimos que es el mismo para todos los CSVs)
    total_images = len(ground_truth_df['id'].unique())
    if total_images == 0:
        print("Error: No se encontraron imágenes en los datos.")
        exit()

    print(f"Total de imágenes evaluadas: {total_images}")
    print(f"Total de regiones malignas de GT en el dataset: {total_malignant_gt_regions}")


    # 3. Definir los umbrales para evaluar
    # Generamos un rango de umbrales de confianza de 0.0 a 1.0
    thresholds = np.arange(0.0, 1.01, 0.01) # Pasos de 0.01 para una curva suave

    # Iterar sobre cada umbral definido
    for threshold in thresholds:
        print(f"\nProcesando umbral: {threshold:.2f}") # Formato para 2 decimales

        # Filtrar solo las detecciones malignas (donde 'cancer' es MALIGNANCY_INDICATOR) que superan el umbral de confianza
        # malignant_preds = all_detections_df[
        #      all_detections_df['cancer'] == MALIGNANCY_INDICATOR].copy()
        # filtered_preds = malignant_preds[malignant_preds['score'] >= threshold].copy()

        filtered_preds = all_detections_df[all_detections_df['score'] >= threshold].copy() # Filtrar todas las detecciones que superan el umbral

        total_fps_for_threshold = 0 
        detected_malignant_gt_ids = set() # Para almacenar los IDs únicos de GT malignas detectadas


       


        # Procesar imágenes una por una para este umbral
        for image_id in filtered_preds['id'].unique():
            preds_for_image = filtered_preds[filtered_preds['id'] == image_id].sort_values(by='score', ascending=False)
            gt_for_image = unique_malignant_gt_by_image.get(image_id, []) # Obtener GT malignas para esta imagen

            # Queremos mantener un registro de las regiones de ground truth ya detectadas para evitar contar alguna más de una vez
            detected_gt_ids_in_image = set()

            for idx, pred_row in preds_for_image.iterrows():
                # Calcular el centro de la predicción usando las columnas minx, miny, maxx, maxy
                pred_center = calculate_center(pred_row['box_x1'], pred_row['box_y1'], pred_row['box_x2'], pred_row['box_y2'])

                is_tp = False
                for gt_dict in gt_for_image:
                    # Comprobar si el centro de la predicción está dentro de alguna de las cajas de ground truth malignas
                    gt_id = gt_dict.get('gt_id') 
                    if is_center_inside_box(pred_center, gt_dict['minx'], gt_dict['miny'], gt_dict['maxx'], gt_dict['maxy']):
                        # Esta predicción es un TP. Marcar la GT como detectada.
                        if gt_id not in detected_gt_ids_in_image:
                            detected_malignant_gt_ids.add(gt_id) # Se añade el id del ground truth como detectado
                            detected_gt_ids_in_image.add(gt_id) # Se añade el id del ground truth a la lista de detectados para esta imagen 
                            is_tp = True
                            break # Una predicción solo puede detectar una GT
                
                if not is_tp:
                    total_fps_for_threshold += 1 # Si no es un True Positive se cuenta como un Falso Positivo

        # Calcular métricas para este umbral
        num_detected_malignant_gt = len(detected_malignant_gt_ids)
        sensitivity = num_detected_malignant_gt / total_malignant_gt_regions if total_malignant_gt_regions > 0 else 0
        false_positives_per_image = total_fps_for_threshold / total_images if total_images > 0 else 0

        all_threshold_data.append({
            'threshold': threshold,
            'fps_per_image': false_positives_per_image,
            'sensitivity': sensitivity
        })

    # Representación de la curva FROC
    print("\nGenerando curva FROC...")
    
    # Ordenar los puntos para asegurar una curva correcta. Se ordena la lista all_threshold_data en función de los fps_per_image de cada punto, de menor a mayor
    froc_points = sorted(all_threshold_data, key=lambda x: x['fps_per_image'])
    
    fps_per_image_list = [p['fps_per_image'] for p in froc_points]
    sensitivity_list = [p['sensitivity'] for p in froc_points]

    plt.figure(figsize=(10, 7))
    plt.plot(fps_per_image_list, sensitivity_list, marker='o', linestyle='-', color='blue')
    plt.title('Curva FROC')
    plt.xlabel('Número Promedio de Falsos Positivos por Imagen')
    plt.ylabel('Sensibilidad a Nivel de Lesión')
    plt.grid(True)
    plt.xscale('log') # El eje X se escala logarítmicamente para una mejor visualización

    # # Opcional: Añadir un punto (0,0) si no está presente, común en FROC para indicar 0 FPs/imagen y 0 sensibilidad
    # if (0,0) not in zip(fps_per_image_list, sensitivity_list):
    #     plt.plot(0, 0, 'ro') # Punto rojo en el origen
    #     # Asegurarse de que el origen es visible
    #     plt.xlim(left=0) 
    #     plt.ylim(bottom=0)
        
    plt.tight_layout()
    plt.savefig(output_plot) # Para almacenar el gráfico
    print(f"Curva FROC guardada en: {output_plot}")

    from sklearn.metrics import auc
    
    # froc_auc = auc(fps_per_image_list, sensitivity_list)
    # print(f"\nÁrea Bajo la Curva FROC (AUFROC): {froc_auc:.4f} 🚀") # Formato a 4 decimales

    # plt.tight_layout()
    # plt.savefig(output_plot) # Para almacenar el gráfico
    # print(f"Curva FROC guardada en: {output_plot}")
    # print("¡Proceso completado! 🎉")

    scores_cancer = all_detections_df.groupby('id').agg({'score': 'max'})
    print("Numb of images in scores = ", len(scores_cancer))
    print(scores_cancer.head())


 
    # Calculate FROC curve
       # Calcular el área "bruta" bajo la curva FROC
    froc_auc = auc(fps_per_image_list, sensitivity_list)

    # Determinar el máximo valor de FPs/imagen alcanzado en tu curva
    max_fps_reached = max(fps_per_image_list)

    # Normalizar el AUFROC dividiendo por el área máxima posible (max_fps * 1)
    if max_fps_reached > 0:
        normalized_froc_auc = froc_auc / max_fps_reached
    else:
        normalized_froc_auc = 0.0 # Si no hay FPs, el área es 0

    print(f"\nÁrea Bajo la Curva FROC normalizada (AUFROC): {normalized_froc_auc:.4f} 🚀") # Formato a 4 decimales

