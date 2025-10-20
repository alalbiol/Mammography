import pandas as pd

# def transform_annotations_csv(input_csv_path, output_csv_path):
#     """
#     Transforma un CSV de anotaciones del formato (id, x, y, w, h, label, group) 
#     al formato (id, minx, miny, maxx, maxy, pathology, lesion_type, cancer, group).
#     Asume que (x, y) es la esquina inferior izquierda y que el eje Y apunta hacia arriba.

#     Args:
#         input_csv_path (str): Ruta al archivo CSV de entrada.
#         output_csv_path (str): Ruta donde se guardará el nuevo archivo CSV.
#     """
#     try:
#         df = pd.read_csv(input_csv_path)
#     except FileNotFoundError:
#         print(f"Error: No se encontró el archivo de entrada en '{input_csv_path}'")
#         return

#     # 1. Calcular minx, miny, maxx, maxy
#     # Dado que (x, y) es la esquina inferior izquierda y el eje Y apunta hacia arriba:
#     df['minx'] = df['x']
#     df['miny'] = df['y']
#     df['maxx'] = df['x'] + df['w']
#     df['maxy'] = df['y'] + df['h']

#     # 2. Mapear 'label' a 'pathology', 'lesion_type', 'cancer'
#     # Inicializar las nuevas columnas
#     df['pathology'] = ''
#     df['lesion_type'] = ''
#     df['cancer'] = -1 

#     # Aplicar las reglas de mapeo
#     df.loc[df['label'] == 'CALCIFICATION_BENIGN', ['pathology', 'lesion_type', 'cancer']] = ['BENIGN', 'calcification', 0]
#     df.loc[df['label'] == 'MASS_BENIGN', ['pathology', 'lesion_type', 'cancer']] = ['BENIGN', 'mass', 0]
#     df.loc[df['label'] == 'CALCIFICATION_MALIGNANT', ['pathology', 'lesion_type', 'cancer']] = ['MALIGNANT', 'calcification', 1]
#     df.loc[df['label'] == 'MASS_MALIGNANT', ['pathology', 'lesion_type', 'cancer']] = ['MALIGNANT', 'mass', 1]

#     # 3. Seleccionar y reordenar las columnas finales
#     # AÑADIR 'group' aquí a la lista de columnas para la salida
#     output_df = df[['id', 'minx', 'miny', 'maxx', 'maxy', 'pathology', 'lesion_type', 'cancer', 'group']]

#     # 4. Guardar el nuevo CSV
#     output_df.to_csv(output_csv_path, index=False)
#     print(f"CSV transformado guardado exitosamente en '{output_csv_path}' 🎉")

# if __name__ == "__main__":
#     input_csv_file = '/home/Data/CBIS-DDSM-segmentation-2240x1792/bounding_boxes.csv' 
#     output_csv_file = '/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/pruebas_lucia/bboxes_ini_trans.csv'

#     transform_annotations_csv(input_csv_file, output_csv_file)




def transform_annotations_csv(input_csv_path, output_csv_path):
    """
    Transforma un CSV de anotaciones del formato (id, x, y, w, h, label, group) 
    al formato (id, minx, miny, maxx, maxy, pathology, lesion_type, cancer, group).
    Asume que (x, y) es la esquina inferior izquierda y que el eje Y apunta hacia arriba.

    Args:
        input_csv_path (str): Ruta al archivo CSV de entrada.
        output_csv_path (str): Ruta donde se guardará el nuevo archivo CSV.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de entrada en '{input_csv_path}'")
        return

    # 1. Calcular minx, miny, maxx, maxy
    # Dado que (x, y) es la esquina inferior izquierda y el eje Y apunta hacia arriba:
    df['minx'] = df['x']
    df['miny'] = df['y']
    df['maxx'] = df['x'] + df['w']
    df['maxy'] = df['y'] + df['h']

    df['cancer'] = -1 

    # Aplicar las reglas de mapeo
    df.loc[df['label'] == 'CALCIFICATION_BENIGN', ['label', 'cancer']] = ['benign', 0]
    df.loc[df['label'] == 'MASS_BENIGN', ['label', 'cancer']] = ['benign', 0]
    df.loc[df['label'] == 'CALCIFICATION_MALIGNANT', ['label', 'cancer']] = ['malignant', 1]
    df.loc[df['label'] == 'MASS_MALIGNANT', ['label', 'cancer']] = ['malignant', 1]

    # 3. Seleccionar y reordenar las columnas finales
    # AÑADIR 'group' aquí a la lista de columnas para la salida
    output_df = df[['id', 'minx', 'miny', 'maxx', 'maxy', 'w', 'h', 'label', 'cancer', 'group']]

    # 4. Guardar el nuevo CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"CSV transformado guardado exitosamente en '{output_csv_path}'")

if __name__ == "__main__":
    input_csv_file = '/home/Data/CBIS-DDSM-segmentation-2240x1792/bounding_boxes.csv'
    output_csv_file = '/home/lloprib/proyecto_mam/Mammography/Yolo8Mamo/proy_vgg16/data/bounding_boxes_trans.csv'
    transform_annotations_csv(input_csv_file, output_csv_file)