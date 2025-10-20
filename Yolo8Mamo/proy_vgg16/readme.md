# Proyecto VGG16

Este repositorio contiene un proyecto de entrenamiento intentando implementar un modelo replicando las técnicas de VGG16, usando imágenes del CBIS del DDSM.
A continuación se detalla la estructura de carpetas y archivos.

## 📁 configs/
Contiene los archivos para la configuración del entrenamiento.


- **base_config.yaml**: Archivo principal de configuración (rutas, parámetros de entrenamiento, etc.)
- **load_config.py**: Script para cargar y procesar los parámetros definidos en los archivos YAML.

---

## 📁 data/
Incluye los datos con los que se entrenará y los recursos utilizados para el procesamiento y análisis de dichos datos. 

- **bounding_boxes_fixed_trans.csv**: CSV en el que se encuentran los ground truth de las cajas delimitadoras (bounding boxes) de las imágenes de CBIS_DDSM.
- **classes_dataset.py/**: Definición y procesado del dataset y dataloader utilizados en el entrenamiento.
- **histograma_tamaño_grupos.png**: Histograma con la distribución de tamaños de grupos de las cajas delimitadoras de las imágenes. 

---

## 📁 models/
Contiene las definiciones de los distintos modelos utilizados en el proyecto.

- **backbone_vgg16.py**: Implementación del backbone de la arquitectura base VGG16, utilizado para definir el modelo. 
- **classes_model.py**: Definición del modelo completo (detector FasterCNN basado en VGG16).

---

## 📁 train/
Scripts relacionados con el entrenamiento del modelo 

-**train_model_vgg16.py**: Script que lanza el entrenamiento del modelo.
-**callbacks**: Script de python con la descripción de los callbacks usados en el entrenamiento.


---

## 📁 tests/
Scripts y notebooks para la validación y prueba de aspectos del entrenamiento. 

-**test_datamodule.ipynb**: Notebook de prueba para el datamodule y la visualización de imágenes de entrenamiento.

---


## 📁 model_checkpoints/
Carpeta donde se guardan los checkpoints del entrenamiento


---

## 📁 utils/
Funciones auxiliares necesarias en alguna parte del proyecto.

-**utils.py**: Script de python con las funciones auxiliares

---

## 📄 VGG16.v2.pretrained_weights
Archivo con los pesos preentrenados del modelo VGG16.

---