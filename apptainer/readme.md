Esta carpeta emplea apptainer para crear una imagen de Faster R-CNN empleando Caffe como en 
el reto. 

Para crear la imagen hacer:

``` bash
apptainer build mi_imagen.sif container.def
```
Una vez creada el script: `start_inference_apptainer.sh` inicia una instancia. 

Una vez dentro de de la instancia ir a la carpeta faster_rcnn_vgg16/scoring para ejecutar el programa de deteccion con Caffe
