Esta carpeta permite ejecutar el modelo de dezso original con caffe.

Tienes que estar dentro de una instancia de apptainer (ver carpeta ../../apptainer)

La idea es que se ha modificado infer.py para guardar activaciones intermedias para poder hacer debug.

Simplemente ejecutar infer.sh para que se ejecute sobre una imagen y podamos ver los resultados intermedios.
Solo hace una imagen porque esta el modo express: ver linea 52 de infer.py