1. Carpeta 01-yolov8n-default
    * Opcion por defecto con mosaico...
    * Va fatal

2. Carpeta 02-yolov8n-sin-mosaic
    * sigue yendo mal, parece que hay problemas con el NMS muchos candidatos

3. Sigue sin funcionar, veo que hace demasiado zoom, igual en lugar de 0.8 deberia ser scale= 0.1

4. Bajo scale y ademas pongo iou=0.5 como Dezso. Sigue mal el pacience lo corta enseguida
aunque las graficas hay losses bajando. VOy a quitar patience y mas epochs.




|  ID |Descripción                    | ddsm train  | ddsm val|  inbreast | pilot |
|---|-------------------------------------------------------------|-------------|---------|-----------|--------|
| 1 | Opción por defecto con mosaico...                                          |
| 2 | Sigue yendo mal, parece que hay problemas con el NMS muchos candidatos      |
| 3 |                              | Sigue sin funcionar, veo que hace demasiado zoom, igual en lugar de 0.8     |
|   |                              | debería ser scale= 0.1                                                     |
| 4 |                              | Bajo scale y además pongo iou=0.5 como Dezso. Sigue mal el pacience lo corta |
|   |                              | enseguida aunque las gráficas hay losses bajando. Voy a quitar patience y   |
|19 | Entrenado solo con imagenes con anotaciones, cambia                                                                   |
