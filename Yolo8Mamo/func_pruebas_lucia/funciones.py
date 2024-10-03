#Bucle para agrupar todos los casos de una carpeta en una variable

import os
# Itera sobre los archivos en el directorio
for nombre_archivo in os.listdir(directorio):
    # Verifica si es un archivo (no una carpeta)
    if os.path.isfile(os.path.join(directorio, nombre_archivo)):
        archivos.append(nombre_archivo)

# Imprime la lista de archivos
print(archivos)