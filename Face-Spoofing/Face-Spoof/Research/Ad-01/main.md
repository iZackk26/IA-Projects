Papers: https://paperswithcode.com/paper/learn-convolutional-neural-network-for-face

# Modelos utiles
## Viola-Jones
Viola-Jones es un modelo del framework OpenCV que se puede utilizar para la deteccion de personas

## Pasos del metodo

# Paso 1: Localizacion facial
Primero se utiliza un algoritmo de localizacion facial, en este caso, utilizan Viola-Jones, del framework OpenCV.

#Paso 2: Spatial Augmentation
En este modelo, se propone expandir el espacio de la fotografia, para que el modelo tenga mas criterios de evaluar si se trata de un fraude o no apartir del fondo.

#Paso 4
Una vez preparado el dataset, se le pasa al modelo, el cual tiene la capacidad de extraer capas de las imagenes (5 convolutional layers y 3 fully-connected layers) y finalmente apartir de esas capas usa el
algoritmo de support vector machine (SVM) para realizar la clasificacion.


