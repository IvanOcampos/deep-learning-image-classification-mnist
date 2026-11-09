# Clasificación de imágenes con redes neuronales

Proyecto de clasificación de imágenes con redes neuronales densas, desarrollado con TensorFlow/Keras sobre MNIST y Fashion MNIST.

## Notebooks

1. [MNIST](01_mnist_neural_network.ipynb): explora y prepara los datos; compara ReLU y Sigmoid con SGD y Adam; grafica loss y accuracy de entrenamiento y validación; evalúa en test con matriz de confusión y ejemplos de predicciones correctas e incorrectas.
2. [Fashion MNIST](02_fashion_mnist_neural_network.ipynb): aplica la configuración ganadora de MNIST y presenta las mismas métricas de evaluación para las diez clases de indumentaria.

## Método

Se eligió TensorFlow/Keras por su API de alto nivel y el historial de métricas que genera durante el entrenamiento. La arquitectura usada en ambos conjuntos es `784 → Dense(128) → Dense(10, softmax)`: una capa oculta de 128 neuronas, dentro del límite pedido. Las imágenes de 28 × 28 píxeles se normalizan al rango [0, 1] y se aplanan a 784 entradas. Se separan 10.000 imágenes del conjunto de entrenamiento para validación y se reserva el conjunto de test para la evaluación final.

Los experimentos usan 10 epochs, batch size 128 y learning rate 0,01. En MNIST se comparan las cuatro combinaciones de activación y optimizador con los mismos hiperparámetros base.

## Resultados

### MNIST: benchmark de validación

| Activación | Optimizador | Accuracy final de validación | Primer epoch con ≥ 90 % |
|---|---|---:|---:|
| ReLU | SGD | 92,22 % | 4 |
| ReLU | Adam | 96,79 % | 1 |
| Sigmoid | SGD | 88,16 % | No alcanzado |
| Sigmoid | Adam | **96,90 %** | 1 |

La configuración ganadora fue **Sigmoid + Adam**, con una ventaja de 0,11 puntos porcentuales sobre ReLU + Adam. Adam llegó al 90 % en el primer epoch con ambas activaciones. SGD fue más lento y, con Sigmoid, no alcanzó ese umbral durante los 10 epochs. El accuracy final en el conjunto de test fue **97,22 %**. La diferencia final entre accuracy de entrenamiento y validación fue 0,0238.

### Fashion MNIST: configuración ganadora

Con Sigmoid + Adam, el accuracy fue **85,28 % en validación** y **84,36 % en test**. La confusión más frecuente fue Suéter clasificado como Camisa (225 casos). El accuracy de test quedó 12,86 puntos porcentuales por debajo de MNIST; las clases de prendas presentan siluetas y texturas más parecidas entre sí.

## Conclusiones

Para este baseline elegiría **Adam**, que alcanzó mayor accuracy y convergió más rápido que SGD en los experimentos de MNIST. Antes de desplegar el modelo, probaría varias semillas, Early Stopping y Dropout, y compararía con una CNN para conservar la estructura espacial de las imágenes. No se aplicó regularización explícita en estos experimentos.

Le tengo **respeto** al modelo: obtiene buenos resultados, pero la matriz de confusión y los errores de alta confianza muestran que puede equivocarse con seguridad.

## Requisitos y ejecución

Requiere Python, TensorFlow, NumPy y Matplotlib. Abrí cada notebook con Jupyter y ejecutá las celdas en orden. En la primera ejecución, Keras puede descargar los datasets. Los resultados pueden variar levemente según las versiones de software y el entorno.
