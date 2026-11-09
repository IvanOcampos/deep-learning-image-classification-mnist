# Skynet: clasificación de imágenes con redes neuronales

Desafío resuelto en dos notebooks independientes con TensorFlow/Keras. Se eligió Keras por su API clara y el historial de métricas por epoch. Ambos modelos son MLP con una capa oculta Dense de 128 unidades y una salida softmax de 10 clases, dentro del límite de 1–2 capas ocultas. Las imágenes 28×28 se normalizan a [0, 1] y se aplanan a 784 entradas.

## Entregables

- [`01_mnist_neural_network.ipynb`](01_mnist_neural_network.ipynb): exploración, preprocesamiento, benchmark ReLU/Sigmoid × SGD/Adam, curvas por epoch, evaluación de test, matriz de confusión, aciertos, errores y conclusiones.
- [`02_fashion_mnist_neural_network.ipynb`](02_fashion_mnist_neural_network.ipynb): Fashion MNIST con la arquitectura y configuración ganadoras de MNIST (Sigmoid + Adam), curvas, evaluación y análisis de errores.

## Resultados de la ejecución guardada

En MNIST, **Sigmoid + Adam** ganó con `val_accuracy=0.9690` y `test_accuracy=0.9722`. Superó a ReLU + Adam por 0.0011 de accuracy de validación; ambas configuraciones Adam alcanzaron 90% en el primer epoch. ReLU + SGD llegó en el epoch 4 y Sigmoid + SGD no llegó al 90% en 10 epochs. La brecha final train-validación del ganador fue 0.0238.

Con la misma configuración Sigmoid + Adam, Fashion MNIST obtuvo `val_accuracy=0.8528` y `test_accuracy=0.8436`, **12.86 puntos porcentuales** menos que MNIST. La confusión más frecuente fue Suéter → Camisa (225 casos). La diferencia concuerda con que las prendas comparten siluetas y son visualmente más variadas que los dígitos.

Para este baseline elegiría **Adam** por su avance rápido y mejor accuracy que SGD en el benchmark observado. Antes de un uso real probaría más semillas, Early Stopping, Dropout y una CNN. No se aplicó regularización explícita en estos experimentos. Le tengo **respeto** al modelo: su accuracy es alta, pero la matriz y los errores con alta confianza muestran que puede equivocarse con seguridad.

## Ejecución

Abrí cada notebook en Jupyter y ejecutá sus celdas de arriba hacia abajo. Requiere Python, TensorFlow, NumPy y Matplotlib; Keras descarga los conjuntos de datos en la primera ejecución. Los números aquí son los resultados guardados de esta corrida y pueden variar según la semilla, la versión y el entorno.
