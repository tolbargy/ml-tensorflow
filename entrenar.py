import sys
import os

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

# Si hay una sesion de keras en este momento ejecutandose en la computadora vamos a matarlo
K.clear_session()

data_entrenamiento='./data/entrenamiento'
data_validacion='./data/validacion'

# ---- Parametros
# Numero de iteraciones en el dataset
epocas=20
# Tamaño para procesar nuestras imagenes (valores en pixeles)
altura, longitud=100,100
# Numero de imagenes que vamos a procesar en cada uno de los pasos
batch_size=32
# Numero de veces que se va procesar la informacion en cada una de las epocas
pasos=1000
# Al final de cada una de las epocas se va correr para poder validar que este aprendiendo el algoritmo
pasos_validacion=200

# Profundidad de convolucion
filtros_conv1=32
filtros_conv2=64
# Tamaño de filtro que estaremos usando la convolucion (altura,longitud)
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
# Tamaño de filtro en el maxpooling
tamano_pool=(2,2)
# Las clases de animales que van haber
clases=3
# Learning rate, que tan grande haran los ajustes nuestra red neuronal para llegar a una solucion optima
lr=0.0005


# ---- Preprocesamiento de imagenes
# Generador
entrenamiento_datagen=ImageDataGenerator(
    rescale=1./255, # reescalar los pixeles de la imagen esten de 0 a 1 para hacer eficiente el entrenamiento
    shear_range=0.3, # Generar la imagenes y las va inclinar
    zoom_range=0.3, # Le va hacer zoom algunas imagenes a otras no
    horizontal_flip=True # Invertir la imagen para que nuestra red neuronal aprenda a entender la direccion
)

# Dejar la validacion y revisar la imagen tan cual son, sin zoom, ni voltear ni nada
validacion_datagen=ImageDataGenerator(
    rescale=1./255
)

imagen_entrenamiento= entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical' # La clasificacion que estamos haciendo en este algoritmo es categorica (perro,gato,gorila)    
)

imagen_validacion=validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical'
)


# ---- Crear la red convolucional (CNN)
cnn=Sequential() #Va ser secuencial van haber varias capadas apiladas entre ellas

cnn.add(Convolution2D(
    filtros_conv1,
    tamano_filtro1,
    padding='same', # Que es lo que va hacer nuestros filtros en las esquinas
    input_shape=(altura,longitud,3), # Las imagenes que vamos entregar a nuestras capas van a tener ciertos tamaños
    activation='relu' #La funcion de activacion
))

cnn.add(MaxPooling2D(
    pool_size=tamano_pool
))

cnn.add(Convolution2D(
    filtros_conv2,
    tamano_filtro2,
    padding='same',
    activation='relu'
))

cnn.add(MaxPooling2D(
    pool_size=tamano_pool
))

# Hacer la imagen plana, ya que está pequeña y profunda, para que esté de 1 sola dimension
cnn.add(Flatten())
cnn.add(Dense(
    256, # cantidad neuronas
    activation='relu'
))

# A la capa Dense durante el entrenamiento de las neuronas le vamos a apagar el 50% de neuronas en cada paso (para evitar sobreajustar para que no aprenda solo un camino y se adapte a nueva informacion)
cnn.add(Dropout(0.5))
# Nos va ayudar a saber que tanto porcentaje de clase puede ser la imagen (Ej: 10% gorila,50% perro,40% gato)
cnn.add(Dense(clases,activation='softmax'))

cnn.compile(
    loss='categorical_crossentropy', # Decirle a nuestra red neuronal su funcion de perdida, saber que va bien y que va mal
    optimizer=Adam(lr=lr), # El porcentaje de optimizador
    metrics=['accuracy'] # La metrica con la cual esta optimizando tratar de mejorar su precision de la clasificacion de imagenes
)

# --- Entrenando nuestro algoritmo
cnn.fit_generator(
    imagen_entrenamiento,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion
)

dir_modelo='./modelo/'
if not os.path.exists(dir_modelo):
    os.mkdir(dir_modelo)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')