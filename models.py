import keras
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.applications import EfficientNetB7
from keras.applications import MobileNetV2
from keras.applications import DenseNet121
from keras.applications import InceptionV3
from keras.applications import ResNet101V2
from keras.applications import VGG16
from keras.applications import Xception

def EfficientNetB7_model():
    baseModel = EfficientNetB7(include_top=False, input_tensor=keras.layers.Input(shape=(224,224, 3)))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        loss= "categorical_crossentropy", 
        metrics=["accuracy"], 
        optimizer=keras.optimizers.Adam(1e-6)
    )
    return model

def MobileNetV2_model():
    baseModel = MobileNetV2(include_top=False, input_tensor=keras.layers.Input(shape=(224,224, 3)))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        loss= "categorical_crossentropy", 
        metrics=["accuracy"], 
        optimizer=keras.optimizers.Adam(1e-6)
    )
    return model

def DenseNet121_model():
    baseModel = DenseNet121(include_top=False, input_tensor=keras.layers.Input(shape=(224,224, 3)))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        loss= "categorical_crossentropy", 
        metrics=["accuracy"], 
        optimizer=keras.optimizers.Adam(1e-6)
    )
    return model

def InceptionV3_model():
    baseModel = InceptionV3(include_top=False, input_tensor=keras.layers.Input(shape=(224,224, 3)))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        loss= "categorical_crossentropy", 
        metrics=["accuracy"], 
        optimizer=keras.optimizers.Adam(1e-6)
    )
    return model

def ResNet101V2_model():
    baseModel = ResNet101V2(include_top=False, input_tensor=keras.layers.Input(shape=(224,224, 3)))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        loss= "categorical_crossentropy", 
        metrics=["accuracy"], 
        optimizer=keras.optimizers.Adam(1e-6)
    )
    return model

def VGG16_model():
    baseModel = VGG16(include_top=False, input_tensor=keras.layers.Input(shape=(224,224, 3)))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        loss= "categorical_crossentropy", 
        metrics=["accuracy"], 
        optimizer=keras.optimizers.Adam(1e-6)
    )
    return model

def Xception_model():
    baseModel = Xception(include_top=False, input_tensor=keras.layers.Input(shape=(224,224, 3)))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        loss= "categorical_crossentropy", 
        metrics=["accuracy"], 
        optimizer=keras.optimizers.Adam(1e-6)
    )
    return model