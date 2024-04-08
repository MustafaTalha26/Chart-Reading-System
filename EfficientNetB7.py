import keras
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.applications import EfficientNetB7

def EfficientNetB7_model(learning_rate, input_shape,class_number):
    baseModel = EfficientNetB7(include_top=False, input_tensor=keras.layers.Input(shape=input_shape))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    return model

lr = 0.0001
size = (224, 224)
shape = (224,224, 3) 
epochs = 5
class_number = 4

model = EfficientNetB7_model(lr,shape,class_number)
model.compile(loss= "categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

seed = 1
train = keras.utils.image_dataset_from_directory(
    "newdata/",
    seed=1,
    image_size=(224,224),
    batch_size=32,
    validation_split = 0.2,
    subset = "training",
    label_mode='categorical'
)
valid = keras.utils.image_dataset_from_directory(
    "newdata/",
    seed=1,
    image_size=(224,224),
    batch_size=32,
    validation_split = 0.2,
    subset = "validation",
    label_mode='categorical'
)
test = keras.utils.image_dataset_from_directory(
    "testdata/",
    seed=1,
    image_size=(224,224),
    batch_size=32,
    label_mode='categorical'
)

history = model.fit(
    train,
    epochs=epochs,
    validation_data=valid,
    batch_size=32
)

predictions = np.array([])
labels =  np.array([])
for x, y in test:
    predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])
    labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())