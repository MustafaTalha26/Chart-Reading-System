import keras
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf

def define_model():
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(224,224,3)))
    model.add(tf.keras.layers.Rescaling(1./255))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    # compile model
    return model

model = define_model()

seed = 1
train = keras.utils.image_dataset_from_directory(
    "rawdata/",
    seed=1,
    image_size=(224,224),
    batch_size=32,
    validation_split = 0.2,
    subset = "training",
    label_mode='categorical'
)
valid = keras.utils.image_dataset_from_directory(
    "rawdata/",
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
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
epochs= 10

model.compile(
    optimizer=keras.optimizers.Adam(1e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    train,
    epochs=epochs,
    validation_data=valid,
    callbacks=[callback]
)

predictions = np.array([])
labels =  np.array([])
for x, y in test:
    predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])
    labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())

