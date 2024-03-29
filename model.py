import keras
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf

def define_model():
    model = keras.models.Sequential()
    keras.Input(shape=(200,200,3))
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
    "newdata/",
    seed=1,
    image_size=(200,200),
    batch_size=32,
    validation_split = 0.2,
    subset = "training",
    color_mode = 'grayscale',
    label_mode='categorical'
)
valid = keras.utils.image_dataset_from_directory(
    "newdata/",
    seed=1,
    image_size=(200,200),
    batch_size=32,
    validation_split = 0.2,
    subset = "validation",
    color_mode = 'grayscale',
    label_mode='categorical'
)
test = keras.utils.image_dataset_from_directory(
    "testdata/",
    seed=1,
    image_size=(200,200),
    batch_size=32,
    color_mode = 'grayscale',
    label_mode='categorical'
)

epochs= 5

model.compile(
    optimizer='Adam',
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    train,
    epochs=epochs,
    validation_data=valid,
)
model.save("x_model") 

predictions = np.array([])
labels =  np.array([])
for x, y in test:
    predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])
    labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())

