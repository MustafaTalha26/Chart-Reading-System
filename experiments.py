import cv2
import keras
import numpy as np
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from matplotlib import pyplot as plt

# Definition and attributes of the model
def define_model():
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(224,224,3)))
    model.add(tf.keras.layers.Rescaling(1./255))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-6),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    #print(model.summary())
    return model

dataset_path = "graydata/"
# Training function of the model
# Model and weights will be saved for further use
def train_model():
    model = define_model()
    train = keras.utils.image_dataset_from_directory(
        dataset_path,
        seed=1,
        image_size=(224,224),
        batch_size=32,
        validation_split = 0.2,
        subset = "training",
        label_mode='categorical'
    )
    valid = keras.utils.image_dataset_from_directory(
        dataset_path,
        seed=1,
        image_size=(224,224),
        batch_size=32,
        validation_split = 0.2,
        subset = "validation",
        label_mode='categorical'
    )
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    epochs= 10
    history = model.fit(
        train,
        epochs=epochs,
        validation_data=valid,
        callbacks=[callback]
    )
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    model.save_weights('test_model_w.h5')

# for testing testdata/ file
def test_model():
    model = define_model()
    model.load_weights('test_model_w.h5')
    test = keras.utils.image_dataset_from_directory(
        "graytest/",
        seed=1,
        image_size=(224,224),
        batch_size=32,
        label_mode='categorical'
    )
    predictions = np.array([])
    labels =  np.array([])
    for x, y in test:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
    print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())
    cm = metrics.confusion_matrix(labels,predictions)
    sns.heatmap(cm, 
                annot=True,
                fmt='g',
                cmap="Blues", 
                xticklabels=['BarH','BarV','Line','Pie'],
                yticklabels=['BarH','BarV','Line','Pie'])
    plt.xlabel('Prediction',fontsize=13)
    plt.ylabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()
    

# Predict single chart
# Return index number
def predict_chart(imagepath):
    img = cv2.imread(imagepath)
    size = (224, 224)
    img = cv2.resize(img, size)
    img = np.expand_dims(img, axis=0)
    model = define_model()
    model.load_weights('test_model_w.h5')
    y_pred = model.predict(img)
    predicted_class_index = np.argmax(y_pred, axis=1)[0]
    class_labels = ["BarH", "BarV", "Line", "Pie"]
    print(class_labels[predicted_class_index],"Chart")
    return predicted_class_index

train_model()
test_model()