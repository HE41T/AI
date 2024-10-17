import tensorflow as tf
import numpy as np
to_categorical = tf.keras.utils.to_categorical
fashion_mnist = tf.keras.datasets.fashion_mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
VAL_SIZE = 0.2
RANDOM_STATE = 99
BATCH_SIZE = 256

INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Fashion MNIST train - rows:", x_train.shape[0], "columns:", x_train.shape[1], "rows:", x_train.shape[2])
print("Fashion MNIST test - rows:", x_test.shape[0], "columns:", x_test.shape[1], "rows:", x_test.shape[2])

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE)

x_train.shape, x_val.shape, y_train.shape, y_val.shape

# Feature Extraction
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

# Image Classification
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

NO_EPOCHS = 15

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NO_EPOCHS, validation_data=(x_val, y_val))

predicted_classes = model.predict(x_test)
predicted_classes = np.argmax(predicted_classes, axis=1)

y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, predicted_classes)
print(classification_report(y_true, predicted_classes, target_names=labels, digits=4))
