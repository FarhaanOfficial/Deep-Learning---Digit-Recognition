import tensorflow
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_data) = mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

from keras.api._v2.keras import metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_data)
print('Test Acuracy', test_acc)
print('Loss', test_loss)