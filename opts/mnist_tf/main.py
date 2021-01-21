import tensorflow as tf
from tensorflow.keras import datasets

def mnist_classifier(img_width, img_height):
    input_shape = (img_height, img_width)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = input_shape),
        tf.keras.layers.Dense(128, activation= "relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10,activation="softmax"),
    ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def normalize(x):
    x = x / 255.0
    return x

if __name__ == "__main__":
    # load data
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    cnt,h,w = x_train.shape
    
    model = mnist_classifier(w,h)
    model.fit(x = x_train, y = y_train, epochs = 5)
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(acc)






 