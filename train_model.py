import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train/255.0 #covert to 0-1 NN train better with small val
x_test=x_test/255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
#model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')#softmax gives probability
])
#ReLU helps learn features

#Softmax converts features into probabilities
#relu:rectified linear unit:
#ReLU(x) = max(0, x) It helps the network learn complex patterns
#if x > 0 → keep it
#If x < 0 → make it 0
#example:Input:  [-3, -1, 2, 5]
#Output: [ 0,  0, 2, 5]

#[2.3, 1.1, 4.8]-->raw output
#after softmax-->[0.07, 0.02, 0.91]
#it makes sure all outputs are between 0 and 1

model.compile(
    optimizer='adam',#how to adjust weights
    loss='sparse_categorical_crossentropy',#its used to measure error,expects probabilities hence softmax on o/p layer,used cuz label is single digit
    metrics=['accuracy']#what to track
)

model.fit(x_train, y_train, epochs=5)
#backpropagation
model.evaluate(x_test,y_test)
predictions = model.predict(x_test)#make predictions
#each prediction looks like:[0.01, 0.00, 0.92, 0.03, 0.00, 0.00, 0.01, 0.01, 0.01, 0.01]
#Probability of 0 → 1%

#Probability of 1 → 0%
plt.figure(figsize=(10,5))

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(np.argmax(predictions[i]))
    plt.axis('off')

plt.show()

# rebuild with old-style input_shape so Render's Keras can load it
new_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
new_model.set_weights(model.get_weights())
new_model.save("digit_classifier.h5")
test_load = tf.keras.models.load_model("digit_classifier.h5")
print("Model loaded successfully:", test_load.input_shape)