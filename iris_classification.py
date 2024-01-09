import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

print("Scikit-learn Logistic Regression:")
print("Classification Report:")
print(classification_report(y_test, lr_predictions))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_tf, y_train_tf, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
_, accuracy = model.evaluate(X_test_tf, y_test_tf)

tf_predictions = model.predict(X_test_tf)
tf_pred_classes = np.argmax(tf_predictions, axis=1)
y_test_classes = np.argmax(y_test_tf, axis=1)

print("\nTensorFlow Neural Network:")
print("Classification Report:")
print(classification_report(y_test_classes, tf_pred_classes, zero_division=1))
