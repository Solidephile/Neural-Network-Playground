import os
import cv2
import numpy as np
import pickle
import gzip


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path, progress, progress_callback):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_GRAYSCALE)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

            progress += 1
            print(progress)
            # if progress % 700 == 0:
            #     progress_callback.emit(progress)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype("uint8")


# MNIST dataset (train + test)
def create_data_mnist(path, progress_callback):
    # Load both sets separately
    X_test, y_test = load_mnist_dataset(dataset="test", path=path, progress=0, progress_callback=progress_callback)
    X, y = load_mnist_dataset(dataset="train", path=path, progress=10000, progress_callback=progress_callback)

    # And return all the data
    return X, y, X_test, y_test


# Preprocess the dataset
def preprocess_dataset(X, y, X_test, y_test):
    # Scale features
    X = (X.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    # Reshape to vectors
    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Shuffle the training dataset
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    return X, y, X_test, y_test


# Predict a single image
def predict_image(path, model):
    # Read an image
    image_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Resize to the same size as Fashion MNIST images
    image_data = cv2.resize(image_data, (28, 28))

    # Reshape and scale pixel data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    # Predict on the image
    predictions = model.predict(image_data)

    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(predictions)

    print(predictions)


if __name__ == "__main__":
    X, y, X_test, y_test = create_data_mnist("mnist_images", None)
    X, y, X_test, y_test = preprocess_dataset(X, y, X_test, y_test)

    file = gzip.GzipFile("mnist_data/train_images", "wb")
    pickle.dump(X, file)
    file.close()
    file = gzip.GzipFile("mnist_data/train_labels", "wb")
    pickle.dump(y, file)
    file.close()
    file = gzip.GzipFile("mnist_data/test_images", "wb")
    pickle.dump(X_test, file)
    file.close()
    file = gzip.GzipFile("mnist_data/test_labels", "wb")
    pickle.dump(y_test, file)
    file.close()
