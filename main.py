from cnn.cnn import MyCnn
from cnn.layers import Conv2D, Pooling, Dense, Flatten
from typing import List
import numpy as np
import tensorflow as tf

IMG_HEIGHT, IMG_WIDTH = 128, 128
IMG_DIR = './dataset'
SEED = 6459164

def load_img_as_dataset() -> np.array:
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        IMG_DIR,
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    imgs = None
    labels = None
    for img_batch, label_batch in train_ds:
        imgs = img_batch.numpy()
        labels = label_batch.numpy()
        break

    return imgs, labels

def normalize_img(img: np.array) -> np.array:
    scalev = np.vectorize(lambda x: x / 255.0)
    return scalev(img)

def reorganize_layer(img: np.array) -> List[np.array]:
    input_shape = img.shape
    rows = input_shape[0]
    cols = input_shape[1]
    depth = input_shape[2]
    
    layers = [np.array([[0] * cols] * rows) for _ in range(depth)]

    for i, row in enumerate(img):
        for j, col in enumerate(row):
            for k, val in enumerate(col):
                layers[k][i][j] = val
    
    return layers

def interpret_class(res: List[np.array]) -> int:
    return 0 if res[0] > res[1] else 1

if __name__ == "__main__":
    imgs, labels = load_img_as_dataset()
    labels = list(labels)

    img_list = []
    for img in imgs:
        img_norm = normalize_img(img)
        img_reorganized = reorganize_layer(img)
        img_list.append(img_reorganized)
    img_list = img_list[:10]

    cnn = MyCnn()
    cnn.add(Conv2D(0, 8, np.array([4, 4]), 1, np.array([[IMG_HEIGHT, IMG_WIDTH] for _ in range(3)])))
    cnn.add(Pooling(np.array([2, 2]), 2, 'avg'))
    cnn.add(Flatten())
    cnn.add(Dense(2))

    for img, label in zip(img_list, labels):
        res = cnn.feed_forward(img)
        res_class = interpret_class(res)
        print(f'Prediction: {res_class}\t| Correct: {label}\t| Raw: {res}')
