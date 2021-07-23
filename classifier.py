import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

X = np.load("images.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

classes = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=2, train_size=7500, test_size=2500
)

x_train_scaled = x_train / 255
x_test_scaled = x_test / 255

clf = LogisticRegression(solver="saga", multiclass="multinomial").fit(
    x_train_scaled, y_train
)


def get_prediction(image):
    im_PIL = Image.open(image)
    image_bw = im_PIL.convert("L")
    image_resize = image_bw.resize((28, 28))
    pixel_filter = 20
    min_pix = np.percentile(image_resize, pixel_filter)
    image_scaled = np.clip(image_resize - min_pix, 0, 255)
    max_pix = np.max(image_resize)
    image_scaled = np.asarray(image_scaled) / max_pix
    test_sample = np.array(image_scaled).reshape(1, 784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]
