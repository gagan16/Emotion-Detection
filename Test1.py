from keras.models import model_from_json
from pathlib import Path
import numpy as np
import cv2


import matplotlib.pyplot as plt

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

EMOTIONS_LIST = ["Angry", "Fear/Surprise","Sad","Happy", "Neutral"] # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 48  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

# model = tf.keras.models.load_model("model1.json", "chkPt1.h5")
# prediction = model.predict([prepare('happy.jpg')])
# print(EMOTIONS_LIST[int(prediction[0][0])])

f = Path("model.json")
model_structure = f.read_text()
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("all.h5")
prediction = model.predict([prepare('AJ_Cook_0001.jpg')])
print(prediction)
most_likely_class_index = int(np.argmax(prediction))
class_likelihood = EMOTIONS_LIST[most_likely_class_index]
class_label = EMOTIONS_LIST[most_likely_class_index]
print("Emotion : {}".format(class_label))

a=np.concatenate(prediction)
N = len(a)
x = range(N)
width = 1/1.5
plt.bar(x, a*100, width, color=['black', 'red', 'green', 'blue', 'cyan'])
plt.xticks(x, EMOTIONS_LIST)

plt.show()
