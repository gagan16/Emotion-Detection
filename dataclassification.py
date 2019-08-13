# Classifying the dataset. Converting images to pixels and emotions and storing them in csv file.

import pandas as pd
# from keras.layers import *
import numpy as np
import glob, csv
from keras.preprocessing import image

path = "images/"

allFiles = glob.glob(path + "*.jpg")
training_data_df = pd.read_csv("legend.csv")
X = training_data_df.drop('user.id', axis=1).values

csvData = ['emotion', 'pixel']
with open('classifiedData.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(csvData)

csvFile.close()

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral = (0,) * 7
count = 0

for file_ in allFiles:
    count += 1
    img = image.load_img(file_, target_size=(48, 48), grayscale=True)
    file_ = file_[7:]
    pixel = image.img_to_array(img)
    pixel = np.expand_dims(pixel, axis=0)

    for x in range(len(X)):
        if X[x][0] == file_:
            if X[x][1] == 'anger' or X[x][1] == 'ANGER':
                Anger += 1
                emotion = 0
            elif X[x][1].lower() == 'disgust' or X[x][1] == "DISGUST":
                Disgust += 1
                emotion = 1
            elif X[x][1].lower() == 'fear' or X[x][1] == "FEAR":
                Fear += 1
                emotion = 2
            elif X[x][1].lower() == 'happiness' or X[x][1] == "HAPPINESS":
                Happy += 1
                emotion = 3
            elif X[x][1].lower() == 'sadness' or X[x][1] == "SADNESS":
                Sad += 1
                emotion = 4
            elif X[x][1].lower() == 'surprise' or X[x][1] == "SURPRISE":
                Surprise += 1
                emotion = 5
            elif X[x][1].lower() == 'neutral' or X[x][1] == "NEUTRAL":
                Neutral += 1
                emotion = 6

            csvData = [emotion, pixel]

            with open('classifiedData.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(csvData)

            csvFile.close()
            break;

print(Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral, count, sep="\n")