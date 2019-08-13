#  Limiting fer2013 dataset images to 2000 per emotion.

import tarfile, csv
import pandas as pd

data_comp = tarfile.open("fer2013.tar")
ds = pd.read_csv(data_comp.extractfile("fer2013/fer2013.csv"))
ds.head()

train = ds[["emotion", "pixels", "Usage"]].values[ds["Usage"] == "Training"]
test = ds[["emotion", "pixels", "Usage"]].values[ds["Usage"] == "Test"]

csvData = ['emotion', 'pixel', 'usage']
with open('fer2013LimitedImages.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(csvData)

csvFile.close()

def trainOrTest(train_data, length):
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral = (0,) * 7

    for data in range(len(train_data)):
        if train_data[data][0] == 0:
            if Angry >= length:
                continue
            else:
                Angry += 1
        elif train_data[data][0] == 1:
            if Disgust >= length:
                continue
            else:
                Disgust += 1
        elif train_data[data][0] == 2:
            if Fear >= length:
                continue
            else:
                Fear += 1
        elif train_data[data][0] == 3:
            if Happy >= length:
                continue
            else:
                Happy += 1
        elif train_data[data][0] == 4:
            if Sad >= length:
                continue
            else:
                Sad += 1
        elif train_data[data][0] == 5:
            if Surprise >= length:
                continue
            else:
                Surprise += 1
        elif train_data[data][0] == 6:
            if Neutral >= length:
                continue
            else:
                Neutral += 1

        csvData = train[data]

        with open('fer2013LimitedImages.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(csvData)

        csvFile.close()
        # print(Happy, Sad, Angry, Disgust, Surprise, Fear, Neutral, sep="\n")


trainOrTest(train, 2000)
trainOrTest(test, 100)

# for data in range(len(train)):
#     if train[data][0] == 0:
#         if Angry >= 2000:
#             continue
#         else:
#             Angry += 1
#     elif train[data][0] == 1:
#         if Disgust >= 2000:
#             continue
#         else:
#             Disgust += 1
#     elif train[data][0] == 2:
#         if Fear >= 2000:
#             continue
#         else:
#             Fear += 1
#     elif train[data][0] == 3:
#         if Happy >= 2000:
#             continue
#         else:
#             Happy += 1
#     elif train[data][0] == 4:
#         if Sad >= 2000:
#             continue
#         else:
#             Sad += 1
#     elif train[data][0] == 5:
#         if Surprise >= 2000:
#             continue
#         else:
#             Surprise += 1
#     elif train[data][0] == 6:
#         if Neutral >= 2000:
#             continue
#         else:
#             Neutral += 1
#
#     csvData = train[data]
#
#     with open('fer2013LimitedImages.csv', 'a') as csvFile:
#         writer = csv.writer(csvFile)
#         writer.writerow(csvData)
#
#     csvFile.close()
