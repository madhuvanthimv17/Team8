import matplotlib
#matplotlib.use("Agg")

from CNN import CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import tensorflow as tf
import pandas as pd

#Function to extract image pixel data and its corresponding label
def load_split(csvPath, t):

	data = []
	labels = []

	rows = open(csvPath).read().strip().split("\n")[1:]
	random.shuffle(rows)


	for (i, row) in enumerate(rows):

		if i > 0 and i % 1000 == 0:
			print("Processed {} total images".format(i))
		(label, imagePath) = row.strip().split(",")[-2:]

		image = io.imread("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/"+imagePath) # basepath + image path directly
		#Resizing every image to size 32x32
		image = transform.resize(image, (32, 32))

		#Baeline 3 : Application of CLAHE
		image = exposure.equalize_adapthist(image, clip_limit=0.1)
		#End of Baseline 3

		i = image.astype('float')
		#Normalize image pixel value to range from 0 to 255
		data.append(i.__itruediv__(255.0))
		labels.append(int(label))


	data = np.stack(data)
	labels = np.stack(labels)

	return (data, labels)



NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64


labelNames = open("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

print("Loading training and testing data...")
(trainX, trainY) = load_split("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/Train.csv", 1)
(testX, testY) = load_split("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/Test.csv", 0)

numLabels = len(np.unique(trainY))

trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

#Baseline 2 : Augmenting data after assigning weights to the classes
classTotals = trainY.sum(axis=0)
classWeight = dict()

for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]

aug = ImageDataGenerator(
rotation_range=10,
zoom_range=0.15,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.15,
horizontal_flip=False,
vertical_flip=False,
fill_mode="nearest")
#End of Baseline 2

print("Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = CNN.build(width=32, height=32, depth=3,
	classes=numLabels)

model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


print("Training network...")
print(trainX.dtype)
H = model.fit(trainX, trainY,
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	verbose=1)

print("Evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

print("Serializing network to '{}'...".format("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/output"))
model.save("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/output")


N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("/Users/yashvinprakash/Downloads/Assignments/IntroToDataMining/dm_project/archive/output")
