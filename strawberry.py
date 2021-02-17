from sklearn.metrics import log_loss
import cv2
import os
import glob
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Dropout, Conv2D, Dense, BatchNormalization, AveragePooling2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.applications.xception import Xception
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

detail = sorted(
    glob.glob(_input_path))
foldername = [str(i.split("/")[0]) + "/" + str(i.split("/")[1]) + "/" + str(i.split("/")[2]) + "/" + str(i.split("/")[3]) + "/" + str(i.split("/")[4]) + "/"
              + str(i.split("/")[5]) + "/" + str(i.split("/")[6]) + "/" + str(i.split("/")[7]) + "/" + str(i.split("/")[8]) for i in detail]
imagename = [str(i.split("/")[9]) for i in detail]
label = np.array((pd.read_csv('./out-file.csv'))["name"])

data_detail = pd.DataFrame()
data_detail["foldername"] = foldername
data_detail["imagename"] = imagename
data_detail["label"] = label

train_data_detail, test_data_detail = train_test_split(
    data_detail, stratify=data_detail["label"], test_size=0.08)

# Splitting training data into final training set and cross validation set
train_data_detail, cv_data_detail = train_test_split(
    train_data_detail, stratify=train_data_detail["label"], test_size=0.086956)
train_data_detail.shape, test_data_detail.shape, cv_data_detail.shape

# Resetting index of train, cross validation and test set
train_data_detail.reset_index(inplace=True, drop=True)
cv_data_detail.reset_index(inplace=True, drop=True)
test_data_detail.reset_index(inplace=True, drop=True)

base_model = Xception(weights='imagenet', include_top=False)

train_x = []
train_y = []
for i in range(len(train_data_detail)):
    path1 = train_data_detail["foldername"][i]
    path2 = train_data_detail["imagename"][i]
    image = cv2.imread(os.path.join(path1, path2))
    image = cv2.resize(image, (224, 224))
    # here, we are normalizing the images
    image = image/255.0
    image = image.reshape(1, 224, 224, 3)
    image = base_model.predict(image)
    image = image.reshape(image.shape[1], image.shape[2], image.shape[3])
    # Creating and saving each image in the form of numerical data in an array
    train_x.append(image)
    # appending corresponding labels
    train_y.append(train_data_detail['label'][i])
    if i % 500 == 0:
        print("no of images processed =", i)

train_x = np.array(train_x, dtype=np.uint8)
train_y = np.array(pd.get_dummies(train_y), dtype=np.uint8)
print(" for training data ", train_x.shape, train_y.shape)

# for test data
cv_x = []
cv_y = []
for i in range(len(cv_data_detail)):
    path1 = cv_data_detail["foldername"][i]
    path2 = cv_data_detail["imagename"][i]
    image = cv2.imread(os.path.join(path1, path2))
    image = cv2.resize(image, (224, 224))
    # here, we are normalizing the images
    image = image/255.0
    image = image.reshape(1, 224, 224, 3)
    image = base_model.predict(image)
    image = image.reshape(image.shape[1], image.shape[2], image.shape[3])
    # Creating and saving each image in the form of numerical data in an array
    cv_x.append(image)
    # appending corresponding labels
    cv_y.append(cv_data_detail['label'][i])
    if i % 500 == 0:
        print("no of images processed =", i)

cv_x = np.array(cv_x, dtype=np.uint8)
cv_y = np.array(pd.get_dummies(cv_y), dtype=np.uint8)
print(" for cv data ", cv_x.shape, cv_y.shape)


def model():
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_x.shape[1:]))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    return model


model = model()
model.summary()


# Compiling and running the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=["accuracy"])
hist = model.fit(train_x, train_y, validation_data=(cv_x, cv_y), epochs=25)


model.save("strawberry_disease")

# visualizing losses and accuracy with epochs
epoch_number = []
for epoch in range(25):
    epoch_number.append(epoch + 1)
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

log_frame = pd.DataFrame(
    columns=["Epoch", "Train_Loss", "Train_Accuracy", "CV_Loss", "CV_Accuracy"])
log_frame["Epoch"] = epoch_number
log_frame["Train_Loss"] = train_loss
log_frame["Train_Accuracy"] = train_acc
log_frame["CV_Loss"] = val_loss
log_frame["CV_Accuracy"] = val_acc
log_frame


def plotting(epoch, train_acc, CV_acc, title):
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes.plot(epoch, train_acc, color='red', label="Train_Accuracy")
    axes.plot(epoch, CV_acc, color='blue', label="CV_Accuracy")
    axes.set_title(title, fontsize=25)
    axes.set_xlabel("Epochs", fontsize=20)
    axes.set_ylabel("Accuracy", fontsize=20)
    axes.grid()
    axes.legend(fontsize=20)


plotting(list(log_frame["Epoch"]), list(log_frame["Train_Accuracy"]), list(
    log_frame["CV_Accuracy"]), "EPOCH VS ACCURACY")

# for cv data
test_x = []
test_y = []
for i in range(len(test_data_detail)):
    path1 = test_data_detail["foldername"][i]
    path2 = test_data_detail["imagename"][i]
    image = cv2.imread(os.path.join(path1, path2))
    image = cv2.resize(image, (224, 224))
    # here, we are normalizing the images
    image = image/255.0
    image = image.reshape(1, 224, 224, 3)
    image = base_model.predict(image)
    image = image.reshape(image.shape[1], image.shape[2], image.shape[3])
    # Creating and saving each image in the form of numerical data in an array
    test_x.append(image)
    # appending corresponding labels
    test_y.append(test_data_detail['label'][i])
test_x = np.array(test_x, dtype=np.uint8)
test_y = np.array(pd.get_dummies(test_y), dtype=np.uint8)
print(" for test data ", test_x.shape, test_y.shape)

test_predict = model.predict(test_x)

# log loss on test data
loss = log_loss(test_y, test_predict)
loss

# free up ram
del test_x
del test_y
del test_predict
gc.collect()

# We also need to read the test data for prediction from a file which contains data in the form of image.
# The folder is named as 'test' and it contains images different breed of dogs

# First of all we will extract the detail of all the data and save all of them in terms of dataframe with foldername and imagename only
detail = sorted(glob.glob(
    _input_test_path))
foldername = [str(i.split("/")[0]) + "/" + str(i.split("/")[1]) + "/" + str(i.split("/")[2]) + "/" + str(i.split("/")[3]) + "/" + str(i.split("/")[4]) + "/"
              + str(i.split("/")[5]) + "/" + str(i.split("/")[6]) + "/" + str(i.split("/")[7]) + "/" + str(i.split("/")[8]) for i in detail]
imagename = [str(i.split("/")[9]) for i in detail]

# Defining dataframe and saving all the extracted information in that dataframe
test_data_for_prediction_detail = pd.DataFrame()
test_data_for_prediction_detail["foldername"] = foldername
test_data_for_prediction_detail["imagename"] = imagename

# Analying the test data set for prediction detail
print("\nNumber of images in test data set for prediction  = "+str(len(detail)))
print(test_data_for_prediction_detail.columns)
test_data_for_prediction_detail.head()

# Changing the data into an array of pixels and labels so that it can be fed into the model for prediction
# Initially it was in the form of a DataFrame

# for test data for prediction data
prediction = []
for i in range(len(test_data_for_prediction_detail)):
    path1 = test_data_for_prediction_detail["foldername"][i]
    path2 = test_data_for_prediction_detail["imagename"][i]
    image = cv2.imread(os.path.join(path1, path2))
    image = cv2.resize(image, (224, 224))
    # here, we are normalizing the images
    image = image/255.0
    image = image.reshape(1, 224, 224, 3)
    image = base_model.predict(image)
    image = image.reshape(image.shape[1], image.shape[2], image.shape[3])
    # Creating and saving each image in the form of numerical data in an array
    prediction.append(image)
    if i % 500 == 0:
        print("no of images processed =", i)
prediction = np.array(prediction, dtype=np.uint8)
print(" for test data for prediction ", prediction.shape)

# Now prediction on data to be predicted
prediction_predict = model.predict(prediction)

prediction_predict

np.argmax(prediction_predict, axis=1)
