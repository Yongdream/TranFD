

#import sys
#import os


#os.system("pause")

#Svm 训练：
import sys
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import pickle
import dataset_get
import torch
#help(SVC)

SHAPE = (30, 30)
def getImageData(directory):
   s = 1
   feature_list = list()
   label_list   = list()
   num_classes = 0
   for root, dirs, files in os.walk(directory):
      for d in dirs:
         num_classes += 1
         images = os.listdir(root+d)
         for image in images:
            s += 1
            label_list.append(d)
            feature_list.append(extractFeaturesFromImage(root + d + "/" + image))

   return np.asarray(feature_list), np.asarray(label_list)

def extractFeaturesFromImage(image_file):
   img = cv2.imread(image_file)
   img = cv2.resize(img, SHAPE, interpolation = cv2.INTER_CUBIC)
   img = img.flatten()
   img = img / np.mean(img)
   return img


if __name__ == "__main__":

   # directory ="./SVM_test/image/image/"
   #
   #
   # feature_array, label_array = getImageData(directory)
   #
   # X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size = 0.2, random_state = 42)

   one_channel = 1

   train_data_UDDS_o, train_label_UDDS_o, val_data_UDDS_o, val_label_UDDS_o = dataset_get.Generate_Dataset_from_Dir("./Dataset", "UDDS")

   for i in range(0, len(train_data_UDDS_o)):
      current = np.array(torch.tensor(train_data_UDDS_o[i]).permute(2, 1, 0))
      np.random.shuffle(current)
      current = np.array(torch.tensor(current).permute(2, 1, 0))
      train_data_UDDS_o[i] = current

   if one_channel == 1:
      train_data_UDDS_o, val_data_UDDS_o = train_data_UDDS_o[:, 0, :, :], val_data_UDDS_o[:, 0, :, :]

   train_data_UDDS = train_data_UDDS_o.reshape((train_data_UDDS_o.shape[0], -1))
   val_data_UDDS = val_data_UDDS_o.reshape((val_data_UDDS_o.shape[0], -1))
   train_label_UDDS = []
   val_label_UDDS = []
   for i in range(0, len(train_label_UDDS_o)):
      if train_label_UDDS_o[i, 0, 0] == 1:
         train_label_UDDS.append("Cor")
      if train_label_UDDS_o[i, 0, 1] == 1:
         train_label_UDDS.append("Isc")
      if train_label_UDDS_o[i, 0, 2] == 1:
         train_label_UDDS.append("Noi")
      if train_label_UDDS_o[i, 0, 3] == 1:
         train_label_UDDS.append("Nor")
      if train_label_UDDS_o[i, 0, 4] == 1:
         train_label_UDDS.append("Vis")
   for i in range(0, len(val_label_UDDS_o)):
      if val_label_UDDS_o[i, 0, 0] == 1:
         val_label_UDDS.append("Cor")
      if val_label_UDDS_o[i, 0, 1] == 1:
         val_label_UDDS.append("Isc")
      if val_label_UDDS_o[i, 0, 2] == 1:
         val_label_UDDS.append("Noi")
      if val_label_UDDS_o[i, 0, 3] == 1:
         val_label_UDDS.append("Nor")
      if val_label_UDDS_o[i, 0, 4] == 1:
         val_label_UDDS.append("Vis")
   train_label_UDDS = np.array(train_label_UDDS)
   val_label_UDDS = np.array(val_label_UDDS)

   train_data_FUDS_o, train_label_FUDS_o, val_data_FUDS_o, val_label_FUDS_o = dataset_get.Generate_Dataset_from_Dir("./Dataset", "FUDS")

   for i in range(0, len(train_data_FUDS_o)):
      current = np.array(torch.tensor(train_data_FUDS_o[i]).permute(2, 1, 0))
      np.random.shuffle(current)
      current = np.array(torch.tensor(current).permute(2, 1, 0))
      train_data_FUDS_o[i] = current

   if one_channel == 1:
      train_data_FUDS_o, val_data_FUDS_o = train_data_FUDS_o[:, 0, :, :], val_data_FUDS_o[:, 0, :, :]

   train_data_FUDS = train_data_FUDS_o.reshape((train_data_FUDS_o.shape[0], -1))
   val_data_FUDS = val_data_FUDS_o.reshape((val_data_FUDS_o.shape[0], -1))
   train_label_FUDS = []
   val_label_FUDS = []
   for i in range(0, len(train_label_FUDS_o)):
      if train_label_FUDS_o[i, 0, 0] == 1:
         train_label_FUDS.append("Cor")
      if train_label_FUDS_o[i, 0, 1] == 1:
         train_label_FUDS.append("Isc")
      if train_label_FUDS_o[i, 0, 2] == 1:
         train_label_FUDS.append("Noi")
      if train_label_FUDS_o[i, 0, 3] == 1:
         train_label_FUDS.append("Nor")
      if train_label_FUDS_o[i, 0, 4] == 1:
         train_label_FUDS.append("Vis")
   for i in range(0, len(val_label_FUDS_o)):
      if val_label_FUDS_o[i, 0, 0] == 1:
         val_label_FUDS.append("Cor")
      if val_label_FUDS_o[i, 0, 1] == 1:
         val_label_FUDS.append("Isc")
      if val_label_FUDS_o[i, 0, 2] == 1:
         val_label_FUDS.append("Noi")
      if val_label_FUDS_o[i, 0, 3] == 1:
         val_label_FUDS.append("Nor")
      if val_label_FUDS_o[i, 0, 4] == 1:
         val_label_FUDS.append("Vis")
   train_label_FUDS = np.array(train_label_FUDS)
   val_label_FUDS = np.array(val_label_FUDS)

   train_data_US06_o, train_label_US06_o, val_data_US06_o, val_label_US06_o = dataset_get.Generate_Dataset_from_Dir("./Dataset", "US06")

   for i in range(0, len(train_data_US06_o)):
      current = np.array(torch.tensor(train_data_US06_o[i]).permute(2, 1, 0))
      np.random.shuffle(current)
      current = np.array(torch.tensor(current).permute(2, 1, 0))
      train_data_US06_o[i] = current

   if one_channel == 1:
      train_data_US06_o, val_data_US06_o = train_data_US06_o[:, 0, :, :], val_data_US06_o[:, 0, :, :]

   train_data_US06 = train_data_US06_o.reshape((train_data_US06_o.shape[0], -1))
   val_data_US06 = val_data_US06_o.reshape((val_data_US06_o.shape[0], -1))
   train_label_US06 = []
   val_label_US06 = []
   for i in range(0, len(train_label_US06_o)):
      if train_label_US06_o[i, 0, 0] == 1:
         train_label_US06.append("Cor")
      if train_label_US06_o[i, 0, 1] == 1:
         train_label_US06.append("Isc")
      if train_label_US06_o[i, 0, 2] == 1:
         train_label_US06.append("Noi")
      if train_label_US06_o[i, 0, 3] == 1:
         train_label_US06.append("Nor")
      if train_label_US06_o[i, 0, 4] == 1:
         train_label_US06.append("Vis")
   for i in range(0, len(val_label_US06_o)):
      if val_label_US06_o[i, 0, 0] == 1:
         val_label_US06.append("Cor")
      if val_label_US06_o[i, 0, 1] == 1:
         val_label_US06.append("Isc")
      if val_label_US06_o[i, 0, 2] == 1:
         val_label_US06.append("Noi")
      if val_label_US06_o[i, 0, 3] == 1:
         val_label_US06.append("Nor")
      if val_label_US06_o[i, 0, 4] == 1:
         val_label_US06.append("Vis")
   train_label_US06 = np.array(train_label_US06)
   val_label_US06 = np.array(val_label_US06)

   # X_train, y_train, X_test, y_test = train_data_UDDS, train_label_UDDS, val_data_UDDS, val_label_UDDS
   X_train, y_train = train_data_US06, train_label_US06
   # X_train, y_train = train_data_FUDS, train_label_FUDS

   X_test, y_test = val_data_UDDS, val_label_UDDS
   X_test_2, y_test_2 = val_data_FUDS, val_label_FUDS
   X_test_3, y_test_3 = val_data_US06, val_label_US06


   if os.path.isfile("1c_svm_model_US06.pkl"):
      print("Loading")
      svm = pickle.load(open("1c_svm_model_US06.pkl", "rb"))
   else:
      print("Training")
      svm = SVC(kernel='rbf', gamma=0.01)  # 3通道时默认0.001
      svm.fit(X_train, y_train)
      pickle.dump(svm, open("1c_svm_model_US06.pkl", "wb"))

   print("Testing[Train]...\n")

   right = 0
   total = 0
   num_o_cor = 0
   num_cor = 0
   num_o_isc = 0
   num_isc = 0
   num_o_noi = 0
   num_noi = 0
   num_o_nor = 0
   num_nor = 0
   num_o_vis = 0
   num_vis = 0
   for x, y in zip(X_train, y_train):
      x = x.reshape(1, -1)
      prediction = svm.predict(x)[0]
      if y == prediction:
         right += 1
      total += 1
      if y == "Cor":
         num_o_cor += 1
         if prediction == "Cor":
            num_cor += 1
      if y == "Isc":
         num_o_isc += 1
         if prediction == "Isc":
            num_isc += 1
      if y == "Noi":
         num_o_noi += 1
         if prediction == "Noi":
            num_noi += 1
      if y == "Nor":
         num_o_nor += 1
         if prediction == "Nor":
            num_nor += 1
      if y == "Vis":
         num_o_vis += 1
         if prediction == "Vis":
            num_vis += 1

   if num_cor == 0:
      acc_cor = 0
   else:
      acc_cor = float(num_cor) / float(num_o_cor) * 100
   if num_isc == 0:
      acc_isc = 0
   else:
      acc_isc = float(num_isc) / float(num_o_isc) * 100
   if num_noi == 0:
      acc_noi = 0
   else:
      acc_noi = float(num_noi) / float(num_o_noi) * 100
   if num_nor == 0:
      acc_nor = 0
   else:
      acc_nor = float(num_nor) / float(num_o_nor) * 100
   if num_vis == 0:
      acc_vis = 0
   else:
      acc_vis = float(num_vis) / float(num_o_vis) * 100

   accuracy = float(right) / float(total) * 100

   print(str(accuracy) + "% accuracy")
   print(str(acc_cor) + "% acc_cor")
   print(str(acc_isc) + "% acc_isc")
   print(str(acc_noi) + "% acc_noi")
   print(str(acc_nor) + "% acc_nor")
   print(str(acc_vis) + "% acc_vis")
   print("Manual Testing\n")


   print("Testing[UDDS]...\n")

   right = 0
   total = 0
   num_o_cor = 0
   num_cor = 0
   num_o_isc = 0
   num_isc = 0
   num_o_noi = 0
   num_noi = 0
   num_o_nor = 0
   num_nor = 0
   num_o_vis = 0
   num_vis = 0
   for x, y in zip(X_test, y_test):
      x = x.reshape(1, -1)
      prediction = svm.predict(x)[0]
      if y == prediction:
         right += 1
      total += 1
      if y == "Cor":
         num_o_cor += 1
         if prediction == "Cor":
            num_cor += 1
      if y == "Isc":
         num_o_isc += 1
         if prediction == "Isc":
            num_isc += 1
      if y == "Noi":
         num_o_noi += 1
         if prediction == "Noi":
            num_noi += 1
      if y == "Nor":
         num_o_nor += 1
         if prediction == "Nor":
            num_nor += 1
      if y == "Vis":
         num_o_vis += 1
         if prediction == "Vis":
            num_vis += 1

   if num_cor == 0:
      acc_cor = 0
   else:
      acc_cor = float(num_cor) / float(num_o_cor) * 100
   if num_isc == 0:
      acc_isc = 0
   else:
      acc_isc = float(num_isc) / float(num_o_isc) * 100
   if num_noi == 0:
      acc_noi = 0
   else:
      acc_noi = float(num_noi) / float(num_o_noi) * 100
   if num_nor == 0:
      acc_nor = 0
   else:
      acc_nor = float(num_nor) / float(num_o_nor) * 100
   if num_vis == 0:
      acc_vis = 0
   else:
      acc_vis = float(num_vis) / float(num_o_vis) * 100

   accuracy = float(right) / float(total)*100

   print(str(accuracy) + "% accuracy")
   print(str(acc_cor) + "% acc_cor")
   print(str(acc_isc) + "% acc_isc")
   print(str(acc_noi) + "% acc_noi")
   print(str(acc_nor) + "% acc_nor")
   print(str(acc_vis) + "% acc_vis")
   print("Manual Testing\n")

   print("Testing[FUDS]...\n")

   right = 0
   total = 0
   num_o_cor = 0
   num_cor = 0
   num_o_isc = 0
   num_isc = 0
   num_o_noi = 0
   num_noi = 0
   num_o_nor = 0
   num_nor = 0
   num_o_vis = 0
   num_vis = 0
   for x, y in zip(X_test_2, y_test_2):
      x = x.reshape(1, -1)
      prediction = svm.predict(x)[0]
      if y == prediction:
         right += 1
      total += 1
      if y == "Cor":
         num_o_cor += 1
         if prediction == "Cor":
            num_cor += 1
      if y == "Isc":
         num_o_isc += 1
         if prediction == "Isc":
            num_isc += 1
      if y == "Noi":
         num_o_noi += 1
         if prediction == "Noi":
            num_noi += 1
      if y == "Nor":
         num_o_nor += 1
         if prediction == "Nor":
            num_nor += 1
      if y == "Vis":
         num_o_vis += 1
         if prediction == "Vis":
            num_vis += 1

   if num_cor == 0:
      acc_cor = 0
   else:
      acc_cor = float(num_cor) / float(num_o_cor) * 100
   if num_isc == 0:
      acc_isc = 0
   else:
      acc_isc = float(num_isc) / float(num_o_isc) * 100
   if num_noi == 0:
      acc_noi = 0
   else:
      acc_noi = float(num_noi) / float(num_o_noi) * 100
   if num_nor == 0:
      acc_nor = 0
   else:
      acc_nor = float(num_nor) / float(num_o_nor) * 100
   if num_vis == 0:
      acc_vis = 0
   else:
      acc_vis = float(num_vis) / float(num_o_vis) * 100

   accuracy = float(right) / float(total) * 100

   print(str(accuracy) + "% accuracy")
   print(str(acc_cor) + "% acc_cor")
   print(str(acc_isc) + "% acc_isc")
   print(str(acc_noi) + "% acc_noi")
   print(str(acc_nor) + "% acc_nor")
   print(str(acc_vis) + "% acc_vis")

   print("Manual Testing\n")

   print("Testing[US06]...\n")

   right = 0
   total = 0
   num_o_cor = 0
   num_cor = 0
   num_o_isc = 0
   num_isc = 0
   num_o_noi = 0
   num_noi = 0
   num_o_nor = 0
   num_nor = 0
   num_o_vis = 0
   num_vis = 0
   for x, y in zip(X_test_3, y_test_3):
      x = x.reshape(1, -1)
      prediction = svm.predict(x)[0]
      if y == prediction:
         right += 1
      total += 1
      if y == "Cor":
         num_o_cor += 1
         if prediction == "Cor":
            num_cor += 1
      if y == "Isc":
         num_o_isc += 1
         if prediction == "Isc":
            num_isc += 1
      if y == "Noi":
         num_o_noi += 1
         if prediction == "Noi":
            num_noi += 1
      if y == "Nor":
         num_o_nor += 1
         if prediction == "Nor":
            num_nor += 1
      if y == "Vis":
         num_o_vis += 1
         if prediction == "Vis":
            num_vis += 1

   if num_cor == 0:
      acc_cor = 0
   else:
      acc_cor = float(num_cor) / float(num_o_cor) * 100
   if num_isc == 0:
      acc_isc = 0
   else:
      acc_isc = float(num_isc) / float(num_o_isc) * 100
   if num_noi == 0:
      acc_noi = 0
   else:
      acc_noi = float(num_noi) / float(num_o_noi) * 100
   if num_nor == 0:
      acc_nor = 0
   else:
      acc_nor = float(num_nor) / float(num_o_nor) * 100
   if num_vis == 0:
      acc_vis = 0
   else:
      acc_vis = float(num_vis) / float(num_o_vis) * 100

   accuracy = float(right) / float(total) * 100

   print(str(accuracy) + "% accuracy")
   print(str(acc_cor) + "% acc_cor")
   print(str(acc_isc) + "% acc_isc")
   print(str(acc_noi) + "% acc_noi")
   print(str(acc_nor) + "% acc_nor")
   print(str(acc_vis) + "% acc_vis")

   print("Manual Testing\n")
print("success")



# os.system("pause")



