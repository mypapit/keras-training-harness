'''

Training harness for Keras Sequential Model 
Compatible with Tensorflow  2.4.0 and above

This training harness has been modified and re-written 
extensively by:
Mohammad Hafiz bin Ismail (mypapit@gmail.com)
September 2020


Please cite the work accordingly as
Ismail, M.H (2022) Training harness with Classification Report for Keras Model v2.


'''

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import cv2
from tensorflow.keras.utils import to_categorical

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,SeparableConv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Input
from tensorflow.keras.layers import BatchNormalization


from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping





from tensorflow.keras.callbacks import TensorBoard,LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

from imutils import paths

import sklearn
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import seaborn as sns
import pandas as pd
import numpy as np




# change the pre-trained model here

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.xception import Xception


##################### CONFIG AREA ##############################
#
#
#    CONFIGURATION AREA
#
#
################################################################


# dimensions of input image.
img_width = 128
img_height = 128

#folder training images, ada juga projek yang guna nama lain
# yang penting, pastikan nama training folder ni ditukar
# kepada sepatutnya

TRAINING_ROOT = "train"

# nama file label
LABEL_FILENAME = "tlabels.txt"




# nama model / Boleh letak nama model sendiri Be creative 
MODEL_NETWORK_NAME = "MobileNet"

# folder dimana kita akan savekan model selepas training
# boleh saja ubah nama model ni
MODEL_SAVE = "saved_model\model-" + MODEL_NETWORK_NAME
MODEL_SAVE_WEIGHT = MODEL_NETWORK_NAME + "_weights.h5"



#
# Boleh Kawal bilangan epoch di sini
#
EPOCHS = 8
#
# boleh guna mana2 optimizers sama ada adam, sgd, atau lain2
# learning rate pun boleh setting sini
# default value lr = 1e-3
#

learning_rate = 6e-4
optim=optimizers.Adam(learning_rate=learning_rate)
#optim= optimizers.SGD(learning_rate=learning_rate,momentum=0.75)


###########################
##
## Training/validataion split
##
##  default train_valid_split = 0.25 
##
##  common value = 0.2, 0.25, 0.3,0.5
##
############################
train_valid_split = 0.25


##############################
##
## Random seed value - boleh guna apa-apa value
## Purpose : untuk repeatable experiment 
## 
##
#############################
random_seed = 8755


#################################
#
# Data augmentation? True/False.
#    
# True = each image will be augmented
# False = the image will be processed as it is
# 
# Recommend to set it to True.
###################################
data_augmentation = True


#
#
# Change this for data augmentation
#
#  arotate = maximum random rotation
#  azoom = maximum image zoom
#  ashear = maximum shearing factor for image
#  hflip = whether to randomly flip the image
#
arotate=30
azoom=0.20
ashear=0.15
hflip=True




early_stopping = False
es = EarlyStopping(monitor='val_accuracy', mode='max',verbose=1,min_delta=0.05,patience=10)



############ END OF CONFIG AREA ###################################






IMAGE_SHAPE = (img_width,img_height)

batch_size = 60
chanDim = -1

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)




#
# function yang ditulis sendiri untuk memudahkan automatic image 
# classification label  generation and saving
#
def read_write_labels(TRAINING,LABEL):
		list = os.listdir(TRAINING)
		try:
			labelfile = open(LABEL_FILENAME,"w")	
			for lines in list:
				
				labelfile.write(lines)
				labelfile.write("\n")

				
		except:
			print("I/O Label Exception")
			pass
			
		finally:
			labelfile.close()


		imagenet_labels = np.array(open(LABEL_FILENAME).read().splitlines())

		
		print("Detected labels: {0}, count : {1}".format(imagenet_labels,len(imagenet_labels)))
		
		
		return imagenet_labels


imagePaths = list(paths.list_images(TRAINING_ROOT))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename, load the image, and
	# resize it to be a fixed 64x64 pixels, ignoring aspect ratio
	label = imagePath.split(os.path.sep)[-2]
	try:
		image = cv2.imread(imagePath)
#		print(imagePath)
		image = cv2.resize(image, (img_width, img_height))
	except Exception as ex:
		print("Exception {0}".format(ex))
		print("Problem at - {0}".format(imagePath))
		pass
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)
    

data = np.array(data, dtype="float") / 255.0




    


imagenet_labels = read_write_labels(TRAINING_ROOT,LABEL_FILENAME)

numclasses = imagenet_labels.shape[0]

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, numclasses)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=train_valid_split, random_state=random_seed)

 


model = Sequential() 
model.add(MobileNetV2 (weights='imagenet',include_top=False,input_shape=(img_width,img_height,3), classes=numclasses))
model.add(MaxPooling2D(2,2))
model.add(GlobalAveragePooling2D())
#model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(24, activation='relu'))
#model.add(BatchNormalization())





model.add(Dense(numclasses,activation="softmax"))




# summarize layers
print(model.summary())

# plot graph
plot_model(model, to_file=MODEL_NETWORK_NAME + '_layer.png',show_shapes=True)

    

##########################################
#
#jika lebih 10 class, kita masukkan top5 accuracy kedalam pengiraan
#
###############################################
if (numclasses >=10) :
    metrics=[
        CategoricalAccuracy(name="accuracy"),
        TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ]
else:
    metrics=[
        CategoricalAccuracy(name="accuracy")        
    ]
        
        
        

model.compile(loss='categorical_crossentropy',
              optimizer=optim,
              metrics=metrics)


######
#  Tensorboard dump
#
#####
tensorboard = TensorBoard(log_dir='output\\Graph', histogram_freq=0, write_graph=True, write_images=True)

cb = [tensorboard]

if (early_stopping == True) :
    cb = [tensorboard, es]


#################
###
###
####
####
print("[INFO] performing 'on the fly' data augmentation")

if (data_augmentation == True) :
    aug = ImageDataGenerator(
    		rotation_range=arotate,
    		zoom_range=azoom,
    		shear_range=ashear,
            width_shift_range=0.10,
            height_shift_range=0.10,
    		horizontal_flip=hflip,
    		fill_mode="nearest")
else :
    aug = ImageDataGenerator(
    		fill_mode="nearest")
    
training_aug=aug.flow(trainX, trainY);
    
trainsteps=training_aug.n // training_aug.batch_size

print ("[[[ Training Information ]]]")
print ("Training samples {0}, training batch {1}, training steps {2}".format(training_aug.n, training_aug.batch_size,trainsteps))




# kita start training di sini
history=model.fit(
    x=training_aug,
    epochs=EPOCHS,
    validation_data=(testX,testY),
    batch_size=training_aug.batch_size,
    steps_per_epoch = training_aug.n // training_aug.batch_size,
    callbacks = cb
)


 

model.save_weights(MODEL_SAVE_WEIGHT)
model.save(MODEL_SAVE)

################ END OF TRAINING ####################
def showpix(cimg):
    kcimg = np.float32(cimg)
    kcimg = cv2.cvtColor(kcimg,cv2.COLOR_BGR2RGB)
    plt.imshow(kcimg)


###
#
#
# code untuk buat gambarajah confusion matrix yang normalized
#

proba = model.predict(testX)
ypred = np.argmax(proba,axis=-1)
valid_labels= np.argmax(testY,axis=-1)
result_label_batch = imagenet_labels[ypred]



confusion = tf.math.confusion_matrix(labels=valid_labels, predictions=ypred).numpy()
conmat = np.around(confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis], decimals=3)
conmat = pd.DataFrame(conmat,index = imagenet_labels, columns = imagenet_labels)

###
###
################## PLOTTING AUGMENTED IMAGES ################ 
###
###

if (data_augmentation == True) :
    augpix, _ = training_aug[0]
    augarray = training_aug.index_array[:30]
    
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(hspace=0.20)
    for n in range(30):
      plt.subplot(5,6,n+1)
      showpix(augpix[n])
      
      color = "green" if ypred[n] == valid_labels[n] else "red"
      plt.axis('off')
    _ = plt.suptitle("Sample Augmented Images")
    plt.tight_layout()
    plt.show()
    
    
    plt.clf()
    plt.close()


###
###
################## PLOTTING ORIGINAL IMAGES ################ 
###
###
if (data_augmentation 
    == True) :
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(hspace=0.20)
    
    num_i=1
    for n in augarray:
      plt.subplot(5,6,num_i)
      num_i=num_i+1
      showpix(trainX[n])
      plt.axis('off')
    _ = plt.suptitle("Sample Original Images")
    plt.tight_layout()
    plt.show()
    
    
    plt.clf()
    plt.close()


##
##
############ PLOTTING SAMPLE IMAGE PREDICTION
##
##

plt.figure(figsize=(12,10))
plt.subplots_adjust(hspace=0.20)
for n in range(30):
  stat = "{0} ({1:.2f})".format(result_label_batch[n],proba[n][ypred[n]])
  plt.subplot(5,6,n+1)
  
  #convert back the color
  cimg = np.float32(testX[n])
  cimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2RGB)
  plt.imshow(cimg)
  
  color = "green" if ypred[n] == valid_labels[n] else "red"
  plt.title(stat,color=color)
  plt.axis('off')
_ = plt.suptitle(MODEL_NETWORK_NAME)

plt.show()


plt.clf()
plt.close()



figure = plt.figure(figsize=(8, 8))
sns.heatmap(conmat, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.clf()
plt.close()


print ("plotting accuracy graph")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylim([0.0,1.00])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

   
plt.clf()
plt.close()

    
print ("plotting loss graph")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()



##########################################
#
#jika lebih 10 class, kita masukkan top5 accuracy kedalam pengiraan
#
###############################################
if (numclasses>=10):
    test_loss,test_accuracy,test_top5accuracy=model.evaluate(testX,testY)
else:
    test_loss,test_accuracy=model.evaluate(testX,testY)





print("\n\n=== Statistics For Whole Validation Set ===")
print("Labels/Classes : ")
print(imagenet_labels)
print("\nTest Accuracy: {:.6f}, Test Loss: {:.6f}".format(test_accuracy,test_loss))

if (numclasses>=10):
    print("Validation Top-K (Top 5) Accuracy: {:.4f}".format(test_top5accuracy))


print("==============\n\n")


#########################################
#
# ok di sini bahagian untuk kira accuracy/precision/recall
# guna library sklearn saja untuk kira semua metrics ni
# Kiraan untuk keseluruhan test set
#
#####################

accuracy_sc = accuracy_score(valid_labels,ypred)
precision =precision_score(valid_labels,ypred,average='weighted')
recall =recall_score(valid_labels,ypred,average='weighted')
f1 =f1_score(valid_labels,ypred,average='weighted')
kappa = cohen_kappa_score(valid_labels,ypred)
matrix = confusion_matrix(valid_labels,ypred)

classreport = classification_report(valid_labels,ypred,target_names=imagenet_labels)

print("Accuracy : {:.4f}, Precision : {:.4f}, Recall : {:.4f}".format(accuracy_sc,precision,recall) )
print("F1: {:.4f}, Cohen-Kappa : {:4f}".format(f1, kappa))

print("\n\n=== Confusion matrix === ")
print(matrix)

print("\n\n=== Classification Report ===\n")
print(classreport)





