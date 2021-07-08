
# # TASK 2


# ===============================================================================================================
# # Task 2.1 Prep 
# ===============================================================================================================

#package
import tensorflow
import numpy as np
import sklearn
import keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical 
from keras.callbacks import History 
from keras.layers import GaussianNoise

# ===============================================================================================================
#load data
with np.load("training-dataset.npz") as data:
    img = data["x"]
    lbl = data["y"]

#for 26 classes - 0 base index
lbl=lbl-1

print(img.shape)
print(lbl.shape)
#imgl=img[:1000]
#lbll=lbl[:1000]

# ===============================================================================================================
# Task 1 - Split Data into 80-10-10 Train-Test-Val

# train- val - test - random
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(img, lbl, test_size=0.10, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(0.10 / 0.90), random_state=1)

# ===============================================================================================================
# Explortory data analysis

# import matplotlib.pyplot as plt
# import cv2

# print(len(img[1]))

# imgplot = plt.imshow(img[347].reshape(28,28))
# #plt.show()

# captcha = np.load('test-dataset.npy')
# imgplot = plt.imshow(captcha[347])
# #plt.show()

# =============================================================================
# Prepocessing
#data preprocess stages - reshaping to 4 dimensionals
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# change the type as float
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_val = X_val.astype("float32")

# Normalize the data 
X_train /= 255
X_test /= 255
X_val /= 255

# One-hot coding
num_classes = 26
Y_train = keras.utils.to_categorical(y_train, num_classes)
Y_test = keras.utils.to_categorical(y_test, num_classes)
Y_val = keras.utils.to_categorical(y_val, num_classes)


# ===============================================================================================================
# # Task 2.2 Classifer
# ===============================================================================================================
# --------------------------------------------------------------------------------------------------------------
# Tunning - add noise directly on the classifer 
# --------------------------------------------------------------------------------------------------------------

# BUILD CONVOLUTIONAL NEURAL NETWORKS
model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization())
model.add(Dropout(0.2)) 

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization())
model.add(Dropout(0.2)) 

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2)) 
model.add(Dense(26, activation='softmax'))

#COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#DECREASE LEARNING RATE BY 0.95 EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# TRAIN CNNs AND DISPLAY ACCURACIES
epochs = 8

fit_model = model.fit(X_train,Y_train, batch_size=32, epochs = epochs, 
                   validation_data = (X_val,Y_val), callbacks=[annealer], verbose=1)

# # list all data in history
model.summary()

scores = model.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: %.2f%%" % (scores[1] * 100))

# ===============================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import cv2

#loading the data
captcha = np.load('test-dataset.npy')

#creating the array where will store the 28x28 pictures
letters = np.empty([len(captcha),4,28,28])

# ===============================================================================================================
# # Task 2.3. Segmenting Function
# ===============================================================================================================
# A function which takes img_input as an array of the noisy four letter images and 
# segments its img_number into in the appropriate row of segments.
# ===============================================================================================================
# --------------------------------------------------------------------------------------------------------------
#a function which cut an image into four letters
def segmenting(img_input, segments, img_number):

    #loading the image
    image = img_input[img_number]
    
    weighted = cv2.addWeighted(image, alpha=255, src2=0, beta=0, gamma=0, dtype=cv2.CV_8U)
    
    # Apply adaptive threshold
    thresh = cv2.threshold(weighted,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    #sorting the contours by size and taking only the 4 biggest, storing the width values in decrasing order:
    contours.sort(key=lambda s: -len(s))
    contours = contours[0:4]
    w_ord = np.empty(4)
    cut = False
    for i in range(4):
        if cut == False:
            w_ord[i] = cv2.boundingRect(contours[i])[2]
            if len(contours[i]) < 15:
                contours = contours[0:i]
                cut = True
    w_ord = sorted(w_ord, reverse = True)
    #i only want one contour with the biggest w...
    for i in range(3):
        if w_ord[0] == w_ord[i+1]:
            w_ord[i] -= 0.5
    
    #sorting the contours (from left to right order):
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
      
    #the input image of the contours function:
    #imgplot = plt.imshow(thresh)
    #plt.show()
    
    # For each contour, find the bounding rectangle and store it in a 10kx4 matrix
    ltr_number = 0
    #print('num of cnts:',len(contours))
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #print('len:',len(cnt),'x:',x,'w:',w,'y:',y,'h:',h)
        #print(w_ord[0])
        #if we have to split the contour into 4 letters
        if len(contours) == 1:
            if x+w/2-56 < 0:
                w = 112 - 2*x
            if x+w/2+56 > 130:
                w = 148 - 2*x 
            #left letter
            letter = image[1:29, int(x+w/2)-56:int(x+w/2)-28]
            segments[img_number,ltr_number] = letter
            ltr_number += 1        
            #middle left letter
            letter = image[1:29, int(x+w/2)-28:int(x+w/2)]
            segments[img_number,ltr_number] = letter
            ltr_number += 1         
            #middle right letter
            letter = image[1:29, int(x+w/2):int(x+w/2)+28]
            segments[img_number,ltr_number] = letter
            ltr_number += 1         
            #right cetter
            letter = image[1:29, int(x+w/2)+28:int(x+w/2)+56]
            segments[img_number,ltr_number] = letter
            ltr_number += 1      
        #if we have split the contour in 2 letters ()
        elif len(contours) == 2 and w_ord[0] > w_ord[1] + 28 and w == w_ord[0]:
            if x+w/2-42 < 0:
                w = 84 - 2*x
            if x+w/2+42 > 130:
                w = 176 - 2*x 
            #left letter
            letter = image[1:29, int(x+w/2)-42:int(x+w/2)-14]
            segments[img_number,ltr_number] = letter
            ltr_number += 1        
            #middle letter
            letter = image[1:29, int(x+w/2)-14:int(x+w/2)+14]
            segments[img_number,ltr_number] = letter
            ltr_number += 1         
            #right letter
            letter = image[1:29, int(x+w/2)+14:int(x+w/2)+42]
            segments[img_number,ltr_number] = letter
            ltr_number += 1     
        #if there is two contours and both contains 2 letters
        elif len(contours) == 2 and w_ord[0] < w_ord[1] + 29 or len(contours) == 3 and w == w_ord[0]:
            if x+w/2-28 < 0:
                w = 56 - 2*x
            if x+w/2+28 > 130:
                w = 204 - 2*x 
            #left letter
            letter = image[1:29, int(x+w/2)-28:int(x+w/2)]
            segments[img_number,ltr_number] = letter
            ltr_number += 1    
            #right letter
            letter = image[1:29, int(x+w/2):int(x+w/2)+28]
            segments[img_number,ltr_number] = letter
            ltr_number += 1         
        else:
            #cutting on letter from a contour
            if x+w/2-14 < 0:
                w = 28 - 2*x
            if x+w/2+14 > 130:
                w = 232 - 2*x 
            letter = image[1:29, int(x+w/2)-14:int(x+w/2)+14]
            segments[img_number,ltr_number] = letter
            ltr_number += 1           
                
#segmenting(captcha,letters,5512)
segmenting(captcha,letters,195)


# ===============================================================================================================
# # Task 2.3.1 Segmenting the whole dataset
# ===============================================================================================================
# Use the previously written segmenting function and do the segmentizing for the 10k images. 
# After it we can test the results by showing random images.
# ===============================================================================================================

#segmenting over the whole dataset
for i in range(len(captcha)):
    segmenting(captcha,letters,i)

#testing the segmenting proccess with a randomly choosen image from the dataset
k = np.random.randint(0,len(captcha))
#k = 5512 3 letters in one contour
k = 12
print('Image and segments for case:',k)
plt.imshow(captcha[k])
plt.show()
plt.subplot(2,2,1)
plt.imshow(letters[k,0])
plt.subplot(2,2,2)
plt.imshow(letters[k,1])
plt.subplot(2,2,3)
plt.imshow(letters[k,2])
plt.subplot(2,2,4)
plt.imshow(letters[k,3])
plt.show()

# ===============================================================================================================
# # Task 2.4. Making the predictions for the cropped letters
# ===============================================================================================================
#making the letters be able to run with model1
x = np.empty([len(captcha)*4,784])
for i in range(len(captcha)):
    for j in range(4):
        m = 0
        for k in range(28):
            for l in range(28):
                x[4*i+j,m] = (letters[i,j,k,l])
                m += 1
x = np.int64(x)
x=np.reshape(x,(len(captcha)*4,28,28,1))
predictions2 = model.predict(x)
y = model.predict_classes(x)

# ===============================================================================================================
# # Task 2.5. Testing the segmenting and the classification
# ===============================================================================================================
k = np.random.randint(0,len(captcha))
#k = 9998
print('Segments and classes for image:',k)
plt.imshow(captcha[k])
plt.show()
plt.subplot(2,2,1)
plt.imshow(letters[k,0])
plt.subplot(2,2,2)
plt.imshow(letters[k,1])
plt.subplot(2,2,3)
plt.imshow(letters[k,2])
plt.subplot(2,2,4)
plt.imshow(letters[k,3])
plt.show()

char1 = chr(ord('a') + y[4*k+0])
char2 = chr(ord('a') + y[4*k+1])
char3 = chr(ord('a') + y[4*k+2])
char4 = chr(ord('a') + y[4*k+3])
print('The classification:',char1,char2,char3,char4)

char1 = y[4*k+0]
char2 = y[4*k+1]
char3 = y[4*k+2]
char4 = y[4*k+3]
print('The classification:',char1,char2,char3,char4)

# ===============================================================================================================
# # Task 2.6. Making the 5 best predictions
# ===============================================================================================================
#creating the database for the top 5 predictions for all of the letters
ltr_top5 = np.empty([len(captcha),4,5])
ltr_index = np.empty([len(captcha),4,5])

#storing the 4*5 p-values
for i in range(len(captcha)):
    for j in range(4):
        for k in range(5):
            ltr_top5[i,j,k] = sorted(predictions2[4*i+j], reverse = True)[k]
            #print(np.where(predictions2[4*i+j] == ltr_top5[i,j,k])[0][0])
            ltr_index[i,j,k] = np.where(predictions2[4*i+j] == ltr_top5[i,j,k])[0][0]

# ===============================================================================================================
# # Task 2.6.1 Calculating the probability for all combination of the letters' top five predictions
# ===============================================================================================================
#calculating the p values for the 5^4 = 625 possible outcomes
pred_top5 = np.empty([len(captcha),625,2])
for i in range(len(captcha)):
    for j in range(625):
        #calculating the possibility of a cercent case (product of the 4 p values)
        pred_top5[i,j,0] = ltr_top5[i,0,j % 5] * ltr_top5[i,1,int(j/5) % 5] * ltr_top5[i,2,int(j/25) % 5] * ltr_top5[i,3,int(j/125) % 5]
        #making the 8 digit number code
        pred_top5[i,j,1] = int(1000000 * ltr_index[i,0,j % 5] + 10000 * ltr_index[i,1,int(j/5) % 5] + 100 * ltr_index[i,2,int(j/25) % 5] + ltr_index[i,3,int(j/125) % 5])+ 1010101
    #sort the predictions in decreasing order by the p values
    pred_top5[i] = pred_top5[i][(-pred_top5[i][:,0]).argsort()]    

# ===============================================================================================================
# # Task 2.6.2 Saving the predictions to the output file
# ===============================================================================================================

#saving the predictions into a csv file
np.savetxt("prediction.csv", pred_top5[:,:5,1].astype(int), fmt ='%08d', delimiter=",")

# ===============================================================================================================
# # Task 2.7. Testing the 5 guesses
# ===============================================================================================================
# A First a helper function decoding which takes the 8 number
# long integer code of a guess and returns the 4 letters which they represent
# ===============================================================================================================
#

def decoding(guess):
    char1 = chr(ord('a')-1 + int(guess/1000000))
    char2 = chr(ord('a')-1 + int(guess/10000) % 100)
    char3 = chr(ord('a')-1 + int(guess/100) % 100)
    char4 = chr(ord('a')-1 + int(guess) % 100)
    return char1 + " " + char2 + " " + char3 + " " + char4

#testing the program with random images
k = np.random.randint(0,10000)
#or with a speciific one: (remove the #)
#k = 0
print('Segments and classes for image:',k)
plt.imshow(captcha[k])
plt.show()
plt.subplot(2,2,1)
plt.imshow(letters[k,0])
plt.subplot(2,2,2)
plt.imshow(letters[k,1])
plt.subplot(2,2,3)
plt.imshow(letters[k,2])
plt.subplot(2,2,4)
plt.imshow(letters[k,3])
plt.show()

#Below you can see the output of the classification: it should be the first guess, so redundant, but can be useful for checking
# char1 = chr(ord('a') - 1 + y[4*k+0])
# char2 = chr(ord('a') - 1 + y[4*k+1])
# char3 = chr(ord('a') - 1 + y[4*k+2])
# char4 = chr(ord('a') - 1 + y[4*k+3])
# print('The classification:',char1,char2,char3,char4)

# char1 = y[4*k+0]
# char2 = y[4*k+1]
# char3 = y[4*k+2]
# char4 = y[4*k+3]
# print('The classification:',char1,char2,char3,char4)

# The 5 guesses with the highest p values
for i in range(5):
    print( "Guess",i+1,':',decoding(pred_top5[k,i,1]),"Code: {:08d}".format(int(pred_top5[k,i,1])),"Chance: {:0.2%}".format(pred_top5[k,i,0]))
