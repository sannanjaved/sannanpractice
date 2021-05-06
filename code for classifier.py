
# coding: utf-8

# In[7]:


import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#from keras import backend as K
#K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


# In[8]:


DATADIR="D:\FYP\Binary_classifier"
Gestures=["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
training_data=[]
def create_training_data(IMG_SIZE):
    for category in Gestures:
        path=os.path.join(DATADIR,category)
        class_num=Gestures.index(category)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img),0)
            new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            training_data.append([new_array,class_num])
SIZE=50
create_training_data(SIZE)


# In[9]:


print(len(training_data))


# In[10]:


training_data=shuffle(training_data)


# In[11]:


X=[]
Y=[]
IMG_SIZE=50
for gestures,labels in training_data:
    X.append(gestures)
    Y.append(labels)

X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X = X/255


# In[12]:


num_classes=24
Y = np_utils.to_categorical(Y, num_classes)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# In[14]:


print(len(X_train))
print(len(X_test))


# In[15]:


cnn_model=Sequential()
input_shape=X_train.shape[1:]
cnn_model.add(Convolution2D(64, (3,3),border_mode='same',input_shape=input_shape))
cnn_model.add(Activation('relu'))
cnn_model.add(Convolution2D(64, (3, 3)))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.5))

#cnn_model.add(Convolution2D(64, (3, 3)))
#cnn_model.add(Activation('relu'))
#cnn_model.add(Convolution2D(64, (3, 3)))
#cnn_model.add(Activation('relu'))
#cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
#cnn_model.add(Dropout(0.5))

cnn_model.add(Flatten())
cnn_model.add(Dense(64))
cnn_model.add(Activation('relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(num_classes))
cnn_model.add(Activation('softmax'))

cnn_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

cnn_model.summary()
cnn_model.get_config()
cnn_model.layers[0].get_config()
cnn_model.layers[0].input_shape
cnn_model.layers[0].output_shape
cnn_model.layers[0].get_weights()
np.shape(cnn_model.layers[0].get_weights()[0])
cnn_model.layers[0].trainable


# In[16]:


hist = cnn_model.fit(X_train, y_train, batch_size=16, epochs=3,verbose=1, validation_split=0.3)


# In[17]:


score = cnn_model.evaluate(X_test, y_test,  verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])


# In[18]:


from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = cnn_model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = cnn_model.predict_classes(X_test)
#print(y_pred)
target_names = ['class 0(A)', 'class 1(B)', 'class 2(C)','class 3(D)', 'class 4(E)','class 5(F)','class 6(G)','class 7(H)','class 8(I)','class 9(K)','class 10(L)','class 11(M)','class 12(N)','class 13(O)','class 14(P)','class 15(Q)','class 16(R)','class 17(S)','class 18(T)','class 19(U)','class 19(V)','class 19(W)','class 19(X)','class 19(Y)']
Y_test=np.argmax(y_test,axis=1)
print(classification_report(Y_test, y_pred,target_names=target_names))

print(confusion_matrix(Y_test, y_pred))


# In[19]:


get_ipython().magic('matplotlib inline')
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(Y_test, y_pred))

np.set_printoptions(precision=2)

plt.figure(figsize=(11,11))

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')


# In[20]:


from keras.models import load_model
cnn_model.save('binary_classifier_500_bin.hdf5')
