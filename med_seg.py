import os
import glob
import numpy as np
import cv2
import keras
from matplotlib import pyplot as plt
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import concatenate

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import EarlyStopping



#from tqdm import tqdm
img_dir='images'
data_path=os.path.join(img_dir,'*.jpg')
files=glob.glob(data_path)
data=[]
rows=256
cols=256
for file in files:
    print(file)
    img=cv2.imread(file)
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resized_img=cv2.resize(gray_img,(rows,cols))
    data.append(resized_img)



data=np.array(data,np.float32)/255
data=np.expand_dims(data,axis=3)
print(data.shape)

train_x,val_x,train_ground,val_ground=train_test_split(data,data,test_size=0.2,random_state=13)

#batch_size = 64
epochs = 20
inChannel = 1
x, y = 256, 256
input_img = Input(shape = (x, y, inChannel))

def conv2d(input_img,n_filters,size=(3,3),batchnorm=True):
    conv=Conv2D(filters=n_filters,kernel_size=size,kernel_initializer='he_normal',padding='same')(input_img)
    if batchnorm:
        conv=keras.layers.BatchNormalization()(conv)
    conv=keras.layers.Activation('relu')(conv)
    
    conv=Conv2D(filters=n_filters,kernel_size=size,kernel_initializer='he_normal',padding='same')(input_img)
    if batchnorm:
        conv=keras.layers.BatchNormalization()(conv)
    conv=keras.layers.Activation('relu')(conv)
    return conv


def unet_autoencoder(input_img,n_filters=16,dropout=0.1,batchnorm=True):
    #Encoder
    conv1=conv2d(input_img,n_filters*1,size=(3,3),batchnorm=batchnorm)
    pool1=MaxPooling2D((2,2))(conv1)
    pool1=keras.layers.Dropout(dropout)(pool1)
    
    conv2=conv2d(pool1,n_filters*2,size=(3,3),batchnorm=batchnorm)
    pool2=MaxPooling2D((2,2))(conv2)
    pool2=keras.layers.Dropout(dropout)(pool2)
    
    conv3=conv2d(pool2,n_filters*4,size=(3,3),batchnorm=batchnorm)
    pool3=MaxPooling2D((2,2))(conv3)
    pool3=keras.layers.Dropout(dropout)(pool3)
    
    conv4=conv2d(pool3,n_filters*8,size=(3,3),batchnorm=batchnorm)
    print(conv4.shape)
    pool4=MaxPooling2D((2,2))(conv4)
    pool4=keras.layers.Dropout(dropout)(pool4)
    
    conv5=conv2d(pool4,n_filters*16,size=(3,3),batchnorm=batchnorm)
    
    #Decoder
    up6=Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(conv5)
    print(up6.shape)
    up6=concatenate([up6,conv4])
    up6=keras.layers.Dropout(dropout)(up6)
    conv6=conv2d(up6,n_filters*8,size=(3,3),batchnorm=batchnorm)
    
    up7=Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(conv6)
    up7=concatenate([up7,conv3])
    up7=keras.layers.Dropout(dropout)(up7)
    conv7=conv2d(up7,n_filters*4,size=(3,3),batchnorm=batchnorm)
    
    up8=Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(conv7)
    up8=concatenate([up8,conv2])
    up8=keras.layers.Dropout(dropout)(up8)
    conv8=conv2d(up8,n_filters*2,size=(3,3),batchnorm=batchnorm)
    
    up9=Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(conv8)
    up9=concatenate([up9,conv1])
    up9=keras.layers.Dropout(dropout)(up9)
    conv9=conv2d(up9,n_filters*1,size=(3,3),batchnorm=batchnorm)
    
    output=Conv2D(1,(1,1),activation='sigmoid')(conv9)
    model=Model(input_img,output)
    return model

autoencoder=unet_autoencoder(input_img,n_filters=16,dropout=0.1,batchnorm=True)
autoencoder.compile(loss='mean_squared_error',optimizer=Adam())
autoencoder.summary()



with tf.device('/gpu:0'):
    train_autoencoder = autoencoder.fit(train_x, train_ground,batch_size=16,epochs=epochs,verbose=2,validation_data=(val_x, val_ground),callbacks=[EarlyStopping(monitor='val_loss',patience=3)])



loss = train_autoencoder.history['loss']
val_loss = train_autoencoder.history['val_loss']
epochs = range(0,8)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



pred=autoencoder.predict(val_x)



plt.figure(figsize=(20, 4))
print("Validation Images:")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(val_x[i, ..., 0], cmap='brg')  
plt.show()



plt.figure(figsize=(20, 4))
print(" Reconstructed Validation Images:")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(pred[i, ..., 0], cmap='brg')  
plt.show()

