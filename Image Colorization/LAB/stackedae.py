#!/usr/bin/env python
# coding: utf-8

# # StackedAE Lab

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers
from sklearn.model_selection import train_test_split
import random
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import RMSprop


# In[2]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, UpSampling2D, Concatenate, InputLayer
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
import gc


# In[3]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.filters import threshold_otsu
from glob import glob
from scipy import misc
from matplotlib.patches import Circle,Ellipse
from matplotlib.patches import Rectangle


# In[4]:


import os
from PIL import Image
import scipy.misc
import imageio
from skimage.transform import rescale, resize
from skimage.color import lab2rgb


# In[5]:


from matplotlib import pyplot as plt
import numpy as np
import gzip


# In[6]:


tf.random.set_seed(42)


# # Data Augmentation

# In[7]:


from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras import backend as K
import tensorflow_addons as tfa
import tensorflow as tf

'''batch_size= 8
image_size = [120, 120]


ds = image_dataset_from_directory(
    '/kaggle/input/imagedataset/data',
    labels=None,
    image_size=image_size,
    interpolation='nearest',
    batch_size=batch_size,
    shuffle=True,
    color_mode='grayscale'
)

def convert_to_float(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def trans1(img):
    return tfa.image.rotate(tf.image.flip_left_right(tf.image.flip_up_down(img)),-.2,fill_mode="reflect",interpolation="bilinear")

def trans2(img):
    return tfa.image.rotate(img,-.2,fill_mode="reflect",interpolation="bilinear")

def trans3(img):
    return tfa.image.rotate(img,.2,fill_mode="reflect",interpolation="bilinear")
    
ds1,ds2,ds3,ds4 = ds,ds.map(trans1),ds.map(trans2),ds.map(trans3)

ds = ds1.concatenate(ds2).concatenate(ds3).concatenate(ds4)

AUTOTUNE = tf.data.experimental.AUTOTUNE
x = (
    ds
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)'''


# In[8]:


ab_path = "/kaggle/input/image-colorization/ab/ab/ab1.npy"
l_path = "/kaggle/input/image-colorization/l/gray_scale.npy"


# In[9]:


ab_df = np.load(ab_path)[0:3000]
L_df = np.load(l_path)[0:3000]
dataset = (L_df,ab_df )
gc.collect()


# In[10]:


def lab_to_rgb(L, ab):
    """
    Takes an image or a batch of images and converts from LAB space to RGB
    """
    L = L  * 100
    ab = (ab - 0.5) * 128 * 2
    Lab = np.concatenate([L, ab], dim=2).numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = Image.lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


# In[11]:


plt.figure(figsize=(30,30))
for i in range(1,16,2):
    plt.subplot(4,4,i)
    img = np.zeros((224,224,3))
    img[:,:,0] = L_df[i]
    plt.title('B&W')
    plt.imshow(lab2rgb(img))
    
    plt.subplot(4,4,i+1)
    img[:,:,1:] = ab_df[i]
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    plt.title('Colored')
    plt.imshow(img)


# In[12]:


ab_df.shape


# In[13]:


"""x = []
y = []
def read_images(data): #method to read images
    for i in range(len(data)):
        rgb_image = Image.open( data[i] ).resize( ( img_size , img_size ) )
        # Normalize the RGB image array
        rgb_img_array = (np.asarray( rgb_image ) ) / 255
        gray_image = rgb_image.convert( 'L' )
        # Normalize the grayscale image array
        gray_img_array = ( np.asarray( gray_image ).reshape( ( img_size , img_size , 1 ) ) ) / 255
        # Append both the image arrays
        x.append( gray_img_array )
        y.append( rgb_img_array )
    return x,y"""


# In[14]:


"""x, y = read_images(data) #calling readimage"""


# In[15]:


x = x = (L_df/255).astype('float32') 
y = (ab_df/255).astype('float32') 


# In[16]:


#plt.imshow(y[0])


# In[17]:


import tensorflow as tf
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split( x , y , test_size=0.1, random_state=42 )


# In[18]:


lr = 0.00005


# In[19]:


input_img = Input(shape = (224, 224, 1))


# In[20]:


def autoencoder(input_img): #functional model
    #encoder
    #input = 224 x 224 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #224 x 224 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #112 x 112 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #112 x 112 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #64 x 64 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #56 x 56 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #56 x 56 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # x 112 x 112
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(2, (3, 3), activation='softmax', padding='same')(up2) # 224 x 224 x 1
    return decoded


# In[21]:


autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop()) #compiling the model


# In[22]:


autoencoder.summary()


# In[ ]:





# In[23]:


def r_accuracy(img_original, img_reconstructed):
    mse = tf.reduce_mean((img_original - img_reconstructed) ** 2)
    pixel_max = 1.0
    psnr = 20 * tf.math.log(pixel_max / tf.math.sqrt(mse))/tf.math.log(10.0)
    return psnr


# In[24]:


# del x
# del y
# del L_df
# del ab_df


# In[25]:


history = autoencoder.fit(train_x, train_y, epochs=100, batch_size=32, verbose=1)


# In[26]:


pred = autoencoder.predict(test_x)


# In[27]:


import numpy as np
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim


# In[28]:


# Convert the images to numpy arrays
# img1 = tf.keras.preprocessing.image.img_to_array(pred[0])
# img2 = tf.keras.preprocessing.image.img_to_array(test_y[0])
final = []
predfinal = []
for i in range(600):
    img = np.zeros((224,224,3))
    img[:,:,0] = test_x[i]*255
    img[:,:,1:] = pred[i]*255
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    #img = img.reshape(1, 224, 224, 3)
    #img = np.expand_dims(img, axis=0)
    img_y = np.zeros((224,224,3))
    img_y[:,:,0] = test_x[0]*255
    img_y[:,:,1:] = test_y[0]*255
    img_y = img_y.astype('uint8')
    img_y = cv2.cvtColor(img_y, cv2.COLOR_LAB2RGB)
    final.append(img)
    predfinal.append(img_y)
#img_y = img_y.reshape(1, 224, 224, 3)
#img_y = np.expand_dims(img_y, axis=0)


# In[29]:


# Calculate SSIM between the two images
ssimtotal = 0
for i in range(600):
    ssim_val1 = ssim(predfinal[i,:,:,0], final[i,:,:,0])
    ssim_val2 = ssim(predfinal[i,:,:,1], final[i,:,:,0])
    ssim_val3 = ssim(predfinal[i,:,:,2], final[i,:,:,0])
    ssimtotal = ssimtotal + (ssim_val1 + ssim_val2 + ssim_val3)/3


# Print the SSIM value
print('SSIM:', ssimtotal/600)


# In[ ]:





# In[30]:


plt.figure(figsize=(30,30))
for i in range(1,16,2):
    plt.subplot(4,4,i)
    img = np.zeros((224,224,3))
    img[:,:,0] = test_x[i]*255
    plt.title('B&W')
    plt.imshow(lab2rgb(img))
    
    plt.subplot(4,4,i+1)
    img[:,:,1:] = pred[i]*255
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    plt.title('Colored')
    plt.imshow(img)


# In[ ]:


plt.figure(figsize=(30,30))
for i in range(1,16,2):
    plt.subplot(4,4,i)
    img = np.zeros((224,224,3))
    img[:,:,0] = test_x[i]*255
    plt.title('B&W')
    plt.imshow(lab2rgb(img))
    
    plt.subplot(4,4,i+1)
    img[:,:,1:] = test_y[i]*255
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    plt.title('Colored')
    plt.imshow(img)


# In[ ]:


plt.imshow(pred[2])


# In[ ]:


plt.imshow(pred[20])


# In[ ]:


model1.save_weights("weightsvaelab.h5")


# In[ ]:


model1.load_weights('weights')


# In[ ]:


model1.score(test_x, test_y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




