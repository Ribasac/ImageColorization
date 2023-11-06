#!/usr/bin/env python
# coding: utf-8

# # VAE RGB

# In[2]:


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


# In[3]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
import gc


# In[4]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.filters import threshold_otsu
from glob import glob
from scipy import misc
from matplotlib.patches import Circle,Ellipse
from matplotlib.patches import Rectangle


# In[5]:


import os
from PIL import Image
import scipy.misc
import imageio
from skimage.transform import rescale, resize
from skimage.color import lab2rgb


# In[6]:


from matplotlib import pyplot as plt
import numpy as np
import gzip


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


rgb_df = []


# In[11]:


for i in range(0,3000):
    img = np.zeros((224,224,3))
    img[:,:,0] = L_df[i]
    img[:,:,1:] = ab_df[i]
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    rgb_df.append(img)


# In[12]:


rgb_df = np.array(rgb_df)


# In[ ]:


plt.figure(figsize=(30,30))
for i in range(1,16,2):
    plt.subplot(4,4,i)
    img = np.zeros((224,224,3))
    img[:,:,0] = L_df[i]
    plt.title('B&W')
    plt.imshow(lab2rgb(img))
    
    plt.subplot(4,4,i+1)
    img = rgb_df[i]
    plt.title('Colored')
    plt.imshow(img)


# In[ ]:


img_size = 224
batch_size = 64
INPUT_DIM = (img_size,img_size,1)
Z_DIM = 64


# In[ ]:


ab_df.shape


# In[ ]:


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


# In[ ]:


"""x, y = read_images(data) #calling readimage"""


# In[18]:


x = (L_df/255).astype('float32') 
y = (rgb_df/255).astype('float32') 


# In[19]:


del ab_df


# In[ ]:





# In[20]:


#plt.imshow(y[0])


# In[21]:


import tensorflow as tf
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split( x , y , test_size=0.1, random_state=42 )


# In[22]:


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mean_mu, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean_mu), mean=0., stddev=1.) 
        return mean_mu + tf.math.exp(log_var/2)*epsilon 


# In[23]:


lrelu = tf.nn.selu


# In[24]:


def build_vae_encoder(input_dim, output_dim):
    
    # Clear tensorflow session to reset layer index numbers to 0 for LeakyRelu, 
    # BatchNormalization and Dropout.
    # Otherwise, the names of above mentioned layers in the model 
    # would be inconsistent
    global K
    K.clear_session()
    

    # Define model input
    encoder_input = Input(shape = input_dim, name = 'encoder_input')
    x = encoder_input

    # Add convolutional layers
    conv1 = Conv2D( 8 , kernel_size=( 5 , 5 ) , strides=1 )( x )
    conv1 = LeakyReLU()( conv1 )
    conv1 = Conv2D( 16 , kernel_size=( 3 , 3 ) , strides=1)( conv1 )
    conv1 = LeakyReLU()( conv1 )
    conv1 = Conv2D( 16 , kernel_size=( 3 , 3 ) , strides=1)( conv1 )
    conv1 = LeakyReLU()( conv1 )

    conv2 = Conv2D( 16 , kernel_size=( 5 , 5 ) , strides=1)( conv1 )
    conv2 = LeakyReLU()( conv2 )
    conv2 = Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = LeakyReLU()( conv2 )
    conv2 = Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = LeakyReLU()( conv2 )

    conv3 = Conv2D( 32 , kernel_size=( 5 , 5 ) , strides=1 )( conv2 )
    conv3 = LeakyReLU()( conv3 )
    conv3 = Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 )( conv3 )
    conv3 = LeakyReLU()( conv3 )
    conv3 = Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 )( conv3 )
    conv3 = LeakyReLU()( conv3 )

    # Required for reshaping latent vector while building Decoder
    shape_before_flattening = K.int_shape(conv3)[1:] 
    
    x = Flatten()(conv3)
    
    mean_mu = Dense(64, name = 'mu')(x)
    log_var = Dense(64, name = 'log_var')(x)

    # Using a Keras Lambda Layer to include the sampling function as a layer 
    # in the model
    z = Sampling()([mean_mu, log_var])
    cshape1 = tf.shape(conv1)[2]
    cshape2 = tf.shape(conv2)[2]
    cshape3 = tf.shape(conv3)[2]

    return encoder_input, [z, conv3, conv2, conv1], [cshape1, cshape2, cshape3], mean_mu, log_var , shape_before_flattening, Model(inputs = encoder_input, outputs = [mean_mu, log_var, z, conv3, conv2, conv1])


# In[25]:


vae_encoder_input, vae_encoder_output, cshape, mean_mu, log_var, vae_shape_before_flattening, encoder  = build_vae_encoder(input_dim = INPUT_DIM,
                                    output_dim = Z_DIM)

encoder.summary()


# In[26]:


def build_decoder(input_dim, shape_before_flattening):

    # Define model input
    decoder_input = Input(shape = (input_dim,) , name = 'decoder_input')
    conv3 = Input(shape = (200,200,64,) , name = 'conv3')
    conv2 = Input(shape = (208,208,32,) , name = 'conv2')
    conv1 = Input(shape = (216,216,16,) , name = 'conv1')

    # To get an exact mirror image of the encoder
    x = Dense(np.prod(shape_before_flattening))(decoder_input)
    x = Reshape(shape_before_flattening)(x)

    # Add convolutional layers
    concat_1 = Concatenate()( [ x , conv3 ] )
    conv_up_3 = Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 )( concat_1 )
    conv_up_3 = LeakyReLU()( conv_up_3 )
    conv_up_3 = Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 )( conv_up_3 )
    conv_up_3 = LeakyReLU()( conv_up_3 )
    conv_up_3 = Conv2DTranspose( 32 , kernel_size=( 5 , 5 ) , strides=1 )( conv_up_3 )
    conv_up_3 = LeakyReLU()( conv_up_3 )

    concat_2 = Concatenate()( [ conv_up_3 , conv2 ] )
    conv_up_2 = Conv2DTranspose( 32 , kernel_size=( 3 , 3 ) , strides=1 )( concat_2 )
    conv_up_2 = LeakyReLU()( conv_up_2 )
    conv_up_2 = Conv2DTranspose( 16 , kernel_size=( 3 , 3 ) , strides=1 )( conv_up_2 )
    conv_up_2 = LeakyReLU()( conv_up_2 )
    conv_up_2 = Conv2DTranspose( 16 , kernel_size=( 5 , 5 ) , strides=1 )( conv_up_2 )
    conv_up_2 = LeakyReLU()( conv_up_2 )

    concat_3 = Concatenate()( [ conv_up_2 , conv1 ] )
    conv_up_1 = Conv2DTranspose( 16 , kernel_size=( 3 , 3 ) , strides=1 )( concat_3 )
    conv_up_1 = LeakyReLU()( conv_up_1 )
    conv_up_1 = Conv2DTranspose( 8 , kernel_size=( 3 , 3 ) , strides=1 )( conv_up_1 )
    conv_up_1 = LeakyReLU()( conv_up_1 )
    x = Conv2DTranspose( 3 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu')( conv_up_1 )

    # Define model output
    decoder_output = x

    return [decoder_input, conv3, conv2, conv1], decoder_output, Model(inputs = [decoder_input, conv3, conv2, conv1], outputs = decoder_output)


# In[27]:


vae_decoder_input, vae_decoder_output, decoder = build_decoder(input_dim = Z_DIM,
                                        shape_before_flattening = vae_shape_before_flattening
                                        )
decoder.summary()


# In[28]:


lr = 0.00005


# In[29]:


def r_accuracy(img_original, img_reconstructed):
    mse = tf.reduce_mean((img_original - img_reconstructed) ** 2)
    pixel_max = 1.0
    psnr = 20 * tf.math.log(pixel_max / tf.math.sqrt(mse))/tf.math.log(10.0)
    return psnr


# In[30]:


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.r_accuracy_tracker = keras.metrics.Mean(name="r_accuracy")
        self.r_accuracy = r_accuracy
        
        
    def call(self,x):
        z_mean, z_log_var, z, conv3, conv2, conv1 = self.encoder(x)
        reconstruction = self.decoder([z, conv3, conv2, conv1])
        return reconstruction

    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.r_accuracy_tracker,
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, conv3, conv2, conv1 = self.encoder(x)
            reconstruction = self.decoder([z, conv3, conv2, conv1])
            reconstruction_loss = tf.reduce_mean(tf.math.square(y - reconstruction), axis=[1, 2, 3])
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis = 1)
            kl_loss = tf.reduce_mean(kl_loss)
            #coorelation_loss = corr_loss(z)
            
            total_loss = 10000*reconstruction_loss + kl_loss
            r_accuracy = self.r_accuracy(y, reconstruction)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.r_accuracy_tracker.update_state(r_accuracy)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "r_accuracy": self.r_accuracy_tracker.result(),
        }


# In[31]:


model1 = VAE(encoder,decoder)
model1.compile(optimizer=keras.optimizers.Adam(learning_rate = lr))


# In[31]:


history = model1.fit(train_x, train_y, epochs=100, batch_size=32, verbose=1)


# In[32]:


del x
del y
del L_df


# In[ ]:


gc.collect()


# In[ ]:


pred = model1.predict(test_x)


# In[ ]:


# Convert the images to numpy arrays
# img1 = tf.keras.preprocessing.image.img_to_array(pred[0])
# img2 = tf.keras.preprocessing.image.img_to_array(test_y[0])
final = []
predfinal = []
for i in range(600):
    img = np.zeros((224,224,3))
    img = pred[0]*255
    #img = img.reshape(1, 224, 224, 3)
    #img = np.expand_dims(img, axis=0)

    img_y = np.zeros((224,224,3))
    img_y = test_y[0]*255
    
    final.append(img)
    predfinal.append(img_y)
#img_y = img_y.reshape(1, 224, 224, 3)
#img_y = np.expand_dims(img_y, axis=0)
img.shape


# In[ ]:


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


plt.figure(figsize=(30,30))
for i in range(1,16,2):
    plt.subplot(4,4,i)
    img = np.zeros((224,224,3))
    img[:,:,0] = test_x[i+128]*255
    plt.title('B&W')
    plt.imshow(lab2rgb(img))
    
    plt.subplot(4,4,i+1)
    img = pred[i+128]
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




