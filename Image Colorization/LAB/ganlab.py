#!/usr/bin/env python
# coding: utf-8

# # CGAN Lab

# In[ ]:


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


# In[ ]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
import gc


# In[ ]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.filters import threshold_otsu
from glob import glob
from scipy import misc
from matplotlib.patches import Circle,Ellipse
from matplotlib.patches import Rectangle


# In[ ]:


import os
from PIL import Image
import scipy.misc
import imageio
from skimage.transform import rescale, resize
from skimage.color import lab2rgb


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
import gzip


# In[ ]:


tf.random.set_seed(42)


# # Data Augmentation

# In[ ]:


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


# In[ ]:


ab_path = "/kaggle/input/image-colorization/ab/ab/ab1.npy"
l_path = "/kaggle/input/image-colorization/l/gray_scale.npy"


# In[ ]:


ab_df = np.load(ab_path)[0:3000]
L_df = np.load(l_path)[0:3000]
dataset = (L_df,ab_df )
gc.collect()


# In[ ]:


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


# In[ ]:


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


# In[ ]:


x = x = (L_df/255).astype('float32') 
y = (ab_df/255).astype('float32') 


# In[ ]:


#plt.imshow(y[0])


# In[ ]:


import tensorflow as tf
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split( x , y , test_size=0.1, random_state=42 )


# In[ ]:


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mean_mu, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean_mu), mean=0., stddev=1.) 
        return mean_mu + tf.math.exp(log_var/2)*epsilon 


# In[ ]:


lrelu = tf.nn.selu


# In[ ]:


def get_generator_model():

    inputs = tf.keras.layers.Input( shape=( img_size , img_size , 1 ) )

    conv1 = tf.keras.layers.Conv2D( 16 , kernel_size=( 5 , 5 ) , strides=1 )( inputs )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1)( conv1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1)( conv1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )

    conv2 = tf.keras.layers.Conv2D( 32 , kernel_size=( 5 , 5 ) , strides=1)( conv1 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )

    conv3 = tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1 )( conv2 )
    conv3 = tf.keras.layers.LeakyReLU()( conv3 )
    conv3 = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 )( conv3 )
    conv3 = tf.keras.layers.LeakyReLU()( conv3 )
    conv3 = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 )( conv3 )
    conv3 = tf.keras.layers.LeakyReLU()( conv3 )

    bottleneck = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='tanh' , padding='same' )( conv3 )

    concat_1 = tf.keras.layers.Concatenate()( [ bottleneck , conv3 ] )
    conv_up_3 = tf.keras.layers.Conv2DTranspose( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( concat_1 )
    conv_up_3 = tf.keras.layers.Conv2DTranspose( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( conv_up_3 )
    conv_up_3 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu' )( conv_up_3 )

    concat_2 = tf.keras.layers.Concatenate()( [ conv_up_3 , conv2 ] )
    conv_up_2 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( concat_2 )
    conv_up_2 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' )( conv_up_2 )
    conv_up_2 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu' )( conv_up_2 )

    concat_3 = tf.keras.layers.Concatenate()( [ conv_up_2 , conv1 ] )
    conv_up_1 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu')( concat_3 )
    conv_up_1 = tf.keras.layers.Conv2DTranspose( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu')( conv_up_1 )
    conv_up_1 = tf.keras.layers.Conv2DTranspose( 2 , kernel_size=( 5 , 5 ) , strides=1 , activation='relu')( conv_up_1 )

    model = tf.keras.models.Model( inputs , conv_up_1 )
    return model


# In[ ]:


generator = get_generator_model()
generator.summary()


# In[ ]:


def get_discriminator_model():
    
    input1 = tf.keras.layers.Input( shape=( img_size , img_size , 2 ) )
    input2 = tf.keras.layers.Input( shape=( img_size , img_size , 2 ) )

    
    conv1 = tf.keras.layers.Conv2D( 32 , kernel_size=( 7 , 7 ) , strides=1 )( input1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.MaxPooling2D()( conv1 )
    conv2 = tf.keras.layers.Conv2D( 32 , kernel_size=( 7 , 7 ) , strides=1 )( input2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.MaxPooling2D()( conv2 )
    
    conv1 = tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1 )( conv1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.MaxPooling2D()( conv1 )
    conv2 = tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1 )( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.MaxPooling2D()( conv2 )
    
    conv1 = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 )( conv1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.MaxPooling2D()( conv1 )
    conv2 = tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.MaxPooling2D()( conv2 )
    
    conv1 = tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 )( conv1 )
    conv1 = tf.keras.layers.LeakyReLU()( conv1 )
    conv1 = tf.keras.layers.MaxPooling2D()( conv1 )
    conv2 = tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1 )( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.MaxPooling2D()( conv2 )
    
    concat1 = tf.keras.layers.Concatenate()( [ conv1 , conv2 ] )
    
    flatten = tf.keras.layers.Flatten()( concat1 )
    
    dense1 = tf.keras.layers.Dense( 512, activation='relu'  )( flatten )
    dense1 = tf.keras.layers.Dense( 128 , activation='relu' )( dense1 )
    dense1 = tf.keras.layers.Dense( 16 , activation='relu' )( dense1 )
    dense1 = tf.keras.layers.Dense( 1 , activation='sigmoid' )( dense1 )
    
    model = tf.keras.models.Model( [input1, input2] , dense1 )
    
    return model
    


# In[ ]:


"""def get_discriminator_model():
    layers = [
        tf.keras.layers.Conv2D( 32 , kernel_size=( 7 , 7 ) , strides=1 , activation='relu' , input_shape=( 224 , 224 , 2 ) ),
        tf.keras.layers.Conv2D( 32 , kernel_size=( 7, 7 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1, activation='relu'  ),
        tf.keras.layers.Conv2D( 64 , kernel_size=( 5 , 5 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=1, activation='relu'  ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense( 512, activation='relu'  )  ,
        tf.keras.layers.Dense( 128 , activation='relu' ) ,
        tf.keras.layers.Dense( 16 , activation='relu' ) ,
        tf.keras.layers.Dense( 1 , activation='sigmoid' ) 
    ]
    model = tf.keras.models.Sequential( layers )
    return model"""


# In[ ]:


discriminator = get_discriminator_model()


# In[ ]:


lr = 0.001


# In[ ]:


cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()


# In[ ]:


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) , real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) , fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    gen_loss = cross_entropy(tf.ones_like(fake_output) , fake_output)
    return gen_loss


# In[ ]:


def r_accuracy(img_original, img_reconstructed):
    mse = tf.reduce_mean((img_original - img_reconstructed) ** 2)
    pixel_max = 1.0
    psnr = 20 * tf.math.log(pixel_max / tf.math.sqrt(mse))/tf.math.log(10.0)
    return psnr


# In[ ]:


class GAN(keras.Model):
    def __init__(self, generator, discriminator, gen_op=keras.optimizers.Adam(learning_rate = lr), disc_op=keras.optimizers.Adam(learning_rate = lr/10), **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self.r_accuracy_tracker = keras.metrics.Mean(name="r_accuracy")
        self.r_accuracy = r_accuracy
        self.disc_loss = discriminator_loss
        self.gen_loss = generator_loss
        
        self.gen_optimizer = gen_op
        self.disc_optimizer = disc_op
        
    def call(self,x):
        reconstruction = self.generator(x)
        return reconstruction

    
    @property
    def metrics(self):
        return [
            self.gen_loss_tracker,
            self.disc_loss_tracker,
            self.r_accuracy_tracker,
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as disc_tape:
            reconstruction = self.generator(x)
            real_output = self.discriminator([y, y])
            fake_output = self.discriminator([y, reconstruction])
            disc_loss = self.disc_loss(real_output, fake_output)
            

        grad_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        
        self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_weights))
        
        with tf.GradientTape() as gen_tape:
            reconstruction = self.generator(x)
            fake_output = self.discriminator([y, reconstruction])
            gen_loss = self.gen_loss(fake_output) 
            r_accuracy = self.r_accuracy(y, reconstruction)
            
        
        grad_gen = gen_tape.gradient(gen_loss, self.generator.trainable_weights)

        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_weights))
        
        self.gen_loss_tracker.update_state(gen_loss)
        self.disc_loss_tracker.update_state(disc_loss)
        self.r_accuracy_tracker.update_state(r_accuracy)
        return {
            "gen_loss": self.gen_loss_tracker.result(),
            "disc_loss": self.disc_loss_tracker.result(),
            "r_accuracy": self.r_accuracy_tracker.result(),
        }


# In[ ]:


model1 = GAN(generator,discriminator)
model1.compile(optimizer=keras.optimizers.Adam(learning_rate = lr))


# In[ ]:


history = model1.fit(train_x, train_y, epochs=100, batch_size=16, verbose=1)


# In[ ]:


"""del x
del y
del L_df
del ab_df"""


# In[ ]:


model1.save_weights('CGAN100.h5')


# In[ ]:


gc.collect()


# In[ ]:


pred = model1.predict(test_x)


# In[ ]:


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


import numpy as np
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim


# In[ ]:


# Convert the images to numpy arrays
# img1 = tf.keras.preprocessing.image.img_to_array(pred[0])
# img2 = tf.keras.preprocessing.image.img_to_array(test_y[0])
final = []
predfinal = []
for i in range(300):
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


# In[ ]:


final = np.array(final)
predfinal = np.array(predfinal)
gc.collect()


# In[ ]:


# Calculate SSIM between the two images
ssimtotal = 0
for i in range(300):
    ssim_val1 = ssim(predfinal[i,:,:,0], final[i,:,:,0])
    ssim_val2 = ssim(predfinal[i,:,:,1], final[i,:,:,1])
    ssim_val3 = ssim(predfinal[i,:,:,2], final[i,:,:,2])
    ssimtotal = ssimtotal + (ssim_val1 + ssim_val2 + ssim_val3)/3


# Print the SSIM value
print('SSIM:', ssimtotal/600)


# In[ ]:


gc.collect()


# In[ ]:


def compare_mse(img_original, img_reconstructed):
    mse = np.mean((img_original - img_reconstructed) ** 2)
    return mse

mse = 0
for i in range(300):
    mse_i = 0
    mse_i = mse_i + compare_mse(final[i,:,:,0], predfinal[i,:,:,0])
    mse_i = mse_i + compare_mse(final[i,:,:,1], predfinal[i,:,:,1])
    mse_i = mse_i + compare_mse(final[i,:,:,2], predfinal[i,:,:,2])
    mse = mse + mse_i/3

mse = mse/300 
print("MSE: ", mse)


# In[ ]:


from skimage.metrics import peak_signal_noise_ratio

psnr = 0
for i in range(300):
    psnr_i = 0
    psnr_i = psnr_i + peak_signal_noise_ratio(final[i,:,:,0], predfinal[i,:,:,0], data_range = final[i,:,:,0].max() - final[i,:,:,0].min())
    psnr_i = psnr_i + peak_signal_noise_ratio(final[i,:,:,1], predfinal[i,:,:,1], data_range = final[i,:,:,0].max() - final[i,:,:,0].min())
    psnr_i = psnr_i + peak_signal_noise_ratio(final[i,:,:,2], predfinal[i,:,:,2], data_range = final[i,:,:,0].max() - final[i,:,:,0].min())
    psnr = psnr + psnr_i/3

psnr = psnr/300

print("PSNR: ", psnr)

