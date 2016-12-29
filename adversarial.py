import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Dropout, Activation
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D, Flatten
from keras.models import Model
from keras import backend as K
from keras import objectives, regularizers
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from skimage.io import imsave, imread
from scipy.misc import imsave as imsv
from skimage.transform import resize
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
from scipy import fftpack
from PIL import Image
from keras.models import Sequential

from skimage import io, exposure, img_as_uint, img_as_float
import os
import gzip
import sys
from six.moves import cPickle
from skimage.feature import match_template

import gzip
import pickle
import xlrd
import numpy as np
import glob, cv2, os

def some_act_f(x):
	some_act = K.abs(x)/ K.sum(K.abs(x))
	K.function(x,some_act)
	result = some_act(x)
	return result


def supervised_individual_model():

	input_img = Input(shape=(1,32,32))

	c1 = Convolution2D( 32,3,3, activation='relu' ) (input_img)
	c2 = Convolution2D( 32,3,3, activation='relu' ) (c1)
	mp1 = MaxPooling2D( pool_size=(2,2))(c2)
	drp1 = Dropout( 0.25 )(mp1)
	flatten1 = Flatten()(drp1)
	dense_intermediate = Dense( 128, activation='relu' ) (flatten1)
	drp2 = Dropout(0.5)( dense_intermediate )
	dense_final_supervised = Dense(5, activation='softmax' , name ='supervised') (drp2)

	model = Model( [input_img], [dense_final_supervised] )
	model.compile(loss='categorical_crossentropy',
		      optimizer='adadelta',
		      metrics=['accuracy'])


	return (model, input_img, dense_final_supervised )


def supervised_individual_model2():

	input_img = Input(shape=(1,32,32))

	c1 = Convolution2D( 32,1,1, activation='relu' ) (input_img)
	c2 = Convolution2D( 32,9,9, activation='relu' ) (c1)
	mp1 = MaxPooling2D( pool_size=(2,2))(c2)
	c2 = Convolution2D( 32,5,5, activation='relu' ) (mp1)
	c3 = Convolution2D( 32,3,3, activation='relu' ) (c2)
	c4 = Convolution2D( 32,1,1, activation='relu' ) (c3)
	drp1 = Dropout( 0.25 )(c4)
	flatten1 = Flatten()(drp1)
	dense_intermediate = Dense( 128, activation='relu' ) (flatten1)
	drp2 = Dropout(0.5)( dense_intermediate )
	dense_final_supervised = Dense(5, activation='softmax' , name ='supervised') (drp2)

	model = Model( [input_img], [dense_final_supervised] )
	model.compile(loss='categorical_crossentropy',
		      optimizer='adadelta',
		      metrics=['accuracy'])


	return (model, input_img, dense_final_supervised )


def autoencoder_individual_model():


	input_img = Input(shape=(1,32,32))

	c1 = Convolution2D( 32,3,3, activation='relu' ) (input_img)
	c2 = Convolution2D( 32,3,3, activation='relu' ) (c1)
	mp1 = MaxPooling2D( pool_size=(2,2))(c2)
	drp1 = Dropout( 0.25 )(mp1)
	flatten1 = Flatten()(drp1)
	dense_intermediate = Dense( 128, activation='relu' ) (flatten1)
	drp2 = Dropout(0.5)( dense_intermediate )
	dense_final_unsupervised = Dense(32*32, activation='sigmoid' , name ='supervised') (drp2)

	model = Model( [input_img], [dense_final_unsupervised] )
	model.compile(loss='binary_crossentropy',
		      optimizer='adadelta',
		      metrics=['accuracy'])


	return (model, input_img, dense_final_unsupervised )


def autoencoder_individual_model2():


	input_img = Input(shape=(1,32,32))

	c1 = Convolution2D( 32,3,3, activation='relu' ) (input_img)
	c2 = Convolution2D( 32,3,3, activation='relu' ) (c1)
	mp1 = MaxPooling2D( pool_size=(2,2))(c2)
	drp1 = Dropout( 0.25 )(mp1)
	c3 = Convolution2D( 64,3,3, activation='relu' ) (drp1)
	c4 = Convolution2D( 64,3,3, activation='relu') (c3)
	mp2 = MaxPooling2D( pool_size=(2,2))(c4)
	drp2 = Dropout( 0.25 )(mp2)
	flatten1 = Flatten()(drp2)
	dense_intermediate = Dense( 512, activation='relu' ) (flatten1)
	drp3 = Dropout(0.5)( dense_intermediate )
	dense_final_unsupervised = Dense(32*32, activation='sigmoid' , name ='unsupervised') (drp3)	

	model = Model( [input_img], [dense_final_unsupervised] )
	model.compile(loss='binary_crossentropy',
		      optimizer='rmsprop',
		      metrics=['binary_accuracy'])

	print model.summary()

	return (model, input_img, dense_final_unsupervised )



def joint_model():
	input_img = Input(shape=(1,32,32))

	c1 = Convolution2D( 32,3,3, activation='relu' ) (input_img)
	c2 = Convolution2D( 32,3,3, activation='relu' ) (c1)
	mp1 = MaxPooling2D( pool_size=(2,2))(c2)
	drp1 = Dropout( 0.1 )(mp1)
	flatten1 = Flatten()(drp1)
	dense_intermediate = Dense( 128, activation='relu' ) (flatten1)
	drp2 = Dropout(0.25)( dense_intermediate )
	dense_final_supervised = Dense(10, activation='softmax' , name ='supervised') (drp2)
	dense_final_unsupervised = Dense(32*32, activation='sigmoid' , name ='unsupervised') (drp2)	

	model = Model( [input_img], [dense_final_supervised, dense_final_unsupervised] )
	model.compile(optimizer='adadelta',
              loss={'supervised': 'categorical_crossentropy', 'unsupervised': 'binary_crossentropy'},
              loss_weights={'supervised':0.5 , 'unsupervised': 0.5}, metrics=['accuracy'])

	return (model, input_img, dense_final_supervised, dense_final_unsupervised)


def joint_model2():
	input_img = Input(shape=(1,32,32))

	
	c1 = Convolution2D( 32,1,1, activation='relu' ) (input_img)
	c2 = Convolution2D( 32,3,3, activation='relu' ) (c1)
	mp1 = MaxPooling2D( pool_size=(2,2))(c2)
	drp1 = Dropout( 0.25 )(mp1)
	c3 = Convolution2D( 64,3,3, activation='relu' ) (drp1)
	c4 = Convolution2D( 64,3,3, activation='relu' ) (c3)
	mp2 = MaxPooling2D( pool_size=(2,2))(c4)
	drp2 = Dropout( 0.35 )(mp2)
	flatten1 = Flatten()(drp2)
	dense_intermediate = Dense( 512, activation='relu' ) (flatten1)
	drp3 = Dropout(0.5)( dense_intermediate )
	dense_final_unsupervised = Dense(32*32, activation='sigmoid' , name ='unsupervised') (dense_intermediate)	
	dense_final_supervised = Dense(5, activation='softmax' , name ='supervised') (drp3)

	model = Model( [input_img], [dense_final_supervised, dense_final_unsupervised] )
	model.compile(optimizer='adam',
              loss={'supervised': 'categorical_crossentropy', 'unsupervised': 'binary_crossentropy'},
              loss_weights={'supervised': 0.5, 'unsupervised': 0.5}, metrics=['accuracy'])

	return (model, input_img, dense_final_supervised, dense_final_unsupervised)

"""
def load_data( path ):

        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding="bytes")

        f.close()
        return data  #(data, labels)
"""


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def write_image( some_image, filename ):
	some_image = some_image.reshape(1,32,32)	
	some_image = deprocess_image( some_image )

	img_to_save = np.zeros( (32,32,3))
	img_to_save[:,:,:] = some_image

	imsv(filename, img_to_save)

def read_image( filename ):

	image = imread( filename, as_grey=True )
	image = resize(image, (32,32))
	return image


def gradient_function( o, i, class_label):
	g = K.gradients(  o[0][class_label], i  )[0]
	g = normalize( g )
	return K.function([i,  K.learning_phase()],[g])


def load_cifar10_dataset():
	nb_classes = 5
	nb_epoch = 200
	data_augmentation = True

	# input image dimensions
	img_rows, img_cols = 32, 32
	# the CIFAR10 images are RGB
	img_channels = 1

	# the data, shuffled and split between train and test sets
	# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
	fileName = './FullData.pkl.gz'
	f = gzip.open(fileName, 'rb')
	(X_train, y_train), (X_test, y_test) = pickle.load(f)
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')


	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255

	# reshape the data
	X_train = X_train.reshape(len(X_train), 1, 32, 32)
	X_test = X_test.reshape(len(X_test), 1, 32, 32)

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)


	Y_train_unsupervised = X_train.copy().reshape(X_train.shape[0], 32*32)
	Y_test_unsupervised = X_test.copy().reshape(X_test.shape[0], 32*32)


	return (X_train, Y_train, Y_train_unsupervised, y_train) , (X_test, Y_test, Y_test_unsupervised, y_test)

(X_train, Y_train, Y_train_unsupervised, y_train) , (X_test, Y_test, Y_test_unsupervised, y_test) = load_cifar10_dataset()



autoencoder, aut_inp, aut_out = autoencoder_individual_model2()
autoencoder.fit(X_train, Y_train_unsupervised, batch_size=128, nb_epoch=2, verbose=1, validation_data=(X_test, Y_test_unsupervised))
autoencoder.save_weights( 'myutilscifar10-unsupervised-individual2.h5' )
autoencoder.load_weights( 'myutilscifar10-unsupervised-individual2.h5' )


supervised, sup_inp, sup_out = supervised_individual_model2()
supervised.fit(X_train, Y_train, batch_size=128, nb_epoch=2, verbose=1, validation_data=(X_test, Y_test))
supervised.save_weights( 'myutilscifar10-supervised-individual2.h5' )
supervised.load_weights( 'myutilscifar10-supervised-individual2.h5' )



joint, joint_inp, joint_sup_out, joint_unsup_out = joint_model2()
joint.fit(X_train, { 'supervised':Y_train, 'unsupervised':Y_train_unsupervised}, validation_data=([X_test], {'supervised':Y_test, 'unsupervised':Y_test_unsupervised}), nb_epoch=2, verbose=2,)
joint.save_weights( 'myutilscifar10-joint2.h5' )
joint.load_weights( 'myutilscifar10-joint2.h5' )

mistake_sup = 0

#g_5 = gradient_function( sup_out, sup_inp, 1 )
g_5 = gradient_function( sup_out, sup_inp, 1)

def correlation( ix,iy, size ):
	x = K.reshape(x, (size,1) )
	y = K.reshape(y, (size,1) )
	x = x - K.mean( x )
	y = y - K.mean( y )

	p = x*y

	p = p/K.std(x)
	p = p/K.std(y)

	p = K.max(p)

	f_p = K.function( [ix, iy], [p])
	return f_p


	

for i in range(10) : #X_test.shape[0] ):
	# origimg = X_test[i].reshape(1,1,32,32)
	origimg = np.zeros((1,1,32,32))
	#origimg = np.random.random((1,3,32,32))		
	#origimg = bangla_data[0][0][0].reshape(1,1,28,28)
	img = origimg.copy()

	predictions_sup_b = supervised.predict( img )
	predictions_autoenc_b = autoencoder.predict(img)
	predictions_joint_b = joint.predict( img )

	label_joint_b, confidence_joint_b = ( np.argmax( predictions_joint_b[0]), np.max( predictions_joint_b[0]) )


	for somet in range(20):
		img += 0.07 *  np.sign(g_5( [img, 0] )[0]) 

	predictions_sup = supervised.predict( img )
	predictions_autoenc = autoencoder.predict(img)

	predictions_joint = joint.predict( img )

	label_sup = np.argmax( predictions_sup[0] )
	confidence_sup = np.max( predictions_sup[0] )

	label_joint, confidence_joint = ( np.argmax( predictions_joint[0]), np.max( predictions_joint[0]) )

	sup_mistake_label = ''
	if ( label_joint != label_joint_b ):
		mistake_sup += 1
		sup_mistake_label = 'x'

	sup_template = [ 0.1]
	joint_template = [0.1]

	#sup_template = match_template( img.reshape(32,32,3), predictions_autoenc.reshape(32,32,3))
	#joint_template = match_template( img.reshape(32,32,3), predictions_joint[1].reshape(32,32,3))

	sup_sum_of_diff = np.sum(np.absolute(img.reshape(1,32*32) - predictions_autoenc.reshape(1,32*32)))
	joint_sum_of_diff = np.sum(np.absolute(img.reshape(1,32*32) - predictions_joint[1].reshape(1,32*32)))
	

	folderName = './images_to_write/'
	if not os.path.exists(folderName):
		os.makedirs(folderName)

	write_image( origimg, 'images_to_write/{}_orig.png'.format(i))
	write_image( predictions_autoenc_b, 'images_to_write/{}_before_pred.png'.format(i))
	write_image( predictions_joint_b[1], 'images_to_write/{}_before_pred_joint.png'.format(i))
	write_image( img, 'images_to_write/{}_chng.png'.format(i))		
	write_image( predictions_autoenc, 'images_to_write/{}_pred.png'.format(i))
	write_image( predictions_joint[1], 'images_to_write/{}_pred_joint.png'.format(i))
#	print ( "{}  - j:{}  {} {},  s:{}  {} {}".format(i,label_j, confidence_j, j_mistake_label, label_s, confidence_s, s_mistake_label ) )
	if (sup_mistake_label == 'x') :
		print( "XXX - {} - {}: actual:{}, predicted_sup:{}, predicted_joint:{}, sup_conf:{}, joint_conf:{}, {},{}".format(i, sup_mistake_label, y_test[i], label_sup, label_joint, confidence_sup, confidence_joint, sup_template[0], joint_template[0]))

	else:
		print( "{} - {}: actual:{}, predicted_sup:{}, predicted_joint:{}, sup_conf:{}, joint_conf:{}, {},{}".format(i, sup_mistake_label, y_test[i], label_sup, label_joint, confidence_sup, confidence_joint, sup_template[0], joint_template[0]))

#	write_image( img, 'images_to_write/{}_act.png'.format(i))
	

print ("mistakes_sup:{}".format( mistake_sup ))



'''

input_img = Input(shape=(3,32,32))

c1 = Convolution2D( 32,1,1, activation='relu' ) (input_img)
c2 = Convolution2D( 32,3,3, activation='relu' ) (c1)
mp1 = MaxPooling2D( pool_size=(2,2))(c2)
drp1 = Dropout( 0.25 )(mp1)
c3 = Convolution2D( 32,5,5, activation='relu' ) (drp1)
drp2 = Dropout( 0.25 )(c3)
flatten1 = Flatten()(drp2)
dense_intermediate = Dense( 256, activation='relu' ) (flatten1)
dense_intermediate = Dense( 128, activation='relu' ) (dense_intermediate)
drp2 = Dropout(0.5)( dense_intermediate )
dense_final_supervised = Dense(10, activation='softmax' , name ='supervised') (drp2)
dense_final_unsupervised = Dense(3072, activation='sigmoid', name='unsupervised' ) (drp2)

model = Model( [input_img], [dense_final_supervised, dense_final_unsupervised] )

model.compile(optimizer='adadelta',
              loss={'supervised': 'categorical_crossentropy', 'unsupervised': 'binary_crossentropy'},
              loss_weights={'supervised': 0.3, 'unsupervised': 0.7}, metrics=['accuracy'])


img_rows=28
img_cols = 28
nb_classes = 10








#dirname="./isolated_char.pkl.gz"

#img_rows, img_cols = 28, 28
#(X_train, y_train), (X_test, y_test) ,(X_val, y_val)= load_data( dirname )








#seq_model.fit(X_train, Y_train, batch_size=128, nb_epoch=10, verbose=1, validation_data=(X_test, Y_test))
#seq_model.save_weights( 'seq_model.hf' )


#model.fit(X_train, { 'supervised':Y_train, 'unsupervised':Y_train_unsupervised}, validation_data=([X_test], {'supervised':Y_test, 'unsupervised':Y_test_unsupervised}), nb_epoch=15, verbose=2,)

#model.save_weights( 'sup-unsup.h5' )

seq_model.load_weights( 'seq_model.hf' )
model.load_weights('sup-unsup.h5' )




loss = dense_final_supervised[0][4]

grads = K.gradients( dense_final_supervised[0][5], input_img )[0]
grads = normalize( grads )
f_g = K.function( [input_img,K.learning_phase()],[grads]) 

grads= K.gradients( ll.output[0][5], fl.input)[0]
grads = normalize( grads )
f_g = K.function( [fl.input,K.learning_phase()],[grads]) 


'''
