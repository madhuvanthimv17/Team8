from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
class CNN:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# CONV => RELU => BN => POOL
		model.add(Conv2D(8, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		#Baseline 5 : Intriduce Max pooling layer
		#model.add(MaxPooling2D(pool_size=(2, 2)))
		#Baseline 5 ends

		#Baseline 6 : Introduce Average pooling layer
		model.add(AveragePooling2D(pool_size=(2, 2)))
		#Baseline 6 ends

		# first set of (CONV => RELU => CONV => RELU) * 2 => POOL
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		#Baseline 5 : Intriduce Max pooling layer
		#model.add(MaxPooling2D(pool_size=(2, 2)))
		#Baseline 5 ends

		#Baseline 6 : Introduce Average pooling layer
		model.add(AveragePooling2D(pool_size=(2, 2)))
		#Baseline 6 ends

		# second set of (CONV => RELU => CONV => RELU) * 2 => POOL
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		#Baseline 5 : Intriduce Max pooling layer
		#model.add(MaxPooling2D(pool_size=(2, 2)))
		#Baseline 5 ends

		#Baseline 6 : Introduce Average pooling layer
		model.add(AveragePooling2D(pool_size=(2, 2)))
		#Baseline 6 ends


		# first set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		#Line 71 to be commented out only for Baseline 5 and 6
		#model.add(Dropout(0.5))
		# second set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		#Line 78 to be commented out only for Baseline 5 and 6
		#model.add(Dropout(0.5))
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		# return the constructed network architecture
		return model
