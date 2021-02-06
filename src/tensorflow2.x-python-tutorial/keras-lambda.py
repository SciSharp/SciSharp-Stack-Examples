import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

c = tf.constant(['a', 'b'])

e = tf.math.erf([1.0, -0.5, 3.4, -2.1, 0.0, -6.5])
input = tf.keras.Input(shape=(28, 28, 1), name="img")
x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(input)

inputs = Input(24)
x = Dense(128, activation = "relu")(inputs)
value = Dense(24)(x)
adv = Dense(1)(x)
# meam = Lambda(lambda x: K.mean(x, axis = 1, keepdims = True))(adv)
meam = adv - tf.reduce_mean(adv, axis = 1, keepdims = True)
adv = Subtract()([adv, meam])
outputs = Add()([value, adv])
model = Model(inputs, outputs)
model.compile(loss = "mse", optimizer = RMSprop(1e-3))
model.summary()
debug = 1