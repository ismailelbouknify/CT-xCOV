import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import numpy as np

class UNet:
    def __init__(self, input_shape=(256, 256, 1)):
        self.input_shape = input_shape
        self.model = self.build_model()

    def conv_block(self, filters, kernel_size=3, activation='relu', padding='same'):
        return tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer="he_normal")

    def build_model(self):
        inputs = Input(self.input_shape)
		new_dim=256
		
		c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (inputs)
		c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c1)
		c1 = BatchNormalization()(c1)
		p1 = MaxPooling2D((2, 2)) (c1)
		p1 = Dropout(0.25)(p1)

		c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p1)
		c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c2)
		c2 = BatchNormalization()(c2)
		p2 = MaxPooling2D((2, 2)) (c2)
		p2 = Dropout(0.25)(p2)

		c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p2)
		c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c3)
		c3 = BatchNormalization()(c3)
		p3 = MaxPooling2D((2, 2)) (c3)
		p3 = Dropout(0.25)(p3)

		c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p3)
		c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c4)
		c4 = BatchNormalization()(c4)
		p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
		p4 = Dropout(0.25)(p4)

		c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p4)
		c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c5)

		u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
		u6 = concatenate([u6, c4])
		u6 = BatchNormalization()(u6)
		c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u6)
		c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c6)


		u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
		u7 = concatenate([u7, c3])
		u7 = BatchNormalization()(u7)
		c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u7)
		c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c7)


		u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
		u8 = concatenate([u8, c2])
		u8 = BatchNormalization()(u8)
		c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u8)
		c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c8)


		u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
		u9 = concatenate([u9, c1], axis=3)
		u9 = BatchNormalization()(u9)
		c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u9)
		c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c9)

		outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        return Model(inputs=[inputs], outputs=[outputs])

    def compile_model(self):
        self.model.compile(optimizer=Adam(lr=0.0005), loss=self.bce_dice_loss, metrics=[self.dice_coeff])

    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss

    def bce_dice_loss(self, y_true, y_pred):
        loss = 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * self.dice_loss(y_true, y_pred)
        return loss

    def train(self, x_train, y_train, x_test, y_test, batch_size=32, epochs=100):
        filepath_dice_coeff = "unet_covid_weights_dice_coeff.hdf5"
        filepath_loss = "unet_covid_weights_val_loss.hdf5"
        checkpoint_dice = ModelCheckpoint(filepath_dice_coeff, monitor='val_dice_coeff', verbose=1, save_best_only=True, mode='max')
        checkpoint_loss = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=(x_test, y_test),
                       callbacks=[checkpoint_dice, checkpoint_loss])


# Create UNet instance
unet_model = UNet()

# Compile the model
unet_model.compile_model()

# Train the model
unet_model.train(x_train, y_train, x_test, y_test)
