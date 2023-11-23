import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report


from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Lambda
from keras.models import Model
import tensorflow as tf
from keras import backend as K

class MyClassifier:
    def __init__(self, input_shape=(256, 256, 1), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.selected_model = None

    def build_cnn_model(self):
        inputs = Input(self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(inputs)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs=inputs, outputs=outputs)
        
    def build_densenet_model(self):
        inp = Input(self.input_shape)
        backbone = tf.keras.applications.DenseNet121(input_tensor=inp, weights=None, include_top=False)

        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
        out = Lambda(lambda x: x[:, :, 0])(x)

        return Model(inputs=inp, outputs=out)

    def choose_model(self, model_type):
        if model_type == 'cnn':
            self.selected_model = self.build_cnn_model()
        elif model_type == 'densenet':
            self.selected_model = self.build_densenet_model()
        else:
            raise ValueError("Invalid model_type. Choose 'cnn' or 'densenet'.")

    def compile_selected_model(self):
        if self.selected_model is None:
            raise ValueError("No model selected. Use choose_model method first.")
        self.selected_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        

    def train(self, x_train, y_train, x_val, y_val, batch_size=32, epochs=100):
        filepath = "cnn_classifier_weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=(x_val, y_val),
                       callbacks=[checkpoint])

    def evaluate(self, x_test, y_test):
        preds = self.model.predict(x_test)
        y_preds = np.argmax(preds, axis=1)
        y_true = np.argmax(y_test, axis=1)
        return classification_report(y_true, y_preds)

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=2)


# Create MyClassifier instance
my_classifier = MyClassifier()

# Choose and build the CNN model
my_classifier.choose_model('cnn')

# Compile the selected model (CNN in this case)
my_classifier.compile_selected_model()


# Train the model
my_classifier.train(x_train, y_train_onehot, x_val, y_val_onehot)

# Evaluate the model
report = my_classifier.evaluate(x_test, y_test_onehot)
print(report)
