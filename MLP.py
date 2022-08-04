import time
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from sklearn.utils import shuffle
from tensorflow.keras import optimizers
from utils import *
import os
import warnings
from parameters import pooling_attr

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # Ignore warning


class MLP:
    def __init__(self, epochs=200, batch_size=8, valid_proportion=0.1, model=None, pooling_sizes=None, num_classes = None, num_node_attr=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        if self.num_classes:
            self.num_classes = num_classes
            self.metrics = ['accuracy']
            self.optimizer = optimizers.Adam(learning_rate=1e-5)
        else:
            self.metrics = ['mae', 'mse']
            self.optimizer = optimizers.Adam()
        self.valid_proportion = valid_proportion
        if not num_node_attr:
            self.input_shape = (batch_size, len(pooling_attr) * sum(pooling_sizes))
        else:
            self.input_shape = (batch_size, (num_node_attr+len(pooling_attr)) * sum(pooling_sizes))
        if not model:
            # initial model for training
            self.model = self.init_model()
        else:
            # initial model for testing
            self.model = load_model(model)

    def init_model(self):
        if self.num_classes:
            model = Sequential()
            model.add(Dense(512, activation='relu', input_shape=self.input_shape))
            model.add(Dense(1024, activation='relu'))
            # model.add(Dropout(0.5))
            model.add(Dense(1024, activation='relu'))
            # model.add(Dropout(0.5))
            model.add(Dense(512, activation='relu'))
            # model.add(Dropout(0.5))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                        optimizer=self.optimizer,
                        metrics=self.metrics)
            model.summary()
        else:
            model = Sequential()
            model.add(Dense(512, activation='relu', input_shape=self.input_shape))
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(21, activation='hard_sigmoid'))
            model.compile(loss='mean_squared_error',
                        optimizer=self.optimizer,
                        metrics=self.metrics)
            model.summary()
        return model

    def fit(self, x, y, model_path):
        filepath = f'{model_path}.hdf5'
        if self.num_classes:
            # CheckPoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
            #                          mode='max')
            earlyStopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001,patience=25, verbose=1)
            callbacks_list = [earlyStopping]
        else:
            CheckPoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True,
                                     mode='min')
            on_Plateau = ReduceLROnPlateau(monitor='val_mae', patience=20, factor=0.5, min_delta=1e-4,
                                        verbose=1,min_lr=1e-4)
            callbacks_list = [CheckPoint, on_Plateau]
        x, y = shuffle(x, y)
        self.model.fit(
            x=x,
            y=y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=2,
            callbacks=callbacks_list,
            validation_split=self.valid_proportion,
            shuffle=True
        )

    def my_predict(self, x):
        y_pred = self.model.predict(x,batch_size=1)
        return y_pred
