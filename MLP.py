import time
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from sklearn.utils import shuffle
from tensorflow.keras import optimizers
from utils import *
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # Ignore warning


class MLP:
    def __init__(self, epochs=30, batch_size=4, valid_proportion=0.1, model=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.metrics = ['mae', 'mse']
        self.optimizer = optimizers.SGD(learning_rate=1e-1)
        self.valid_proportion = valid_proportion
        if not model:
            # initial model for training
            self.model = self.init_model()
        else:
            # initial model for testing
            self.model = load_model(model)

    def init_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(93, 1)))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(21, activation='hard_sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        model.summary()
        return model

    def fit(self, x, y, model_path):
        filepath = f'{model_path}.hdf5'
        CheckPoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True,
                                     mode='min')
        on_Plateau = ReduceLROnPlateau(monitor='val_mae', patience=4, factor=0.5, min_delta=1e-3,
                                       verbose=1)
        callbacks_list = [CheckPoint, on_Plateau]
        x, y = shuffle(x, y)
        self.model.fit(
            x=x,
            y=y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            callbacks=callbacks_list,
            validation_split=self.valid_proportion,
            shuffle=True
        )

    def my_predict(self, x):
        y_pred = []
        l = len(x)
        pred_times = []
        for i in range(l):
            print(
                '\r', f'predicting network robustness: {i} / {l}...', end='', flush=True)
            start_time = time.time()
            y_pred.append(self.model.predict(np.array(x[i])))
            end_time = time.time()
            pred_times.append(end_time - start_time)
        print(f'predicting time: \n\t'
              f'all time: {sum(pred_times)}\n\t'
              f'average time: {average(pred_times)}')
        return np.array(y_pred).reshape(l, 21, 1)
