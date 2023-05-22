
from nais.Models import AlexNet1D, WaveNet, PhaseNet
from nais.Layers import Resampling1D
from nais.Initializers import UnitCircleInitializer
from nais.metrics import circular_r2_score

import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.utils.class_weight import compute_sample_weight

array = 'arces'
catalog = 'norsar'
dir = 'tf/data/'

X = np.load(f'{dir}X_{catalog}_{array}.npy')
y_cl = np.load(f'{dir}ycl_{catalog}_{array}.npy')
y_reg = np.load(f'{dir}yreg_{catalog}_{array}.npy')
t = np.load(f'{dir}t_{catalog}_{array}.npy', allow_pickle=True)

print(X.shape, y_cl.shape, y_reg.shape)

print('Class distribution:',np.unique(y_cl,return_counts=True))

sample_weights = [compute_sample_weight('balanced',y=y_cl), np.where(y_cl=='N',0,1)]

y_cl = LabelEncoder().fit_transform(y_cl)
X = TimeSeriesScalerMeanVariance().fit_transform(X)

def create_model():

    inp = tf.keras.layers.Input(X.shape[1:])

    x = Resampling1D(length=2048)(inp)
    x = WaveNet(num_outputs=None, pooling='avg', filters=128)(x)

    reg_output = x
    cl_output = x
    for _ in range(2):
        reg_output = tf.keras.layers.Dense(128)(reg_output)
        reg_output = tf.keras.layers.BatchNormalization()(reg_output)
        reg_output = tf.keras.layers.Activation('relu')(reg_output)
        reg_output = tf.keras.layers.Dropout(0.2)(reg_output)
        cl_output = tf.keras.layers.Dense(128)(cl_output)
        cl_output = tf.keras.layers.BatchNormalization()(cl_output)
        cl_output = tf.keras.layers.Activation('relu')(cl_output)
        cl_output = tf.keras.layers.Dropout(0.2)(cl_output)

    cl_output = tf.keras.layers.Dense(len(np.unique(y_cl)), activation='softmax', name='cl_output')(cl_output)
    reg_output = tf.keras.layers.Dense(y_reg.shape[1], activation='linear',
                                       kernel_initializer='zeros',
                                       bias_initializer=UnitCircleInitializer())(reg_output)
    reg_output = tf.keras.layers.Lambda(lambda a: tf.linalg.normalize(a, axis=-1)[0], name='reg_output')(reg_output)

    model = tf.keras.Model(inp,[reg_output,cl_output])
    return model

folds = 4
oof_reg = np.zeros_like(y_reg)
oof_cl = np.zeros_like(y_cl)
for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(folds).split(X,y_cl)):

    model = create_model()
    if fold == 0: model.summary()

    sw = list(map(lambda a: a[train_idx], sample_weights))
    sw_val = list(map(lambda a: a[test_idx], sample_weights))

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=['mse',tf.keras.losses.SparseCategoricalCrossentropy()],
                  metrics=[['mse'],['accuracy']])
    model.fit(X[train_idx],(y_reg[train_idx],y_cl[train_idx]),
              validation_data=(X[test_idx],(y_reg[test_idx],y_cl[test_idx]),sw_val),
              sample_weight=sw,
              callbacks=[tf.keras.callbacks.EarlyStopping('val_loss',patience=5)],
              epochs=100
    )

    p_reg, p_cl = model.predict(X[test_idx])
    oof_reg[test_idx] += p_reg
    oof_cl[test_idx] += np.argmax(p_cl,axis=1)

true = np.angle(y_reg.view(complex))
pred = np.angle(oof_reg.view(complex))

idx = np.where(sample_weights[1]!=0)[0]
print('Balanced accuracy', balanced_accuracy_score(y_cl,oof_cl))
print('Circular R2 Score', circular_r2_score(true[idx],pred[idx]))


