# Copyright 2023 Andreas Koehler, Erik Myklebust, MIT license

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

# code for training ArrayNet

REPEATS = 1
FOLDS = 5
BATCH_SIZE = 128

from sklearn.metrics.pairwise import paired_cosine_distances

import numpy as np

def _smallestSignedAngleBetween(x, y):
    """
    Helper function.
    Returns angle between two angles x and y in radians.
    """
    tau = 2 * np.pi
    a = (x - y) % tau
    b = (y - x) % tau
    return np.where(a < b, -a, b)

def circular_r2_score(true,pred):
    """
    Calculates R2 score between angles (in rad).
    Angles are shifted into (0,2pi) where angle differences are calculated.
    Thereafter, same calulation as for normal R2 score.
    params:
        true : np.array
        pred : np.array

    return:
        float : r2 score between the two sets of angles.
    """
    res = sum(_smallestSignedAngleBetween(true,pred)**2)
    tot = sum(_smallestSignedAngleBetween(true,true.mean())**2)
    return 1 - (res/tot)

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
"""
def distribution_sample_weights(hist, bins, angle):
    idx = find_nearest_idx(bins, angle)
    idx %= len(hist)
    return np.clip(1/hist[idx], 1e-2, 1e9)
"""

from scipy.interpolate import interp1d

def distribution_sample_weights(f, angle):
    return 1/f(angle)

def load_data(X,y,y_class, weight_class=False, weight_angle=False):
    where_noise = np.where(y_class == 'N')[0]

    y[np.isnan(y)] = 0
    X[np.isnan(X)] = 0

    le = OneHotEncoder()
    if weight_class:
        class_weights = compute_sample_weight('balanced',y_class)
    else:
        class_weights = np.ones(y_class.shape[0])

    y_class = np.asarray(le.fit_transform(y_class.reshape((-1,1))).todense())

    angles = np.angle(y.view(complex))
    if weight_angle:
        num_bins = 36
        hist, bins = np.histogram(angles, bins=num_bins, density=True)
        x = np.linspace(-np.pi, np.pi, num=num_bins, endpoint=True)
        f = interp1d(x,hist,kind='cubic')
        sample_weights = 1/f(angles)
        sample_weights = (sample_weights - sample_weights.min()) / (sample_weights.max() - sample_weights.min())
        sample_weights = np.clip(sample_weights, 0.1, 1)
    else:
        sample_weights = np.ones(y.shape[0])

    sample_weights[where_noise] = 0
    sample_weights[(y == 0).all(axis=-1)] = 0

    return X, (y, y_class), (sample_weights, class_weights)

def load_data_sub(X, y, y_class, weight_class=False, weight_angle=False):

    where_p = np.where(['P' in s for s in y_class])[0]
    where_s = np.where(['S' in s for s in y_class])[0]
    where_only_p = np.where(['P' == s for s in y_class])[0]
    where_only_s = np.where(['S' == s for s in y_class])[0]

    p_class = y_class.copy()
    idx = np.union1d(np.setdiff1d(np.arange(len(p_class)), where_p), where_only_p)
    if weight_class:
        p_weights = compute_sample_weight('balanced', p_class)
    else:
        p_weights = np.ones(p_class.shape[0])
    p_class[idx] = 'PN'  # will be weighted by zero
    p_weights[idx] = 0

    s_class = y_class.copy()
    idx = np.union1d(np.setdiff1d(np.arange(len(s_class)), where_s), where_only_s)
    if weight_class:
        s_weights = compute_sample_weight('balanced', s_class)
    else:
        s_weights = np.ones(s_class.shape[0])
    s_class[idx] = 'SN'  # will be weighted by zero
    s_weights[idx] = 0

    y_class = np.array([c[0] for c in y_class]) #removes N/G from PN/PG etc.

    where_noise = np.setdiff1d(np.arange(len(y_class)),
                               np.union1d(np.where(y_class == 'P')[0], np.where(y_class == 'S')[0]))

    y[np.isnan(y)] = 0
    X[np.isnan(X)] = 0

    le = OneHotEncoder()
    lep = OneHotEncoder()
    les = OneHotEncoder()

    if weight_class:
        class_weights = compute_sample_weight('balanced', y_class)
    else:
        class_weights = np.ones(y_class.shape[0])

    y_class = np.asarray(le.fit_transform(y_class.reshape((-1, 1))).todense())
    p_class = np.asarray(lep.fit_transform(p_class.reshape((-1, 1))).todense())
    s_class = np.asarray(les.fit_transform(s_class.reshape((-1, 1))).todense())

    angles = np.angle(y.view(complex))
    if weight_angle:
        num_bins = 36
        hist, bins = np.histogram(angles, bins=num_bins, density=True)
        x = np.linspace(-np.pi, np.pi, num=num_bins, endpoint=True)
        f = interp1d(x,hist,kind='cubic')
        sample_weights = 1/f(angles)
        sample_weights = (sample_weights - sample_weights.min()) / (sample_weights.max() - sample_weights.min())
        sample_weights = np.clip(sample_weights, 0.1, 1)
    else:
        sample_weights = np.ones(y.shape[0])

    sample_weights[where_noise] = 0
    sample_weights[(y == 0).all(axis=-1)] = 0

    return X, (y, y_class, p_class, s_class), (sample_weights, class_weights, p_weights, s_weights)


def angle_diff_tf(true,pred,sample_weight=None):
    true = tf.math.angle(tf.complex(true[:, 0], true[:, 1]))
    pred = tf.math.angle(tf.complex(pred[:, 0], pred[:, 1]))

    diff = tf.math.atan2(tf.math.sin(true-pred), tf.math.cos(true-pred))
    if sample_weight:
        return sample_weight * diff
    return diff

def angle_mae_tf(true,pred,sample_weight=None):
    return tf.reduce_mean(tf.math.abs(angle_diff_tf(true,pred,sample_weight)))

def angle_mse_tf(true,pred,sample_weight=None):
    return tf.reduce_mean(tf.math.square(angle_diff_tf(true,pred,sample_weight)))

from tensorflow.keras import backend as K

def denseblock(x, units=128, k=3, act='relu'):

    features = [x]

    for _ in range(k):
        y = tf.keras.layers.Concatenate()(features) if len(features)>1 else features[0]
        y = tf.keras.layers.Dense(units)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation(act)(y)
        features.append(y)

    return features[-1]

def superblock(x, num_blocks, name='superblock'):
    for i in range(num_blocks):
        x = denseblock(x, 512, 3, 'relu')
        x = tf.keras.layers.Dropout(0.2)(x)
    return x

import tensorflow as tf

def create_model(params):

    inp = tf.keras.layers.Input(params['input_shape'])

    x = inp

    x = superblock(x, num_blocks=2, name='common')
    cl_output = superblock(x, num_blocks=2, name='cl')
    reg_output = superblock(x, num_blocks=2, name='reg')

    cl_output = tf.keras.layers.Dense(params['num_classes'], activation='softmax', name='cl_output')(cl_output)
    reg_output = tf.keras.layers.Dense(params['num_outputs'], activation='linear',
                                       kernel_initializer='zeros',
                                       )(reg_output)
    
    model = tf.keras.Model(inputs=inp, outputs=[reg_output, cl_output])

    alpha = 0.5

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=['mse',
                        tf.keras.losses.CategoricalCrossentropy()],
                  loss_weights=[alpha, 1-alpha],
                  weighted_metrics=([angle_mae_tf], ['accuracy'])
                  )
    return model

def create_model_sub(params):

    inp = tf.keras.layers.Input(params['input_shape'])

    act = 'relu'

    x = inp

    x = superblock(x, num_blocks=2, name='common')
    cl_output = superblock(x, num_blocks=2, name='cl')
    reg_output = superblock(x, num_blocks=2, name='reg')
    p_output = superblock(x, num_blocks=2, name='p')
    s_output = superblock(x, num_blocks=2, name='s')

    cl_output = tf.keras.layers.Dense(params['num_classes'], activation='softmax', name='cl_output')(cl_output)
    p_output = tf.keras.layers.Dense(params['num_p_classes'], activation='softmax', name='p_output')(p_output)
    s_output = tf.keras.layers.Dense(params['num_s_classes'], activation='softmax', name='s_output')(s_output)

    reg_output = tf.keras.layers.Dense(params['num_outputs'], activation='linear',
                                       kernel_initializer='zeros',
                                       )(reg_output)

    model = tf.keras.Model(inputs=inp, outputs=[reg_output, cl_output, p_output, s_output])

    alpha = 0.5

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=['mse',
                        tf.keras.losses.CategoricalCrossentropy(),
                        tf.keras.losses.CategoricalCrossentropy(),
                        tf.keras.losses.CategoricalCrossentropy(),
                        ],
                  loss_weights=[alpha, 1-alpha, 1-alpha, 1-alpha],
                  weighted_metrics=([angle_mae_tf], ['accuracy'], ['accuracy'], ['accuracy'])
                  )
    return model


from sklearn.model_selection import train_test_split

def run_experiment(NAME, X, y_reg, y_cl, weight_class=False, weight_angle=False):

    X, y, sw = load_data(X,y_reg,y_cl,weight_class,weight_angle)

    train_idx, test_idx = train_test_split(np.arange(len(X)),test_size=0.2)
    tmp = X[train_idx], (y[0][train_idx], y[1][train_idx]), (sw[0][train_idx], sw[1][train_idx])
    Xtest, ytest, swtest = X[test_idx], (y[0][test_idx], y[1][test_idx]), (sw[0][test_idx], sw[1][test_idx])
    X, y, sw = tmp

    params = {'input_shape':X.shape[1:],
              'num_outputs':y[0].shape[-1],
              'num_classes':y[1].shape[-1]}

    cmf = lambda : create_model(params)

    oof_reg = np.zeros_like(y[0])
    oof_cl = np.zeros_like(y[1])
    test_reg = np.zeros_like(ytest[0])
    test_cl = np.zeros_like(ytest[1])
    models = []

    for i, (tr_idx, te_idx) in enumerate(KFold(n_splits=FOLDS).split(X,y[0])):
        y_train_reg, y_train_cl = y[0][tr_idx], y[1][tr_idx]
        y_test_reg, y_test_cl = y[0][te_idx], y[1][te_idx]

        X_train = X[tr_idx]
        X_test = X[te_idx]

        model = cmf()

        model.fit(X_train, (y_train_reg, y_train_cl),
                  validation_data=(X_test, (y_test_reg, y_test_cl), (sw[0][te_idx], sw[1][te_idx])),
                  callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=15),
                             tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=7, factor=0.5, verbose=1)],
                  sample_weight=(sw[0][tr_idx], sw[1][tr_idx]),
                  epochs=200,
                  verbose=1,
                  batch_size=BATCH_SIZE)

        p_reg, p_cl = model.predict(X_test)
        oof_reg[te_idx] += p_reg
        oof_cl[te_idx] += p_cl

        p_reg, p_cl = model.predict(Xtest)
        test_reg += p_reg / FOLDS
        test_cl += p_cl / FOLDS

        model.save_weights(f'output/models/{NAME}_model_fold_{i}.h5', save_format='h5')
        tf.keras.backend.clear_session()

    np.save(f'output/{NAME}_oof_reg_predictions.npy', oof_reg)
    np.save(f'output/{NAME}_oof_class_predictions.npy', np.argmax(oof_cl, axis=-1))

    np.save(f'output/{NAME}_test_reg_predictions.npy', test_reg)
    np.save(f'output/{NAME}_test_class_predictions.npy', np.argmax(test_cl, axis=-1))

    true = np.angle(y[0].view(np.complex))
    pred = np.angle(oof_reg.view(np.complex))

    true_test = np.angle(ytest[0].view(np.complex))
    pred_test = np.angle(test_reg.view(np.complex))

    return {
        'OOF R2': circular_r2_score(true[np.where(sw[0] != 0)[0]], pred[np.where(sw[0] != 0)[0]]),
        'OOF (Balanced) Accuracy': accuracy_score(y[1].argmax(axis=-1), np.argmax(oof_cl, axis=-1)),
        'Test R2': circular_r2_score(true_test[np.where(swtest[0] != 0)[0]], pred_test[np.where(swtest[0] != 0)[0]]),
        'Test (Balanced) Accuracy': accuracy_score(ytest[1].argmax(axis=-1), np.argmax(test_cl, axis=-1))
    }


def run_experiment_sub(NAME, X, y_reg, y_cl, weight_class=False, weight_angle=False):

    X, y, sw = load_data_sub(X,y_reg,y_cl,weight_class,weight_angle)

    train_idx, test_idx = train_test_split(np.arange(len(X)),test_size=0.2)
    tmp = X[train_idx], [a[train_idx] for a in y], [s[train_idx] for s in sw]
    Xtest, ytest, swtest = X[test_idx], [a[test_idx] for a in y], [s[test_idx] for s in sw]
    X, y, sw = tmp

    params = {'input_shape':X.shape[1:],
              'num_outputs':y[0].shape[-1],
              'num_classes':y[1].shape[-1],
              'num_p_classes':y[2].shape[-1],
              'num_s_classes':y[3].shape[-1]}

    cmf = lambda : create_model_sub(params)

    oof_reg = np.zeros_like(y[0])
    oof_cl = np.zeros_like(y[1])
    oof_p = np.zeros_like(y[2])
    oof_s = np.zeros_like(y[3])

    test_reg = np.zeros_like(ytest[0])
    test_cl = np.zeros_like(ytest[1])
    test_p = np.zeros_like(ytest[2])
    test_s = np.zeros_like(ytest[3])

    models = []

    for i, (tr_idx, te_idx) in enumerate(KFold(n_splits=FOLDS).split(X,y[0])):
        X_train = X[tr_idx]
        X_test = X[te_idx]

        model = cmf()

        model.fit(X_train, [a[tr_idx] for a in y],
                  validation_data=(X_test, [a[te_idx] for a in y], [a[te_idx] for a in sw]),
                  callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=15),
                             tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=7, factor=0.5, verbose=1)],
                  sample_weight=[a[tr_idx] for a in sw],
                  epochs=200,
                  verbose=1,
                  batch_size=BATCH_SIZE)

        p_reg, p_cl, pp, ss = model.predict(X_test)
        oof_reg[te_idx] += p_reg
        oof_cl[te_idx] += p_cl
        oof_p[te_idx] += pp
        oof_s[te_idx] += ss

        p_reg, p_cl, pp, ss = model.predict(Xtest)
        test_reg += p_reg / FOLDS
        test_cl += p_cl / FOLDS
        test_p += pp / FOLDS
        test_s += ss / FOLDS

        model.save_weights(f'output/models/{NAME}_model_fold_{i}.h5', save_format='h5')
        tf.keras.backend.clear_session()

    np.save(f'output/{NAME}_oof_reg_predictions.npy', oof_reg)
    np.save(f'output/{NAME}_oof_class_predictions.npy', np.argmax(oof_cl, axis=-1))
    np.save(f'output/{NAME}_oof_p_class_predictions.npy', np.argmax(oof_p, axis=-1))
    np.save(f'output/{NAME}_oof_s_class_predictions.npy', np.argmax(oof_s, axis=-1))

    np.save(f'output/{NAME}_test_reg_predictions.npy', test_reg)
    np.save(f'output/{NAME}_test_class_predictions.npy', np.argmax(test_cl, axis=-1))
    np.save(f'output/{NAME}_test_p_class_predictions.npy', np.argmax(test_p, axis=-1))
    np.save(f'output/{NAME}_test_s_class_predictions.npy', np.argmax(test_s, axis=-1))

    true = np.angle(y[0].view(np.complex))
    pred = np.angle(oof_reg.view(np.complex))

    true_test = np.angle(ytest[0].view(np.complex))
    pred_test = np.angle(test_reg.view(np.complex))

    return {
        'OOF R2': circular_r2_score(true[np.where(sw[0] != 0)[0]], pred[np.where(sw[0] != 0)[0]]),
        'OOF (Balanced) Accuracy': accuracy_score(y[1].argmax(axis=-1), np.argmax(oof_cl, axis=-1)),
        'Test R2': circular_r2_score(true_test[np.where(swtest[0] != 0)[0]], pred_test[np.where(swtest[0] != 0)[0]]),
        'Test (Balanced) Accuracy': accuracy_score(ytest[1].argmax(axis=-1), np.argmax(test_cl, axis=-1)),
        'Test P (Balanced) Accuracy': accuracy_score(ytest[2].argmax(axis=-1)[np.where(swtest[2] != 0)[0]],
                                                     np.argmax(test_p[np.where(swtest[2] != 0)[0]], axis=-1)),
        'Test S (Balanced) Accuracy': accuracy_score(ytest[3].argmax(axis=-1)[np.where(swtest[3] != 0)[0]],
                                                     np.argmax(test_s[np.where(swtest[3] != 0)[0]], axis=-1))
    }
