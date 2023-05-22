
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import r2_score, accuracy_score, f1_score
import numpy as np

NAME = 'best_helsinki'

from train import load_data

TAU = 2*np.pi
def smallestSignedAngleBetween(x, y):
    a = (x - y) % TAU
    b = (y - x) % TAU
    return np.where(a < b, -a, b)

def circular_r2_score(true,pred):
    res = sum(smallestSignedAngleBetween(true,pred)**2)
    tot = sum(smallestSignedAngleBetween(true,true.mean())**2)
    return 1 - (res/tot)

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, scale='linear'):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    if scale == 'log':
        radius = np.log(radius)
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

PATH = '/projects/active/MMON/Array_detection/ML_methods/csm_pattern_classification/tf/data/'

X1,y1_reg,y1_cl = np.load(PATH + 'X_helsinki_arces.npy'), np.load(PATH + 'y_reg_helsinki_arces.npy'), np.load(PATH + 'y_cl_helsinki_arces.npy')
X1,y1,sw1 = load_data(X1,y1_reg,y1_cl)

X2,y2_reg,y2_cl = np.load(PATH + 'X_norsar_arces.npy'), np.load(PATH + 'y_reg_norsar_arces.npy'), np.load(PATH + 'y_cl_norsar_arces.npy')
X2,y2,sw2 = load_data(X2,y2_reg,y2_cl)

X = np.concatenate([X1,X2],axis=0)
y = tuple(np.concatenate([a,b],axis=0) for a,b in zip(y1,y2))
sw = tuple(np.concatenate([a,b],axis=0) for a,b in zip(sw1,sw2))

idx = np.where(sw[0]!=0)[0]

FOLDS = 5

MODEL_NAME = 'arces_combined_not_hp_tuned'

for i in range(FOLDS):
    model = tf.keras.models.load_model(f'/staff/erik/Documents/projects/CSM_pattern/tf/output/models/{MODEL_NAME}_model_fold_{i}.h5', compile=False)
    p1, p2 = model.predict(X)
    if i == 0:
        p_reg = p1
        p_cl = p2
    else:
        p_reg += p1
        p_cl += p2


p_reg /= FOLDS
p_cl /= FOLDS
p_cl = np.argmax(p_cl,axis=-1)

print(np.unique(p_cl,return_counts=True))
print(np.unique(y[1].argmax(axis=-1),return_counts=True))

print('Accuracy', accuracy_score(y[1].argmax(axis=-1), p_cl))
print('F1', f1_score(y[1].argmax(axis=-1), p_cl, average=None))

baz = (np.arctan2(y[0].transpose()[0], y[0].transpose()[1]) * (180/np.pi))[idx]
baz_pred = (np.arctan2(p_reg.transpose()[0], p_reg.transpose()[1]) * (180/np.pi))[idx]
baz[np.argwhere(baz<0.0)] += 360
baz_pred[np.argwhere(baz_pred<0.0)] += 360

plt.plot(baz,baz_pred,'o',alpha=0.1)
plt.xlabel('True baz')
plt.ylabel('Predicted baz')
plt.show()

# Visualise by area of bins
angles1 = smallestSignedAngleBetween(baz*np.pi/180,baz_pred*np.pi/180)
plt.hist(angles1, bins=31)
plt.yscale('log')
plt.show()

