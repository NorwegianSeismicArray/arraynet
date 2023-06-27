from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

from train import create_model
from train import load_data

TAU = 2*np.pi
def smallestSignedAngleBetween(x, y):
    a = (x - y) % TAU
    b = (y - x) % TAU
    return np.where(a < b, -a, b)

if __name__ == '__main__':

    plt.rcParams["figure.figsize"] = (10,7)

    array = 'arces'

    dataset = 'data'
    fbands = '_4Fre'

    PATH = 'tf/' + dataset +'/'
    MODEL_NAME = 'merged_'+array+fbands+'_dataset_'+dataset+'_cw_False_sw_False'

    catalog = 'merged'

    X,y_reg,y_cl = np.load(PATH + 'X_'+catalog+'_'+array+fbands+'.npy'), \
                   np.load(PATH + 'y_reg_'+catalog+'_'+array+fbands+'.npy'), \
                   np.load(PATH + 'y_cl_'+catalog+'_'+array+fbands+'.npy')
    # Use all data for predictions (reduced data set) but predict with model trained on full data set.
    # This is just for demonstration. Change to test data not seen in during training for full application.
    test_idx = np.array(range(len(X)))

    X = X[test_idx]
    y_reg = y_reg[test_idx]
    y_cl = y_cl[test_idx]
    X, y, sw = load_data(X,y_reg,y_cl)

    print("Loaded samples:",len(y_cl))

    le = OneHotEncoder()
    # sort out noise class for plotting and comparing back-azimuth
    idx = np.where((y_cl!='N'))[0]

    FOLDS = 5

    params = {'input_shape':X.shape[1:],
          'num_outputs':y[0].shape[-1],
          'num_classes':y[1].shape[-1]}

    cmf = lambda : create_model(params)
    model = cmf()

    for i in range(FOLDS):
        model.load_weights(f'tf/output_full/models/{MODEL_NAME}_model_fold_{i}.h5')
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

    print('Accuracy True vs CSM', accuracy_score(y[1].argmax(axis=-1), p_cl))
    print('Accuracy True vs CSM without noise class', accuracy_score(y[1][idx].argmax(axis=-1), p_cl[idx]))

    baz = (np.arctan2(y[0].transpose()[0], y[0].transpose()[1]) * (180/np.pi))[idx]

    try :
        font = {'family' : 'DejaVu Sans',
                 'weight' : 'normal',
                 'size'   : 22}
        plt.rc('font', **font)
        ConfusionMatrixDisplay.from_predictions(y[1].argmax(axis=-1), p_cl, \
                            cmap="Blues", display_labels=['Noise','P','S'])
        matrix = confusion_matrix(y[1].argmax(axis=-1), p_cl)
        print("Class accuracy:",matrix.diagonal()/matrix.sum(axis=1))
        plt.show()
    except AttributeError :
        pass

    print('F1', f1_score(y[1].argmax(axis=-1), p_cl, average=None))

    baz = (np.arctan2(y[0].transpose()[0], y[0].transpose()[1]) * (180/np.pi))[idx]
    plt.hist(baz,180)
    plt.title('Test data',fontsize=20)
    plt.xlabel("Back-azimuth",fontsize=20)
    plt.ylabel("Counts",fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    baz_pred = (np.arctan2(p_reg.transpose()[0], p_reg.transpose()[1]) * (180/np.pi))[idx]
    baz[np.argwhere(baz<0.0)] += 360
    baz_pred[np.argwhere(baz_pred<0.0)] += 360
    plt.plot(baz,baz_pred,'o',alpha=0.1)
    labels=['ArrayNet']
    plt.xlabel('True back-azimuth',fontsize=20)
    plt.ylabel('Predicted baz-azimuth',fontsize=20)
    leg=plt.legend(labels,fontsize=25)
    for lh in leg.legend_handles: lh.set_alpha(1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    ax1=plt.subplot(121)
    angles1 = smallestSignedAngleBetween(baz*np.pi/180,baz_pred*np.pi/180)
    rms1 = np.sqrt(np.sum(np.square(angles1*180/np.pi))/len(angles1))
    mean1 = np.median(angles1*180/np.pi)
    ax1.hist(angles1*180/np.pi,range=(-60,60),bins=91, alpha=0.7)
    ax1.text(12,450,"RMS=%1.1f\nMED=%1.1f" % (rms1,mean1),fontsize=17,c=plt.cm.tab10(1))
    labels=['ArrayNet']
    plt.legend(labels,fontsize=20)
    plt.xticks(np.arange(-60, 65, 30))
    ax1.set_xlabel('BAZ residual',fontsize=20)
    ax1.set_ylabel('Counts',fontsize=20)
    ax1.set_xlim([-65,65])
    ax1.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
