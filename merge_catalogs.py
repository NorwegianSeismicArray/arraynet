import matplotlib.pyplot as plt
import numpy as np
import pickle
from numpy import asarray
import os
import sys
from utilities import find_best_match,add_to_tree

# merged catalogs from output of extract_P_and_S_arrivals_from_bulletins_withcoherence.py or extract_P_and_S_arrivals_from_bulletins.py

if __name__ == '__main__':

    pass

    datapath =''
    array = 'arces'
    #array = 'spits'
    #array = 'nores'

    basename = 'traindata_'
    #basename = 'traindata_cohthr_'

    catalog = 'helsinki'
    training_data=pickle.load(open(datapath+basename+catalog+'_'+array+'.p','rb'))
    times_helsinki = np.array(training_data['arrival time'])
    label_helsinki = np.array(training_data['label'])
    baz_helsinki = np.array(training_data['baz'])

    catalog = 'norsar'
    training_data=pickle.load(open(datapath+basename+catalog+'_'+array+'.p','rb'))
    times_norsar = np.array(training_data['arrival time'])
    label_norsar = np.array(training_data['label'])
    baz_norsar = np.array(training_data['baz'])

    tree = {} 
    for t in times_helsinki : add_to_tree(tree,t)
    idx=[]
    for i,t in enumerate(times_norsar) :
        best_match = find_best_match(tree,t,2.0)
        if best_match is not None :
            #print(t,best_match)
            idx.append(i)
    print(len(times_norsar),"phases in norsar cat")
    print(len(times_helsinki),"phases in helsinki cat")
    print(len(idx),"duplicates")
    times = np.concatenate((np.delete(times_norsar,idx),times_helsinki),axis=0)
    label = np.concatenate((np.delete(label_norsar,idx),label_helsinki),axis=0)
    baz = np.concatenate((np.delete(baz_norsar,idx),baz_helsinki),axis=0)

    print(len(times),"phases after merging catalogs")

    print("P waves:",len([lab for lab in label if lab[0] == 'P']))
    print("S waves:",len([lab for lab in label if lab[0] == 'S']))
    print("Noise:",len([lab for lab in label if lab[0] == 'N']))


    trainingdata = {
        'baz' : baz,
        'label' : label,
        'arrival time' :times
    }

    catalog = 'merged'
    outdir = "./"
    pickle.dump(trainingdata,open(outdir+basename+catalog+"_"+array+".p","wb"))


def main():
    datapath = 'rawdata/'
    array = 'arces'
    basename = 'traindata_cohthr_'
    catalog = 'helsinki'
    training_data1 = pickle.load(open(datapath + basename + catalog + '_' + array + '.p', 'rb'))
    catalog = 'norsar'
    training_data2 = pickle.load(open(datapath + basename + catalog + '_' + array + '.p', 'rb'))

    tree = {}
    for t in training_data1['arrival time']:
        add_to_tree(tree, t)
    idx = []
    for i, t in enumerate(training_data2['arrival time']):
        best_match = find_best_match(tree, t, 2.0)
        if best_match is not None:
            idx.append(i)

    idx_to_keep = np.setdiff1d(np.arange(len(training_data2['arrival time'])), np.asarray(idx))
    for key in training_data2:
        training_data2[key] = [a for i, a in training_data2[key] if i in idx_to_keep]

    for key in training_data1:
        training_data1[key].extend(training_data2[key])

    catalog = 'merged'
    pickle.dump(training_data1,open(datapath+basename+catalog+"_"+array+".p","wb"))

main()




