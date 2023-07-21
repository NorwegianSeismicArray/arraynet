# Copyright 2023 Andreas Koehler, MIT license

import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from obspy.geodetics.base import gps2dist_azimuth
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import utm
from phase_pattern import get_coarray
from phase_pattern import CrossSpectralMatrix, qc_stream

# Code for generating input data for ArrayNet

def get_array_geometry(inv):
    dx=[]
    dy=[]
    for stat in inv[0]:
        x,y,_,_=utm.from_latlon(stat.latitude, stat.longitude)
        dx.append(x)
        dy.append(y)
    dx = np.array(dx)
    dy = np.array(dy)
    dx=(dx-dx.mean())/1000.
    dy=(dy-dy.mean())/1000.
    geometry = {}
    for i,stat in enumerate(inv[0]):
        geometry[stat.code] = {}
        geometry[stat.code]['dx']=dx[i]
        geometry[stat.code]['dy']=dy[i]
        geometry[stat.code]['dz']=0.0
    return geometry


if __name__ == '__main__':

    # input parameters
    client = Client('UIB-NORSAR')
    # plot phase pattern and waveforms
    plot = False
    #plot = True
    array = "arces"
    # start time window before pick
    offset = 0.25
    # length of time window including arrival
    twindow = 3.25
    # filter
    fmin=2.5
    fmax=8.0
    # list of frequencies (Hz) for which phase patterns should be computed
    #target_frequencies = [2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0]
    target_frequencies = [2.5,4.0,5.5,7.0]
    # reference data with complete array
    starttime=UTCDateTime(2017, 4, 22, 0, 18, 30)
    # number of missing stations allowed for setting phase patterns to zero for station affected pairs
    missing_stations = 2
    # frequencies to be extracted for training
    training_frequencies = [2.5,4.0,5.5,7.]
    # coherency threshold for training data selection
    coh_thr = 0.0

    # preparing cross-spectral matrix and co-array phase pattern computation
    st_ref = client.get_waveforms('NO','AR*','*','BHZ,sz,bz',starttime,starttime+twindow)
    inv = client.get_stations(network='NO',station='AR*',starttime=starttime,endtime=starttime+twindow)
    geometry = get_array_geometry(inv)
    coarray=get_coarray(geometry,plot=plot,full=False)
    missing_pairs = missing_stations * len(st_ref)
    r = CrossSpectralMatrix(st_ref,spec_method='PRIETO')
    # find frequency indices for desired band
    flist=[]
    for freq in target_frequencies:
        flist.append(np.argmin(np.abs(r.flist - freq)))
    freqkeys_train=[]
    for freq in training_frequencies:
        freqkeys_train.append("%1.1f" % r.flist[np.argmin(np.abs(r.flist - freq))])
    pattern_train = r.csm_coarray_pattern(coarray,flist[0],plot=False)
    print("Frequencies:",[r.flist[f_ind] for f_ind in flist])
    print("Training Frequencies:",freqkeys_train)
    print("Number of station pairs:",len(pattern_train))

    trainingdata = {
        'phases' : [],
        'baz' : [],
        'label' : [],
        'arrival time' :[],
        'coherency' : []
    }

    # Read arrival times, labels and back-azimuth from training data set and re-generate input data
    # Use only a small sub-set for testing
    # samples = -1
    samples = 1000
    # Add you own arrival picks here
    # Storing waveforms locally instead of requesting via client for each arrival could speed up processing
    catalog = 'merged'
    dataset = 'data'
    fbands = '_4Fre'
    # inculding regional phases for sub-model
    regional = True
    if regional : fbands = '_4Fre_regional'
    PATH = 'tf/' + dataset +'/'
    arrival_times = np.load(PATH + 'times_'+catalog+'_'+array+fbands+'.npy',allow_pickle=True)[:samples]
    y = np.load(PATH + 'y_reg_'+catalog+'_'+array+fbands+'.npy')[:samples]
    baz = np.arctan2(y.transpose()[0], y.transpose()[1]) * (180/np.pi)
    labels = np.load(PATH + 'y_cl_'+catalog+'_'+array+fbands+'.npy')[:samples]

    # loop over arrivals and compute phase patterns
    for starttime,label,baz_orig in zip(arrival_times,labels,baz) :
        if starttime is None : continue
        try:
            # loading longer time window including arrival
            st_org = client.get_waveforms('NO','AR*','*','BHZ,sz,bz',starttime-10,starttime+30)
            if len(st_org) < 2 :
                print(starttime,label,'Only one or less stations. Skipping')
                continue
        
        except ( ValueError, FDSNNoDataException ) as e :
            print(starttime,label,'Something wrong with waveforms. Skipping.')
            continue
        print(starttime,label)

        # preprocessing
        st_org.detrend()
        st_org.taper(0.1)
        st_org.filter('bandpass',freqmin=fmin, freqmax=fmax);
        if plot : st_org[:8].plot()
        st=st_org.copy().trim(starttime-offset,starttime + twindow - offset)
        try :
            qc_stream(st)
            r.nchan = len(st)
            r.chanlist=[tr.stats.station for tr in st]
            r.compute_CSM(st)
        except Exception as e :
            print("Something wrong with cross-spectral matrix computation (most likely Stream qc failed)",e)
            continue

        phases_allf = {}
        coherence_allf = {}
        # loop over frequencies and station pairs and extract phase differences as vector
        for f_ind in flist :
            if plot : print(r.flist[f_ind],'Hz')
            pattern=r.csm_coarray_pattern(coarray,f_ind,plot=plot)
            phases=[]
            coherence = []
            missing=0
            for key in pattern_train :
                try :
                    phases.append([np.cos(pattern[key]['phase']),np.sin(pattern[key]['phase'])])
                    coherence.append(pattern[key]['coherency'])
                except KeyError :
                    phases.append([0.0,0.0])
                    coherence.append(0.0)
                test = np.sqrt(phases[-1][0]*phases[-1][0]+phases[-1][1]*phases[-1][1])
                if test < 0.999 : missing = missing + 1
            f_key = "%1.1f" % r.flist[f_ind]
            phases_allf[f_key] = asarray(phases).flatten()
            coherence_allf[f_key] = asarray(coherence).flatten()
        if missing < missing_pairs :
            trainingdata['arrival time'].append(starttime)
            trainingdata['phases'].append(phases_allf)
            trainingdata['label'].append(label)
            trainingdata['coherency'].append(coherence_allf)
            if label == 'N' :
                slowx = 0.0
                slowy = 0.0
            else :
                slowx = np.sin(baz_orig/180 * np.pi)
                slowy = np.cos(baz_orig/180 * np.pi)
            trainingdata['baz'].append([slowx,slowy])
        else : print(missing," station pairs missing. Skipping event.")

    # Concatenating frequencies
    trainingdata['phases']=[np.concatenate([pha[key] for key in freqkeys_train]) for pha in trainingdata['phases']]
    trainingdata['coherency']=[np.concatenate([coh[key] for key in freqkeys_train]) for coh in trainingdata['coherency']]

    # Mean coherency
    meancoh=np.mean(np.asarray(trainingdata['coherency']),axis=1)
    # Pre-select data for trainin above coherency threshold
    idx = np.where(((np.array(trainingdata['label']) == 'N') | ( meancoh > coh_thr )) & (np.isfinite(trainingdata['phases']).all(axis=-1)) )[0]
    times = np.array(trainingdata['arrival time'])[idx]
    x = np.array(trainingdata['phases'])[idx]
    y = np.array(trainingdata['baz'])[idx]
    y_class = np.array(trainingdata['label'])[idx]

    # plot the back-azimuth distribution
    idx_real = np.where(y_class != 'N')[0]
    baz_real = np.arctan2(y[idx_real].transpose()[0], y[idx_real].transpose()[1]) * (180/np.pi)
    counts_real, bins_real, bars = plt.hist(baz_real, bins=91)
    plt.xlabel("Back-azimuth")
    plt.ylabel("Counts")
    plt.title(array.upper())
    #plt.yscale('log')
    plt.gca().set_ylim(bottom=1)
    plt.show()

    catalog = 'test'
    if regional: catalog = 'test_regional'
    datapath = 'tf/data/'
    np.save(f'{datapath}/times_{catalog}_{array}.npy', times)
    np.save(f'{datapath}/X_{catalog}_{array}.npy', x)
    np.save(f'{datapath}/y_reg_{catalog}_{array}.npy', y)
    np.save(f'{datapath}/y_cl_{catalog}_{array}.npy', y_class)

    print("Number of samples:",len(trainingdata['phases']))
    print("Input data dimension:",len(trainingdata['phases'][0]))
    print("Phase patterns per frequency band:",int(len(trainingdata['phases'][0])/(2*len(freqkeys_train))))
    print("Classes:",set(trainingdata['label']))
    for key in set(trainingdata['label']):
        print(key,trainingdata['label'].count(key))
