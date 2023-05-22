from obspy import UTCDateTime
from seismonpy.norsardb import Client
from seismonpy.array_analysis.cross_spectral_fk import get_coarray
from seismonpy.array_analysis.cross_spectral_fk import CrossSpectralMatrix
client = Client()
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import pickle
import os
import sys
from obspy.geodetics.base import gps2dist_azimuth
from utilities import events_from_mongodb, events_from_oracledb, events_from_helsinki
from obspy.signal.trigger import z_detect, trigger_onset
from tqdm import tqdm

try:
    from tslearn.preprocessing import TimeSeriesResampler
except:
    pass

def flatten(t):
    return [item for sublist in t for item in sublist]

# Andreas Koehler, 18/01/2022
# This script extracts arrival times, phase type label and back-azimuth (towards event origin) for a given array and catalog and write it to a dictionary stored as pickle file
# It also checks the coherency of the waveform arrivals over the array and sort out below a threshold

day_length = 24*60*60

def adjust_starttime(array, starttime):
    # array instrumntation changes over time!
    # This was no problem for time differences but may be for learning waveforms unless instrument correction is applied!
    if array == "arces":
        # stime=UTCDateTime(2014, 9, 27, 0, 0, 0) # ARCES starts with BH channels, before sz and bz
        starttime = max(starttime, UTCDateTime(1998, 1, 1, 0, 0, 0))
        statcode = 'ARCES'
    if array == "nores":
        # stime=UTCDateTime(2015, 8, 13, 0, 0, 0) # NORES upgrade from SHZ to HHZ
        starttime = max(starttime, UTCDateTime(2011, 1, 1, 0, 0, 0))  # NORES re-opened
        statcode = 'NORES'
    if array == "spits":
        # stime=UTCDateTime(2015, 5, 8, 0, 0, 0) # SPITS upgrade from BHZ to HHZ
        starttime = max(starttime, UTCDateTime(2004, 8, 12, 0, 0, 0))  # SPITS upgrade from sz (40 Hz) to BHZ (80 Hz)
        statcode = 'SP'
    return starttime, statcode

def get_events(catalog, starttime, endtime, statcode, phaselist=['all']):
    outputs = []

    new_stime = starttime
    old_stime = starttime + 1
    with tqdm(total=int((endtime-starttime)/day_length), desc='Loading events, days:') as pbar:

        while new_stime < endtime:

            pbar.update(int((new_stime-old_stime)/day_length))

            old_stime = new_stime
            sys.stdout.flush()
            if catalog == 'norsar':
                events,arrivals,locations,labels,events_ids = events_from_mongodb(new_stime,new_stime+14*24*3600,phaselist,statcode)
                if len(events) > 0:
                    new_stime = UTCDateTime(events[-1]['origins'][0]['time']) + 1.0
                else:
                    break

            if catalog == 'helsinki':
                # read day-wise
                try :
                    events,arrivals,locations,labels,events_ids = events_from_helsinki(new_stime,new_stime,phaselist,statcode)
                except Exception as e:
                    print(e)
                    print("No events because of some error this day")
                new_stime += 24*3600

            if len(events) > 0:
                outputs.extend([dict(event_id=a,
                                arrival=b,
                                location=c,
                                label=d) for a,b,c,d in zip(events_ids,arrivals,locations,labels)])
    return outputs

def get_noise_events(events, noise_offset=180):
    for event in events:
        if 'P' in event['label']:
            event['arrival'] -= noise_offset
            event['label'] = 'N'
            events.append(event)
    return events

def get_single_waveform(array, loc, starttime, before_event, after_event):
    baz_orig = None
    try:
        if array == "arces":
            st_org = client.get_waveforms("AR*", "BHZ,sz,bz", starttime - before_event, starttime + after_event)
            if loc is not None: _, baz_orig, _ = gps2dist_azimuth(69.5349, 25.5058, loc[1], loc[0])
        if array == "nores":
            st_org = client.get_waveforms("NRA*,NRB*", "HHZ,SHZ", starttime - before_event,
                                          starttime + after_event)
            if loc is not None: _, baz_orig, _ = gps2dist_azimuth(60.7353, 11.5414, loc[1], loc[0])
        if array == "spits":
            st_org = client.get_waveforms("SP*", "HHZ,BHZ,sz,bz", starttime - before_event,
                                          starttime + after_event)
            if loc is not None: _, baz_orig, _ = gps2dist_azimuth(78.1777, 16.3700, loc[1], loc[0])
    except ValueError:
        print("Something wrong with waveforms (sampling rate etc.)")
        return False

    return baz_orig, st_org

def onsets(st_org):
    return len(flatten([trigger_onset(z_detect(trace.data, int(10 * trace.stats.sampling_rate)), 2.5, 0.2) for trace in st_org]))

def coherence(st_org, coarray, starttime, cov_before_event, cov_after_event):
    st = st_org.copy().trim(starttime - cov_before_event, starttime + cov_after_event)
    try:
        r = CrossSpectralMatrix(st, spec_method='PRIETO')
    except Exception as e:
        return 0.0
    flist = np.array([i for i, f in enumerate(r.flist) if f > 2.0 and f < 10.0])
    patterns = list(map(lambda i: r.csm_coarray_pattern(coarray, i, plot=False), flist))
    ch = [[pattern[key]['coherency'] if key in pattern else float('nan') for key in pattern] for pattern in patterns]
    return np.nanmean(flatten(ch))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_single_event(event, array, coarray, stations, window, cov_window, th, downsampling):
    before_event, after_event = window
    cov_before_event, cov_after_event = cov_window

    starttime = event['arrival']
    loc = event['location']
    label = event['label']
    event_id = event['event_id']

    try:
        baz_orig, st_org = get_single_waveform(array, loc, starttime, before_event, after_event)
    except IndexError:
        return None

    mean_coherence = -1
    num_onsets = 1

    if label == 'N':
        num_onsets = onsets(st_org)
    elif th > 0:
        mean_coherence = coherence(st_org, coarray, starttime, cov_before_event, cov_after_event)
    else:
        mean_coherence = 1
        num_onsets = 0

    if len(st_org.traces) > 0 and ((label != 'N' and mean_coherence > th) or (label == 'N' and num_onsets == 0)):
        trace_length = int(st_org.traces[0].stats.sampling_rate * (before_event + after_event))
        st_stations = [trace.stats.station for trace in st_org]
        d = [st_org.select(station=station).traces[0].data[:trace_length] if station in st_stations else np.zeros(trace_length) for station in stations]

        lengths = list(map(lambda a: len(a) == trace_length, d))
        if not all(lengths):
            return None

        d = np.asarray(d).T
        if downsampling is not None:
            try:
                d = np.squeeze(TimeSeriesResampler(downsampling).fit_transform(np.expand_dims(d,axis=0)))
            except:
                pass

        return event_id, starttime, loc, label, baz_orig, d
    else:
        return None

from multiprocessing import Pool

def get_waveforms(events, array, coarray, stations, window=(60,120), cov_window=(3,3), th=0.3, downsampling=None):

    outputs = {'arrival':[],'location':[],'label':[],'event_id':[],'data':[],'baz':[]}

    for item in map(lambda event: process_single_event(event, array, coarray, stations, window, cov_window, th, downsampling),
                    tqdm(events, desc='Loading waveforms')):
        if item is not None:
            event_id, arrival, location, label, baz, data = item
            outputs['event_id'].append(event_id)
            outputs['arrival'].append(arrival)
            outputs['location'].append(location)
            outputs['label'].append(label)
            outputs['baz'].append(baz)
            outputs['data'].append(data)

    return outputs

def main():
    # Helsinki catalog in nordic format available from 1998
    stime = UTCDateTime(2000, 1, 1, 0, 0, 0)  # changed for individual arrays below
    etime = UTCDateTime(2022, 1, 1, 0, 0, 0)

    array = "arces"  # Helsinki and NORSAR catalog
    # array = "nores" # only NORSAR
    # array = "spits" # only NORSAR

    #catalog = 'norsar'
    catalog = 'helsinki'

    if array != 'arces' and catalog == 'helsinki':
        print("Helsinki catalog has only stations from ARCES array")
        exit()

    phaselist = ['all']
    if catalog == 'norsar':
        phaselist = ['Pn', 'Pg', 'P', 'Sn', 'Sg', 'S', 'Lg', 'LG']
    if catalog == 'helsinki':
        phaselist = ['PN', 'PG', 'P', 'SN', 'SG', 'S', 'PB', 'SB']

    starttime, statcode = adjust_starttime(array, stime)
    inv = client.get_array_inventory(array.upper())
    geometry = inv.array_offsets()
    coarray = get_coarray(geometry, plot=False, full=False)

    stations = list(map(lambda s: s.replace('NO.','').replace(')','').replace('(','').strip() ,inv.get_contents()['stations']))

    events = get_events(catalog, starttime, etime, statcode, phaselist)
    events = get_noise_events(events, noise_offset=5*60)
    wf = get_waveforms(events,array,coarray,stations,window=(60,120), cov_window=(0.25,3), th=0.1, downsampling=None)

    print('Datapoints:',len(wf['event_id']))

    datapath = '/nobackup/erik/data/'
    basename = 'traindata_cohthr_'
    pickle.dump(wf, open(datapath + basename + catalog + '_' + array + '.p', 'wb'))

if __name__ == '__main__':
    main()

