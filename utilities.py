from obspy import UTCDateTime
from seismonpy.norsardb import Client
client = Client()
import matplotlib.pyplot as plt
import numpy as np
import pymongo
from obspy import read_events
from obspy.io.nordic import NordicParsingError

arces_stations = ['ARA0', 'ARA1', 'ARA2', 'ARA3', 'ARB1', 'ARB2', 'ARB3', 'ARB4', 'ARB5', 'ARC1', 'ARC2', 'ARC3', 'ARC4', 'ARC5', 'ARC6', 'ARC7', 'ARD1', 'ARD2', 'ARD3', 'ARD4', 'ARD5', 'ARD6', 'ARD7', 'ARD8', 'ARD9', 'ARE0']


def get_arrivals_and_locations(events,phaselist,statcode):
    arrivals=[]
    locations=[]
    label = []
    event_ids = []
    for event in events :
        for arrival in event['origins'][0]['arrivals']:
            if arrival['phase'] in phaselist or phaselist[0] == 'all' :
                pick_id = arrival['pick_id']
                for pick in event['picks']:
                    if pick['resource_id'] == pick_id :
                        stat = pick['waveform_id']['station_code']
                        if statcode == "NORES" :
                            if stat == "NORES" or stat == "NC602" :
                                arrivals.append(UTCDateTime(pick['time']))
                                label.append(arrival['phase'])
                                locations.append([event['origins'][0]['longitude'],event['origins'][0]['latitude']])
                        elif statcode == "ARCES" :
                            if stat == "ARCES" or stat in arces_stations :
                                arrivals.append(UTCDateTime(pick['time']))
                                label.append(arrival['phase'])
                                locations.append([event['origins'][0]['longitude'],event['origins'][0]['latitude']])
                        # SPITS and single SPA* and SPB* are being used in norsar bulletin, no other station SP*
                        else :
                            if stat[0:len(statcode)] == statcode :
                                arrivals.append(UTCDateTime(pick['time']))
                                label.append(arrival['phase'])
                                locations.append([event['origins'][0]['longitude'],event['origins'][0]['latitude']])
                        event_ids.append(event['resource_id'])

    return arrivals,locations,label,event_ids


def events_from_mongodb(stime,etime,phaselist,statcode):
    # complains if more events - do querry in batches of 100
    maxnumevents = 100
    dbclient = pymongo.MongoClient("mongodb://guest:guest@mongo.norsar.no/?authSource=test")
    rewdb = dbclient.seismon
    #rewcol = rewdb.events
    rewcol = rewdb.reviewed
    events = list(rewcol.find({"$and":[{"origins.evaluation_status": "final"},{"origins.time": {"$gt": stime.format_iris_web_service()}},{"origins.time": {"$lt": etime.format_iris_web_service()}}]}).sort("origins.time").limit(maxnumevents))
    arrivals,locations,labels,events_ids = get_arrivals_and_locations(events,phaselist,statcode)
    return events,arrivals,locations,labels,events_ids

def events_from_oracledb(stime,etime,phaselist,statcode):
    # has to be done in shell before
    #export ORACLE_HOME=/ndc/programs/oracle/product/11.2.0
    #export LD_LIBRARY_PATH=${ORACLE_HOME}/lib
    events=client.get_events(stime,etime,includearrivals=True,orderby='time')
    arrivals,locations,labels,events_ids = get_arrivals_and_locations(events,phaselist,statcode)
    return events,arrivals,locations,labels,events_ids

def events_from_helsinki(stime,etime,phaselist,statcode):
    nextday = stime
    while nextday.julday <= etime.julday and nextday.year == etime.year :
        print("/projects/active/MMON/Array_detection/ML_methods/helsinki-catalog/nordic/%d/%d%03d.nordic" % (nextday.year,nextday.year,nextday.julday))
        try :
            try : events.extend(read_events("/projects/active/MMON/Array_detection/ML_methods/helsinki-catalog/nordic/%d/%d%03d.nordic" % (nextday.year,nextday.year,nextday.julday)))
            except NameError : events = read_events("/projects/active/MMON/Array_detection/ML_methods/helsinki-catalog/nordic/%d/%d%03d.nordic" % (nextday.year,nextday.year,nextday.julday))
            nextday += 24*3600
        except NordicParsingError as e:
            print(e)
            nextday += 24*3600
            continue
        #if len(events)>100 : break
    arrivals,locations,labels,events_ids = get_arrivals_and_locations(events,phaselist,statcode)
    return events,arrivals,locations,labels,events_ids

def initialize_tree(tree, utctime):
    """ Populate the tree with year-month-day-hour values from 'utctime' """
    if not utctime.year in tree:
        tree[utctime.year] = {}

    if not utctime.month in tree[utctime.year]:
        tree[utctime.year][utctime.month] = {}

    if not utctime.day in tree[utctime.year][utctime.month]:
        tree[utctime.year][utctime.month][utctime.day] = {}

    if not utctime.hour in tree[utctime.year][utctime.month][utctime.day]:
        tree[utctime.year][utctime.month][utctime.day][utctime.hour] = []


def add_to_tree(tree, t):
    # Initialize the search tree
    initialize_tree(tree, t)

    tree[t.year][t.month][t.day][t.hour].append(t)

    # If on 'the edge' to previous or next hour, add it there too
    if t.minute < 5:
        prev_t = UTCDateTime(t) - 3600
        initialize_tree(tree, prev_t)
        tree[prev_t.year][prev_t.month][prev_t.day][prev_t.hour].append(t)

    if t.minute > 55:
        next_t = UTCDateTime(t) + 3600
        initialize_tree(tree, next_t)
        tree[next_t.year][next_t.month][next_t.day][next_t.hour].append(t)


def find_best_match(tree,t,match_threshold):
    best_match = None
    try : potential_matches = tree[t.year][t.month][t.day][t.hour]
    except : potential_matches = []
    for pm in potential_matches:
        diff = abs(t - pm)
        if diff < match_threshold:
            if best_match is None:
                best_match = pm
            else:
                if diff < (t - best_match):
                    best_match = pm
    return best_match
