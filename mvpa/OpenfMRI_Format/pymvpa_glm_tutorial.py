#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:44:00 2018

@author: logancross
"""

from mvpa2.tutorial_suite import *


tutorial_data_path = '/Users/logancross/Documents/PyMVPA/tutorial_data'
ds = get_raw_haxby2001_data(path=tutorial_data_path, roi=(36,38,39,40))

poly_detrend(ds, polyord=1, chunks_attr='chunks')

events = find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
events = [ev for ev in events if ev['targets'] in ['house', 'face']]

# temporal distance between samples/volume is the volume repetition time
TR = np.median(np.diff(ds.sa.time_coords))
# convert onsets and durations into timestamps
for ev in events:
    ev['onset'] = (ev['onset'] * TR)
    ev['duration'] = ev['duration'] * TR
    
evds = fit_event_hrf_model(ds, events, time_attr='time_coords', condition_attr=('targets', 'chunks'))

zscore(evds, chunks_attr=None)

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
cv_glm = cv(evds)
print '%.2f' % np.mean(cv_glm)