from __future__ import absolute_import
from __future__ import print_function
import os
import numpy
from string import Template
import csv
import operator
import scipy.stats
import collections
import math
from numpy import log, log10
import six
from six.moves import range

#####################################################################
#Codes designed to take transposed data from jazzKey.py             #
#Assemble statistics about voicings, scale degree sets, progressions#
#####################################################################

#--------------------------------------------------------------------
#Tallying and finding voicings and scale degree (class) sets

def tallySDSets(cmin,probs=False,sendData=False):
    '''
    Takes output of chordFinder (csv of locally-transposed, 50ms scale degree vectors)
    Tallies up all the SD sets with cardinality >= cmin.
    If probs=False, gives tallies; if True, gives probs
    Outputs a csv of (SDset, tallies or probs)
    Returns a dict
    '''
    #Get the scale degree vector data csv
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind TS.csv','r',newline='\n'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row[0:12])#cut off any additional annotation past column 12
    #print(allMidiList)
    
    #Add the vectors to a counter
    SDSets = collections.Counter()
    for midvec in allMidiList:
        SDs = set([])#avoid multisets
        for n in range(12):
            if midvec[n] != '0.0':#any nonzero duration counts
                SDs.add(n)
        if len(SDs) >= cmin:#cardinality restriction
            SDSets[str(sorted(SDs))] += 1
            
    #sort by descending frequency and prepare for csv output
    if probs==True:
        SDSets_probs = getProbsFromFreqs(SDSets)
        sorted_SDSets = sorted(six.iteritems(getProbsFromFreqs(SDSets)),key=operator.itemgetter(1),reverse=True)
        fp = Template('50ms $cm SDSets Probs.csv')
    else:
        sorted_SDSets = sorted(six.iteritems(SDSets), key=operator.itemgetter(1),reverse=True)
        fp = Template('50ms $cm SDSets.csv')

    #send out csv of data
    if sendData:
        csvName = fp.substitute(cm=cmin)
        x = csv.writer(open(csvName, 'w',newline='\n'))
        for pair in sorted_SDSets:
            x.writerow([pair[0], pair[1]])
    if probs==True: return SDSets_probs
           
def sdsSupersetVoicings(sds):
    '''
    For a given scale degree set (sds), locate all relevant SD supersets
    Iterate through transMIDI voiced scale degree set data from jazzKey.py/chordFinder 
    (0<bass<12, upper scale degree voicings)
    Output sorted list of voicing tallies
    '''
    #get transposed, windowed midi data
    allPanes = csv.reader(open('50Do 2xwind transMIDI.csv','r'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
        
    #look for supersets
    SDSetsT = collections.Counter()
    for transmid in allMidiList:
        SDset = set([])
        #if the scale degree set is a superset
        for n in range(len(transmid)):
            SDset.add(int(transmid[n])%12)
        #then tally the voicing
        if SDset.issuperset(sds) == True:
            SDSetsT[str(transmid)] += 1
            
    #write the CSV
    fileName = Template('$voic voiced SDSS 50ms.csv')
    csvName = fileName.substitute(voic = str(sds))
    lw = csv.writer(open(csvName, 'w',newline='\n'))
    sorted_SDSets = sorted(six.iteritems(SDSetsT), key=operator.itemgetter(1),reverse=True)
    for pair in sorted_SDSets:
        lw.writerow([pair[0], pair[1]])

def voicingAsSDS(voicing):
    '''
    for a given (untransposed) voicing, what are its most common SD superset deployments?
    this can be done from the locally-transposed MIDI data (NOT the pc or sd class set data)
    '''
    #get the windowed, transposed midi data
    allPanes = csv.reader(open('50Do 2xwind transMIDI.csv','r'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
        
    #built collection of scale degree superset voicings (no octave reduction)
    SDSetsT = collections.Counter()
    for sliceMidi in allMidiList:
        #print sliceMidi
        SDset = set([])
        for n in sliceMidi:
            SDset.add(int(n))
        #print SDset
        for n in sliceMidi:
            #Build the voicing on each pitch of the slice midi to check for a match
            testVoicing = set([int(n) + p for p in voicing])
            #print testVoicing
            if SDset.issuperset(testVoicing):
                SDSetsT[str(sliceMidi)] += 1
                break
            
    #write the CSV
    fileName = Template('$voic as SDsets 50ms.csv')
    csvName = fileName.substitute(voic = str(voicing))
    lw = csv.writer(open(csvName, 'w'))
    sorted_SDSets = sorted(six.iteritems(SDSetsT), key=operator.itemgetter(1),reverse=True)
    for pair in sorted_SDSets:
        lw.writerow([pair[0], pair[1]])
        
def sdsTracker(sds,mode='chord',then=None):
    '''
    Search for a scale degree class set sds; find the tracks and timestamps where they occur
    Output a (sorted) list of which tracks and the timestamps within them
    if mode == 'chord': lists locations
    if mode == 'prog' and then = another sdc set: find progressions between sds and then
    '''
    #get the timestamped scale degree class data
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind TS.csv','r',newline='\n'))
    allMidiList = []
    allTMS = []
    tms = {}#a dict; each file name will be a key, and the value is a list of times where the sds occurs
    for row in allPanes:
        allMidiList.append(row[0:-2])#just the pitches
        allTMS.append(row[-2:])#just the timestamp and file name
    #print(allMidiList)
    
    #look through scale degree class set data for sds
    for i, midvec in enumerate(allMidiList):
        SDset = set([])#the chord being tested
        for n in range(12):
            if midvec[n] != '0.0':
                SDset.add(n)
        if sorted(SDset) == sds:#the chord matches what we're looking for
            if mode=='chord':
                try:
                    tms[allTMS[i][1]].append(int(allTMS[i][0]))#add a timestamp to the dict entry for this track
                except KeyError:#unless there's no dict entry for that track... in which case it should be a list
                    tms[allTMS[i][1]] = []
                    tms[allTMS[i][1]].append(int(allTMS[i][0]))
            elif mode=='prog':
                for j in range(100):#look at next 100 slices
                    nextSDs = set([])
                    for m in range(12):
                        if allMidiList[i+j][m] != '0.0':
                            nextSDs.add(m)
                    if sorted(nextSDs) == then:#if second chord of prog matches, tally it
                        if allTMS[i][1] not in tms: tms[allTMS[i][1]] = {'time':[],'dist':[]}
                        tms[allTMS[i][1]]['time'].append(int(allTMS[i][0]))#add a timestamp to the dict entry for this track
                        tms[allTMS[i][1]]['dist'].append(j)#distance between chords of prog
                        
    #once all entries are added to the tms dict, listify it and sort by total tally
    outputList = []
    fileName = Template('$voic SDs loc.csv')
    if mode=='chord':
        for key, value in tms.items():
            outputList.append([len(value),key,sorted(value)])
        csvName = fileName.substitute(voic = str(sds))
    elif mode=='prog':
        for key, value in tms.items():
            outputList.append([len(value['time']),key,value['time'],value['dist']])
        csvName = fileName.substitute(voic = str(sds)+'to'+str(then))
    #print(tms)
    print(outputList)
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    sol = sorted(outputList,key=operator.itemgetter(0),reverse=True)
    for row in sol:
        lw.writerow(row)

#----------------------------------------------------------------------------------
#Finding temporal progressions, phonetic and syntactic, forwards and backwards
        
def nAfterSDS(sds,numWin,supersets=False,probs='None',filtp=0.0):
    '''
    Takes a given scale degree set and tallies up what happens with next numWin time windows
    By default, doesn't count scale degree supersets, but it will consider all SDSS as equivalent 
    if supersets==True
    Also normalizes by unigram probabilities if probs='Rel'; just absolute, distance-based probs if probs='Abs'
    Tallies only if probs='None'
    filtp sets a unigram prob cutoff for stuff that gets counted.  Below it, no stats kept (not very helpful)
    If sds is a list, it assumes a single scale degree set and tracks what follows
    If sds is a tuple, first entry is chord to test; second entry is dict of manually-imposed categories
    '''
    
    #Get the scale degree vector data
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind TS.csv','r',newline='\n'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
    #print(allMidiList)
    
    #Start the tally
    #SDSetsT is a counter of the form SDSetsT[time distance][destination chord] = tallies
    SDSetsT = collections.Counter()
    for i, midvec in enumerate(allMidiList):
        #first, assemble each consecutive set for testing
        SDset = set([])
        for n in range(12):
            #print(n)
            if midvec[n] != '0.0':
                SDset.add(n)
                
        """
        if sds is a list, then it's a single scale degree set
        if sds is a tuple (sds, dict), look for all ocs that are in dict[sds]
        Ex: ('ii',{'ii': [[0,2,5],[0,2,5,9],[2,5,9]],'V': [[5,7,11],[2,5,7,11],[2,7,11]], 'I': [[0,4,7],[0,3,7],[0,4,11],[0,3,10],[0,4,7,11],[0,3,7,10]]})
        """
        if type(sds) == list:
            print('Running single origin chord mode')
            #if the chord being tested is of the right kind, assemble stats
            if sorted(SDset) == sds or (supersets == True and SDset.issuperset(sds) == True):
                j = i+1
                while j < i + numWin + 1 and j < len(allMidiList):#time constraints
                    if allMidiList[j][-1] != allMidiList[i][-1]:#if we change tracks, don't count progressions
                        break
                    nextSet = set([])#subsequent chord at time distance j
                    for m in range(12):
                        if allMidiList[j][m] != '0.0':
                            nextSet.add(m)
                    if len(nextSet) < 3:#skip small cardinality chords
                        j += 1
                        continue
                    try:
                        SDSetsT[j-i][str(sorted(nextSet))] += 1
                    except TypeError:
                        SDSetsT[j-i] = collections.Counter()
                        SDSetsT[j-i][str(sorted(nextSet))] += 1
                    j += 1
        elif type(sds) == tuple:
            print('Running manually-defined category mode')
            catDict = sds[1]#all of our categories in one dict, indexed by label
            ocs = catDict[sds[0]]#these are the possible origin chords in category
            #print('possible origin chords in category: ',ocs)
            #Same procedure as above, but allowing stats for any chord in category
            if sorted(SDset) in ocs:
                j = i+1
                while j < i + numWin + 1 and j < len(allMidiList):
                    if allMidiList[j][-1] != allMidiList[i][-1]:
                        break
                    nextSet = set([])
                    #print(allMidiList[j])
                    for m in range(12):
                        if allMidiList[j][m] != '0.0':
                            nextSet.add(m)
                    if len(nextSet) < 3:
                        j += 1
                        continue
                    #make sure to label destination chord with its proper category from catDict
                    counted = False
                    for k,v in catDict.items():
                        if sorted(nextSet) in v:
                            counted = True#if nextSet is in our categories, count it there
                            try:
                                SDSetsT[j-i][str(k)] += 1
                            except TypeError:
                                SDSetsT[j-i] = collections.Counter()
                                SDSetsT[j-i][str(k)] += 1
                    if counted == False:#if nextSet was not in any of our categories, count it normally
                        try:
                            SDSetsT[j-i][str(sorted(nextSet))] += 1
                        except TypeError:
                            SDSetsT[j-i] = collections.Counter()
                            SDSetsT[j-i][str(sorted(nextSet))] += 1
                    j += 1
    
    #print(SDSetsT[0])
    #for each time distance, turn absolute tallies into unigram-relative log probs
    if probs=='Rel':
        uni_probs = tallySDSets(3,probs=True)#unigram probs
        if type(sds) == tuple:
            for k,v in catDict.items():
                uni_probs[k] = 0.0#new unigram prob dict entry for each manual category
                for ve in v:#for each manual category, iterate over included sds
                    uni_probs[k] += uni_probs[str(ve)]#add the probabilities up to get unigram prob for cat
        for row in SDSetsT:#note that these are now distances
            #here's the probs for each sds at given distance
            #note that the probs at each distance are calculated BEFORE low-uniP chords removed
            probs_at_dist = getProbsFromFreqs(SDSetsT[row])
            
            #now normalize them by the sds unigram probs
            for sd in probs_at_dist:
                #if unigram prob below filtp, skip it
                if uni_probs[sd] < filtp:
                    del SDSetsT[row][sd]
                    continue
                #calculate unigram-relative log probability
                logprob = log(probs_at_dist[sd])-log(uni_probs[sd])
                SDSetsT[row][sd] = logprob
                
    #Easy version: each distance gets its own probability normalization, no unigram business
    elif probs=='Abs':
        for row in SDSetsT:
            probs_at_dist = getProbsFromFreqs(SDSetsT[row])
            SDSetsT[row] = probs_at_dist
            
    #now try to stick the counter-of-counters, SDSetsT, in a csv
    cols = set()
    for row in SDSetsT:
        for col in SDSetsT[row]:
            cols.add(col)
    fieldnames = ['nth slice...'] + list(cols)
    #populate row labels
    for row in SDSetsT:
        SDSetsT[row]['nth slice...'] = row
    #write the CSV
    if type(sds) == list:
        if probs=='None':
            fileName = Template('$voic SDs prog 50ms.csv') if supersets == False else Template('$voic SDSS prog 50ms.csv')
        elif probs=='Rel':
            fileName = Template('$voic SDs prog rlogprobs 50ms.csv') if supersets == False else Template('$voic SDSS prog rlogprobs 50ms.csv')
        elif probs=='Abs':
            fileName = Template('$voic SDs prog probs 50ms.csv') if supersets == False else Template('$voic SDSS prog probs 50ms.csv')
        csvName = fileName.substitute(voic = str(sds))
    elif type(sds) == tuple:
        if probs=='None':
            fileName = Template('$voic SDs prog 50ms.csv') if supersets == False else Template('$voic SDSS prog 50ms.csv')
        elif probs=='Rel':
            fileName = Template('$voic SDs prog rlogprobs 50ms.csv') if supersets == False else Template('$voic SDSS prog rlogprobs 50ms.csv')
        elif probs=='Abs':
            fileName = Template('$voic SDs prog probs 50ms.csv') if supersets == False else Template('$voic SDSS prog probs 50ms.csv')
        csvName = fileName.substitute(voic = sds[0])
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in SDSetsT:
        dw.writerow(SDSetsT[row])

def syntax_backwards(sds,numWin,supersets=False,probs='None',filtp=0.0):
    '''
    Close parallel to nAfterSDS
    Takes a given scale degree set and tallies up what happens in the PRECEDING numWin windows
    By default, doesn't count scale degree supersets, but it will consider all SDSS as equivalent if supersets==True
    Generates unigram-relative log probs if probs=='Rel'
    Absolute, distance-based probs if probs=='Abs'
    Tallies instead of probs if probs=='None'
    filtp sets a unigram prob cutoff for stuff that gets counted.  Below it, no stats kept (not very helpful)
    If sds is a list, it assumes a single scale degree set and tracks what follows
    If sds is a tuple, manual given categories are imposed!
    '''
    #Get scale degree vector data
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind TS.csv','r',newline='\n'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
        
    #Start the tally
    #SDSetsT is a counter of the form SDSetsT[time distance][destination chord] = tallies
    SDSetsT = collections.Counter()
    for i, midvec in enumerate(allMidiList):
        SDset = set([])
        for n in range(12):
            if midvec[n] != '0.0':
                SDset.add(n)
                
        """
        if sds is a set, then it's a single scale degree set, and leave the old methods alone
        if sds is a tuple (sds, dict), look for all ocs that are in dict[sds]
        Ex: ('ii',{'ii': [[0,2,5],[0,2,5,9],[2,5,9]],'V': [[5,7,11],[2,5,7,11],[2,7,11]], 'I': [[0,4,7],[0,3,7],[0,4,11],[0,3,10],[0,4,7,11],[0,3,7,10]]})
        """
        if type(sds) == list:#This seems to work backwards correctly
            #print('Running single origin chord mode')
            if sorted(SDset) == sds or (supersets == True and SDset.issuperset(sds) == True):
                j = i-1
                while j > i - numWin - 1:
                    if j < 0 or allMidiList[j][-1] != allMidiList[i][-1]:
                        break
                    nextSet = set([])
                    for m in range(12):
                        if allMidiList[j][m] != '0.0':
                            nextSet.add(m)
                    if len(nextSet) < 3:
                        j -= 1
                        continue
                    try:
                        SDSetsT[j-i][str(sorted(nextSet))] += 1
                    except TypeError:
                        SDSetsT[j-i] = collections.Counter()
                        SDSetsT[j-i][str(sorted(nextSet))] += 1
                    j -= 1
        elif type(sds) == tuple:#Edited to work backwards
            #print('Trying manually-defined category mode')
            catDict = sds[1]#all of our categories in one dict, indexed by label
            ocs = catDict[sds[0]]#these are the possible origin chords in category
            #print('possible origin chords in category: ',ocs)
            if sorted(SDset) in ocs:
                j = i-1
                while j > i - numWin - 1:
                    if j < 0 or allMidiList[j][-1] != allMidiList[i][-1]:
                        break
                    nextSet = set([])
                    for m in range(12):
                        if allMidiList[j][m] != '0.0':
                            nextSet.add(m)
                    if len(nextSet) < 3:
                        j -= 1
                        continue
                    counted = False
                    for k,v in catDict.items():
                        if sorted(nextSet) in v:
                            counted = True#if nextSet is in our categories, count it there
                            try:
                                SDSetsT[j-i][str(k)] += 1
                            except TypeError:
                                SDSetsT[j-i] = collections.Counter()
                                SDSetsT[j-i][str(k)] += 1
                    if counted == False:#if nextSet was not in any of our categories, count it normally
                        try:
                            SDSetsT[j-i][str(sorted(nextSet))] += 1
                        except TypeError:
                            SDSetsT[j-i] = collections.Counter()
                            SDSetsT[j-i][str(sorted(nextSet))] += 1
                    j -= 1
    #print(SDSetsT)
    #for each dist, turn absolute tallies into relative probs
    if probs=='Rel':
        uni_probs = tallySDSets(3,probs=True)
        if type(sds) == tuple:
            for k,v in catDict.items():
                uni_probs[k] = 0.0#new unigram prob dict entry for each manual category
                for ve in v:#for each manual category, iterate over included sds
                    uni_probs[k] += uni_probs[str(ve)]#add the probabilities up to get unigram prob for cat
        for row in SDSetsT:#note that these are now distances
            #here's the probs for each sds at given distance
            #note that the probs at each distance are calculated BEFORE low-uniP chords removed
            probs_at_dist = getProbsFromFreqs(SDSetsT[row])
            #now normalize them by the sds unigram probs
            for sd in probs_at_dist:
                #if unigram prob below filtp, skip it
                if uni_probs[sd] < filtp:
                    del SDSetsT[row][sd]
                    continue
                logprob = log(probs_at_dist[sd])-log(uni_probs[sd])
                SDSetsT[row][sd] = logprob
    elif probs=='Abs':
        for row in SDSetsT:
            probs_at_dist = getProbsFromFreqs(SDSetsT[row])
            SDSetsT[row] = probs_at_dist
    #now try to stick the counter-of-counters, SDSetsT, in a csv
    cols = set()
    for row in SDSetsT:
        for col in SDSetsT[row]:
            cols.add(col)
    fieldnames = ['nth slice...'] + list(cols)
    #populate row labels
    for row in SDSetsT:
        SDSetsT[row]['nth slice...'] = row
    #write the CSV
    if type(sds) == list:
        if probs=='None':
            fileName = Template('$voic SDs backprog 50ms.csv') if supersets == False else Template('$voic SDSS prog 50ms.csv')
        elif probs=='Rel':
            fileName = Template('$voic SDs backprog rlogprobs 50ms.csv') if supersets == False else Template('$voic SDSS prog logprobs 50ms.csv')
        elif probs=='Abs':
            fileName = Template('$voic SDs backprog probs 50ms.csv') if supersets == False else Template('$voic SDSS prog logprobs 50ms.csv')
        csvName = fileName.substitute(voic = str(sds))
    elif type(sds) == tuple:
        if probs=='None':
            fileName = Template('$voic SDs backprog 50ms.csv') if supersets == False else Template('$voic SDSS prog 50ms.csv')
        elif probs=='Rel':
            fileName = Template('$voic SDs backprog rlogprobs 50ms.csv') if supersets == False else Template('$voic SDSS prog logprobs 50ms.csv')
        elif probs=='Abs':
            fileName = Template('$voic SDs backprog probs 50ms.csv') if supersets == False else Template('$voic SDSS prog logprobs 50ms.csv')
        csvName = fileName.substitute(voic = sds[0])
    file = open('Rel Syntax Backwards_rev/'+csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in SDSetsT:
        dw.writerow(SDSetsT[row])

def runtopNchords(numchords,numwind,prb='Rel',backwards=False):
    '''
    Takes a list of the top numchord most probable SDSets from '50ms 3 SDSets.csv'
    Runs each one through nAfterSDS (or syntax_backwards, if backwards=True)
    Spits out the usual csvs; for a given "origin chord," everything that shows up within numwind windows
    '''
    #Get the scale degree sets
    allChords = csv.reader(open('50ms 3 SDSets.csv','r',newline='\n'))
    allChordsList = []#will be populated with the numchords most probable scale degree sets
    
    for i, row in enumerate(allChords):
        #stop at the right number of chords
        if i > numchords - 1:
            break
        
        #kludgy string manipulation to get csv string -> [ints]
        chd = []
        #print(row[0].split(','))
        for j,char in enumerate(row[0].split(',')):
            if j == 0:
                chd.append(int(char[1:]))
                continue
            if j == len(row[0].split(','))-1:
                chd.append(int(char[0:-1]))
                break
            chd.append(int(char))
        #print(chd)
        allChordsList.append(chd)
        
    #print(allChordsList)#So far, so good.  note: lists of ints, not strings
    #run that list of numchords through syntax trackers
    for chd in allChordsList:
        if backwards == False:
            nAfterSDS(chd, numwind, probs=prb, filtp=0.0)
        else:
            syntax_backwards(chd, numwind, probs=prb, filtp=0.0)
            

        
#sdsTracker([2,7,11],mode='prog',then=[0,5,9])
#runtopNchords(1850, 100, prb='Rel',backwards=False)#1850 should be ALL of them.  Run these forwards, too, and zip/send to Jeremy
#syntax_backwards([0,1,2,4,9], 100, probs='Abs', filtp=0.0)
#syntax_backwards(('ii',{'ii': [[0,2,5],[0,2,5,9],[2,5,9]],'V': [[5,7,11],[2,5,7,11],[2,7,11]], 'I': [[0,4,7],[0,3,7],[0,4,11],[0,3,10],[0,4,7,11],[0,3,7,10]]}),100,supersets=False,probs='Rel',filtp=0.0)
#sdsSupersetVoicings([0,2,7])
#nAfterSDS(('I',{'ii': [[0,2,5],[0,2,5,9],[2,5,9]],'V': [[5,7,11],[2,5,7,11],[2,7,11]], 'I': [[0,4,7],[0,3,7],[0,4,11],[0,3,10],[0,4,7,11],[0,3,7,10]]}),150,supersets=False,probs='Rel',filtp=0.0)
#nAfterSDS([2,7,11],100,supersets=False,probs='Abs',filtp=0.0)
#tallySDSets(1,probs=True)             
#voicingAsSDS([0,5,10])