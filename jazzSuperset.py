from music21 import *
from sets import *
import operator
import numpy
import scipy.stats
import os
import csv
import pickle

'''
GOAL: From all the (binned?) slices, make intelligent bigrams from common large chords
1.  look for cases of OC -> DC where corpus-wide p(OC) < p(superset(OC)).
    Based on voicing or pcset?
    If more than one, count the most probable superset(OC)?
    I'm guessing not many of these, since p ~ #(OC)^-n
    Count how many times this happens?
2.  Build a bigram tally where B1 is most probable member of supersets(OC) and B2 of supersets(DC)
'''

def supersetBigrams(bassSize):
    from string import Template
    sliceCount = {}
    theImportantSlices = []
    skippedThings = 0
    corpusSize = 0
    binnedThings = 0
    supersettedThings = 0
    #Load the pickled slices that have not been bass-normalized into types
    theSlices = pickle.load( open ('quantizedChordDictSlices1122.pkl', 'rb') )
    for i, slicey in enumerate(theSlices):
        if slicey == ['start'] or slicey == ['end']:
            continue
        #keep count of the total number of slices before reduction
        corpusSize += 1
        if theSlices[i+1] == ['end']:
            continue
        #First, deal with singletons of bass motion 0
        if len(slicey['midi']) == 1 and theSlices[i]['bass'] - theSlices[i+1]['bass'] == 0:
            skippedThings += 1
            continue
        #Next, only look at cases where |bass motion| > bassSize
        if abs(theSlices[i+1]['bass'] - theSlices[i]['bass']) > bassSize:
            secondSliceMidi = []
            for n in theSlices[i+1]['midi']:
                secondSliceMidi.append((n+theSlices[i+1]['bass']))
            firstSliceMidi = []
            for m in theSlices[i]['midi']:
                firstSliceMidi.append((m+theSlices[i]['bass']))
            #make sure second thing is superset of first thing
            continueIfZero = 0
            #even one note wrong means no!
            for n in firstSliceMidi:
                if n not in secondSliceMidi:
                    continueIfZero += 1
                    break
            #If it passes bass motion and superset test, skip it
            if continueIfZero == 0:
                skippedThings += 1
                continue
        #if the slice is still around, it's "important"    
        theImportantSlices.append(slicey)
    #Now, from the important ones, find voicing probs
    for slicey in theImportantSlices:
        try:
            sliceCount[str(slicey['midi'])] += 1
        except KeyError:
            sliceCount[str(slicey['midi'])] = 1
    sliceProbs = getProbsFromFreqs(sliceCount)
    #Now make a list of the really important slices
    theReallyImportantSlices = []
    skipNext = 0
    #OK, now go again, looking for non-superset bass leaps
    for i, slicey in enumerate(theImportantSlices):
        if i == len(theImportantSlices) - 1:
            break
        if skipNext == 1:
            skipNext = 0
            continue
        #Next, only look at cases where |bass motion| > bassSize
        if abs(theImportantSlices[i+1]['bass'] - theImportantSlices[i]['bass']) > bassSize:
            combinedSlices = []
            for n in theImportantSlices[i]['midi']:
                combinedSlices.append(n + theImportantSlices[i]['bass'])
            for m in theImportantSlices[i+1]['midi']:
                if m + theImportantSlices[i+1]['bass'] in combinedSlices:
                    continue
                combinedSlices.append(m + theImportantSlices[i+1]['bass'])
            sortedSlice = sorted(combinedSlices)
            sortedSlice_type = [x - sortedSlice[0] for x in sortedSlice]
            try:
                testProb = sliceProbs[str(sortedSlice_type)]
            except KeyError:
                theReallyImportantSlices.append(slicey)
                continue
            #Deal with singletons, which always have higher p
            #If both are singletons:
            if len(slicey['midi']) == 1 and len(theImportantSlices[i+1]['midi']) == 1:
                continue
            #If the first is a singleton
            if len(slicey['midi']) == 1 and len(theImportantSlices[i+1]['midi']) > 1:
                if testProb < sliceProbs[str(theImportantSlices[i+1]['midi'])]:
                    continue
            #If the second is a singleton
            if len(theImportantSlices[i+1]['midi']) == 1 and len(slicey['midi']) > 1:
                if testProb < sliceProbs[str(slicey['midi'])]:
                    continue
            combinedSlice = {}
            combinedSlice['bass'] = sortedSlice[0]
            combinedSlice['midi'] = sortedSlice_type
            theReallyImportantSlices.append(combinedSlice)
            skipNext = 1
            binnedThings += 1
    #Now, from the binned/REALLY important ones, find voicing probs
    binnedCount = {}
    for slicey in theReallyImportantSlices:
        #Tally up the count of the slice voicing
        try:
            binnedCount[str(slicey['midi'])] += 1
        except KeyError:
            binnedCount[str(slicey['midi'])] = 1
    binnedProbs = getProbsFromFreqs(binnedCount)
    #Now go through the slices and make (potentially-supersetted) bigrams
    bigramTally = {}
    #look for chords st p(OC) < p(superset(OC))
    for slicey in theReallyImportantSlices:
        if len(slicey['midi']) == 1:
            originChord = slicey['midi']
        else:
            sliceyProb = binnedProbs[str(slicey['midi'])]
            bestSupersetProb = sliceyProb
            bestSuperset = slicey['midi']
            #Turn the dict key strings into things we can check for supersets
            for key, value in binnedProbs.iteritems():
                if value <= bestSupersetProb:
                    continue
                keyparts = key.strip('[]').split(',')
                keyvoicing = [int(n) for n in keyparts]
                #This is reasonably clever, but take into account transpositional diffs!
                for n in keyvoicing:
                    #Build the voicing on each pitch of the slice midi to check for a match
                    testVoicing = [n + p for p in slicey['midi']]
                    #This is just a toggle
                    zeroIsMatch = 0
                    for m in testVoicing:
                        #If any pitch from the built voicing isn't in the slice, build a different one
                        if m not in keyvoicing:
                            zeroIsMatch += 1
                            break
                    #If all the pitches of the built voicing ARE in the slice, it's a superset slice
                    if zeroIsMatch == 0:
                        supersettedThings += 1
                        bestSuperset = keyvoicing
                        bestSupersetProb = value
                        break
            originChord = bestSuperset
            
            
                for n in slicey['midi']:
                    if n not in keyvoicing:
                        break
                    bestSupersetProb = value
                    bestSuperset = keyvoicing
                
                
                
                
                                 
        
        try:
            bigramTally[str(originChord)][str(distAboveBass)] += 1
        except KeyError:
            try:
                bigramTally[startToken][str(distAboveBass)] = 1
            except KeyError:
                bigramTally[startToken] = {}
                bigramTally[startToken][str(distAboveBass)] = 1
            
                
def cardinalityScale():
    #Load the pickled slices that have not been bass-normalized into types
    sliceDict = {}
    theSlices = pickle.load( open ('quantizedChordDictSlices1122.pkl', 'rb') )
    for slicey in theSlices:
        if slicey == ['start'] or slicey == ['end']:
            continue
        sliceCard = len(slicey['midi'])
        try:
            sliceDict[str(sliceCard)] += 1
        except KeyError:
            sliceDict[str(sliceCard)] = 1
    sorted_sliceDict = sorted(sliceDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    print 'how the cardinality breaks down',sorted_sliceDict

#cardinalityScale()
