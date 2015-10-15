from music21 import *
from sets import *
import csv
import pickle
import numpy
import os
import operator
import midi
import scipy.stats
from music21.ext.jsonpickle.util import itemgetter
from openpyxl.reader.excel import load_workbook
import collections
from string import Template

path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIunquant/'
                

def voicedPCSet(pcs):
    """
    For a given PCSet, looks through and tallies the voicing instantiations of it
    """
    theSlices = pickle.load( open ('pickles/quantizedChordDictSlices1015.pkl', 'rb') )
    voicings = collections.Counter()
    for i, slicey in enumerate(theSlices):
        #if i > 10:
            #break
        if slicey == ['start'] or slicey == ['end']:
            continue
        slicePC = set()
        for mn in slicey['midi']:
            slicePC.add(mn%12)
        PCset = chord.Chord(slicePC).primeForm
        if PCset == pcs:
            voicings[str(slicey['midi'])] += 1
    #tally up the frequencies for each chord
    sorted_voicings = sorted(voicings.iteritems(), key=operator.itemgetter(1), reverse=True)
    #export the tally as a csv file
    cs = Template('$p is voiced_1015.csv')
    csvName = cs.substitute(p = str(pcs))
    x = csv.writer(open(csvName, 'wb'))
    for pair in sorted_voicings:
        x.writerow([pair[0], pair[1]])

def tallyPCSets(cmin):
    """
    Looks through all the lightly quantized tracks (as of Sept 2015) and tallies verticalities of pcset cardinality >= cmin
    """
    theSlices = pickle.load( open ('pickles/quantizedChordDictSlices1015.pkl', 'rb') )
    PCSets = collections.Counter()
    for i, slicey in enumerate(theSlices):
        #if i > 10:
            #break
        if slicey == ['start'] or slicey == ['end']:
            continue
        slicePC = set()
        for mn in slicey['midi']:
            slicePC.add(mn%12)
        PCs = chord.Chord(slicePC).primeForm
        if len(PCs) >= cmin:
            PCSets[str(PCs)] += 1
    #tally up the frequencies for each chord
    sorted_PCSets = sorted(PCSets.iteritems(), key=operator.itemgetter(1), reverse=True)
    #export the tally as a csv file
    csvName = 'quantizedPCSets1015.csv'
    x = csv.writer(open(csvName, 'wb'))
    for pair in sorted_PCSets:
        x.writerow([pair[0], pair[1]])

def categoryFinder(bassSize):
    """
    Looks through pickled slices in order to return a tally/list of categories.
    1. Assembles a probability distribution for all the slices
    2. Going back, for each slice, looks for the superset with highest corpus prob
        Tallies the slice in the category labeled by its most prob superset
        *Consider also keeping track of what kinds of stuff ends up in the category
    If meth=voicing, does this with figured bass types; if meth=pcs, does this with ([PCs],bassSD)
    """
    from string import Template
    catCollapse = {}
    sliceCount = {}
    theImportantSlices = []
    skippedThings = 0
    corpusSize = 0
    binnedThings = 0
    supersettedThings = 0
    #Load the pickled slices that have not been bass-normalized into types
    theSlices = pickle.load( open ('1122MajModeSliceDictwSDB.pkl', 'rb') )
    for i, slicey in enumerate(theSlices):
        if slicey == ['start'] or slicey == ['end']:
            continue
        #keep count of the total number of slices before reduction
        corpusSize += 1
        if theSlices[i+1] == ['end']:
            continue
        #First, deal with singletons of bass motion 0
        if len(slicey['voicing_type']) == 1 and theSlices[i]['bassMIDI'] - theSlices[i+1]['bassMIDI'] == 0:
            skippedThings += 1
            continue
        #Next, only look at cases where |bass motion| > bassSize
        if abs(theSlices[i+1]['bassMIDI'] - theSlices[i]['bassMIDI']) > bassSize:
            secondSlicePCs = []
            theKey = theSlices[i+1]['key']
            theTonic = str(theKey).split(' ')[0]
            theKeyPC = pitch.Pitch(theTonic).pitchClass
            keyTransPCs = [(n - theKeyPC)%12 for n in theSlices[i+1]['pcset']]
            for n in keyTransPCs:
                secondSlicePCs.append(n)
            firstSlicePCs = []
            theKey = theSlices[i]['key']
            theTonic = str(theKey).split(' ')[0]
            theKeyPC = pitch.Pitch(theTonic).pitchClass
            keyTransPCs = [(n - theKeyPC)%12 for n in theSlices[i]['pcset']]
            for m in keyTransPCs:
                firstSlicePCs.append(m)
            #make sure second thing is superset of first thing
            continueIfZero = 0
            #even one note wrong means no!
            for n in firstSlicePCs:
                if n not in secondSlicePCs:
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
        theKey = slicey['key']
        theTonic = str(theKey).split(' ')[0]
        theKeyPC = pitch.Pitch(theTonic).pitchClass
        keyTransPCs = [(n - theKeyPC)%12 for n in slicey['pcset']]
        #rightChord = chord.Chord(sorted(keyTransPCs))
        slicey_label = (sorted(keyTransPCs),slicey['bassSD'])
        try:
            sliceCount[str(slicey_label)] += 1
        except KeyError:
            sliceCount[str(slicey_label)] = 1
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
        #First, if there's no bass leap, just go on and add it like a normal slice
        if abs(theImportantSlices[i+1]['bassMIDI'] - theImportantSlices[i]['bassMIDI']) <= bassSize:
            theKeyPC = pitch.Pitch(str(slicey['key']).split(' ')[0]).pitchClass
            keyTransPCs = [(n - theKeyPC)%12 for n in slicey['pcset']]
            theReallyImportantSlices.append((sorted(keyTransPCs),slicey['bassSD']))
            continue
        #Next, only look at cases where |bass motion| > bassSize
        if abs(theImportantSlices[i+1]['bassMIDI'] - theImportantSlices[i]['bassMIDI']) > bassSize:
            combinedSlices = []
            theKeyPC = pitch.Pitch(str(slicey['key']).split(' ')[0]).pitchClass
            keyTransPCs = [(n - theKeyPC)%12 for n in slicey['pcset']]
            for n in keyTransPCs:
                combinedSlices.append(n)
            theKeyPC = pitch.Pitch(str(theImportantSlices[i+1]['key']).split(' ')[0]).pitchClass
            nextkeyTransPCs = [(n - theKeyPC)%12 for n in theImportantSlices[i+1]['pcset']]
            for m in nextkeyTransPCs:
                if m in combinedSlices:
                    continue
                combinedSlices.append(m)
            sortedSlice = sorted(combinedSlices)
            #Pick whichever bass is literally lower in pitch, and use its SD for combo
            slicey_bass = slicey['bassMIDI']
            nextslice_bass = theImportantSlices[i+1]['bassMIDI']
            if slicey_bass <= nextslice_bass:
                bassSD = slicey['bassSD']
            if nextslice_bass < slicey_bass:
                bassSD = theImportantSlices[i+1]['bassSD']
            sortedSlice_type = (sortedSlice,bassSD)
            #If the combination never occurs, don't combine and move on
            try:
                testProb = sliceProbs[str(sortedSlice_type)]
            except KeyError:
                theKeyPC = pitch.Pitch(str(slicey['key']).split(' ')[0]).pitchClass
                keyTransPCs = [(n - theKeyPC)%12 for n in slicey['pcset']]
                theReallyImportantSlices.append((sorted(keyTransPCs),slicey['bassSD']))
                continue
            #Deal with singletons, which always have higher p
            #If both are singletons, move on:
            if len(slicey['pcset']) == 1 and len(theImportantSlices[i+1]['pcset']) == 1:
                continue
            #If the first is a singleton and second more probable than comb., move on
            elif len(slicey['pcset']) == 1 and len(theImportantSlices[i+1]['pcset']) > 1:
                if testProb < sliceProbs[str((sorted(nextkeyTransPCs),theImportantSlices[i+1]['bassSD']))]:
                    continue
            #If the second is a singleton and first more probable than comb., move on
            elif len(theImportantSlices[i+1]['pcset']) == 1 and len(slicey['pcset']) > 1:
                if testProb < sliceProbs[str((sorted(keyTransPCs),slicey['bassSD']))]:
                    continue
            #Otherwise, if p(comb) is less than either by themselves, move on
            elif testProb < sliceProbs[str((sorted(keyTransPCs),slicey['bassSD']))] or testProb < sliceProbs[str((sorted(nextkeyTransPCs),theImportantSlices[i+1]['bassSD']))]:
                continue
            #Once we rule out those cases, we know we want to combine.
            theReallyImportantSlices.append(sortedSlice_type)
            skipNext = 1
            binnedThings += 1
    #Tally up theReallyImportantSlices to get new sliceProbs
    #Now use sliceProbs to check the most common superset for each non-singleton slice
    sliceCount = {}
    for i, slicey in enumerate(theReallyImportantSlices):
        #if i > 10:
        #    break
        if slicey == ['start'] or slicey == ['end'] or i == len(theReallyImportantSlices) - 1:
            continue
        if len(slicey[0]) == 1:
            continue
        slicey_prob = sliceProbs[str(slicey)]
        bestSupersetProb = slicey_prob
        bestSuperset = slicey
        #Find superset entries in sliceProbs with higher prob
        for key, probvalue in sliceProbs.iteritems():
            if probvalue < bestSupersetProb:
                continue
            #something funny here... what exactly does iteritems() do?
            keything = key.split('], ')[0]
            keyparts = keything.strip('([')
            if len(keyparts) == 1:
                listofPCs = [int(n) for n in keyparts]
            else:
                pclist  = keyparts.split(', ')
                listofPCs = [int(n) for n in pclist]
            continueIfZero = 0
            #even one note wrong means no!  For now, allow NEW bass note?
            for n in slicey[0]:
                if n not in listofPCs:
                    continueIfZero += 1
                    break
            if continueIfZero == 0:
                supersettedThings += 1
                bestSuperset = key
                bestSupersetProb = probvalue
                break
            #MESSED THIS UP
        if bestSuperset != str(slicey):
            #print bestSuperset, slicey
            try:
                catCollapse[str(bestSuperset)][str(slicey)] += 1
            except KeyError:
                try:
                    catCollapse[str(bestSuperset)][str(slicey)] = 1
                except KeyError:
                    catCollapse[str(bestSuperset)] = {}
                    catCollapse[str(bestSuperset)][str(slicey)] = 1
        try:
            sliceCount[str((bestSuperset,bestSupersetProb))] += 1
        except KeyError:
            sliceCount[str((bestSuperset,bestSupersetProb))] = 1
    sorted_slicecount = sorted(sliceCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #export the probs as a csv file
    csvName = 'pcset superset tallies.csv'
    x = csv.writer(open(csvName, 'wb'))
    for pair in sorted_slicecount:
        x.writerow([pair[0], pair[1]]) 
    #print "supersetted things",supersettedThings
    #now put the bigramTally in some kind of csv table
    """
    cols = set()
    for row in catCollapse:
        for col in catCollapse[row]:
            cols.add(col)
    fieldnames = ['rowlabel'] + list(cols)
    #populate row labels
    for row in catCollapse:
        catCollapse[row]['rowlabel'] = row
    #write the CSV
    file = open('whatsincategories1122.csv', 'wb')
    #write the column headers
    #first, use plain CSV writer to write the field list
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    #now write the body of the table
    #use a different CSV writer object
    dw = csv.DictWriter(file, fieldnames)
    for row in catCollapse:
        dw.writerow(catCollapse[row])
    """
def pcSetFitni(fp,card):
    '''
    Inputs: raw cluster prototypes csv (n x 12), choice of card (up to 7, currently)
    Tallies up 
    '''
    openCSV = open(fp,'rb')
    allPro = csv.reader(openCSV)
    #turn all the prototypes into a list of 12-d floats
    ProList = []
    for row in allPro:
        ProList.append(row)
    ProList.pop(0)
    for row in ProList:
        row.pop(0)
        for i,thingy in enumerate(row):
            row[i] = float(thingy)
    #check all possible pcssets of cardinality 
    pcSetFitni = {}
    for i in range(12):
        for j in range(i,12):
            for k in range(j,12):
                for m in range(k,12):
                    for n in range(m,12):
                        for p in range(n,12):
                            for q in range(p,12):
                                pcset = set()
                                pcset.add(i)
                                pcset.add(j)
                                pcset.add(k)
                                pcset.add(m)
                                pcset.add(n)
                                pcset.add(p)
                                pcset.add(q)
                                #out of all pcset combinations, discard those of wrong card
                                if len(pcset) != card:
                                    continue
                                pcSetFitness = 0.0
                                maxSetFitness = -1000
                                for row in ProList:
                                    #print row
                                    pcSetSum = 0.0
                                    for pc in pcset:
                                        pcSetSum += (row[pc])
                                    pcSetSum -= (sum(row)-pcSetSum)
                                    #print pcSetSum
                                    #if pcSetSum > maxSetFitness:
                                        #maxSetFitness = pcSetSum
                                    pcSetFitness += pcSetSum
                                pcSetFitni[tuple(sorted(pcset))] = pcSetFitness
    sorted_pcS = sorted(pcSetFitni.iteritems(), key=operator.itemgetter(1), reverse=True)
    print sorted_pcS
    csvName = 'pcSetFitni'+str(card)+'.csv'
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    for row in sorted_pcS:
        lw.writerow(row)   
        
voicedPCSet([0,2,4,7])              
#tallyPCSets(4)
#pcSetFitni('C:/Users/Andrew/workspace/JazzHarmony/1200Laprototypes.csv', 4)