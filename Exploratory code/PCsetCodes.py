from music21 import *
from sets import *
import csv
import pickle
import numpy
import os
import operator

def getProbsFromFreqs(DictionaryOfTallies):
    totalSum = 0.0
    dictOfProbs = {}
    for key, freq in DictionaryOfTallies.iteritems():
        totalSum += float(freq)
    for key, freq in DictionaryOfTallies.iteritems():
        dictOfProbs[key] = float(freq)/totalSum
    return dictOfProbs

def whatsNAfter(voicing,dist,meth):
    """
    This takes a voicing (if meth=aj) or ([PCset],bassSD) (if meth=iq)
    tells you what comes next after 1, 2, ..., dist slices.
    Note that this doesn't do any binning or gluing or skipping or whatever
    """
    from string import Template
    sliceCount = {}
    #Load the pickled slices that have not been bass-normalized into types
    if meth == 'aj':
        theSlices = pickle.load( open ('quantizedChordSlicesWithBass1122.pkl', 'rb') )
    elif meth == 'iq':
        theSlices = pickle.load( open ('1122MajModeSliceDictwSDB.pkl', 'rb') )
    if meth == 'aj':
        for i, slicey in enumerate(theSlices):
            #if i > 10:
            #    break
            if slicey == 'start' or slicey == 'end' or theSlices[i + 1] == 'end':
                continue
            sorted_slicey = sorted(slicey)
            sliceyBass = sorted_slicey[0]
            #find out the voicing this slice is
            slicey_type = [n - sliceyBass for n in sorted_slicey]
            if slicey_type != voicing:
                continue
            #If it's a series of identical, repeated chords, only count the last one
            nextsorted_sliceytype = [n - sorted(theSlices[i+1])[0] for n in sorted(theSlices[i+1])]
            if nextsorted_sliceytype == voicing:
                continue
            j = 0
            while j < dist:
                #Get the next slice j away
                if theSlices[i + j + 1] == 'end':
                    nextSlice_label = 'end'
                    break
                #Reject it if it's too small
                if len(theSlices[i + j + 1]) > 3 or len(theSlices[i + j + 1]) == 3:
                    nextSlice = sorted(theSlices[i+1+j])
                    #remove octave duplications
                    for pitchy in nextSlice:
                        n = 1
                        while n < 8:
                            if pitchy - 12*n in nextSlice:
                                try:
                                    nextSlice.remove(pitchy)
                                except ValueError:
                                    break
                            n += 1  
                    #If diffToggle == 0, the slice is skipped
                    diffToggle = 1
                    #reject again if too small after octaves removed
                    if len(nextSlice) < 3:
                        diffToggle = 0
                    #Reject if it's the same
                    if nextSlice == sorted_slicey:
                        diffToggle = 0
                    else:
                        #Reject if it's a subset
                        setToggle1 = 0
                        setToggle2 = 0
                        for pitchy in nextSlice:
                            if pitchy not in sorted_slicey:
                                setToggle1 += 1
                        #Reject if it's a superset
                        for pitchy in sorted_slicey:
                            if pitchy not in nextSlice:
                                setToggle2 += 1
                        if setToggle1 == 0 or setToggle2 == 0:
                            diffToggle = 0
                    if diffToggle != 0:                          
                        #Find its bass and type.  For now, allow self-bass motions
                        nextSlice_bass = nextSlice[0]
                        nextSlice_type = [n - nextSlice_bass for n in nextSlice]
                        nextSlice_label = (nextSlice_bass - sliceyBass, nextSlice_type)
                        howFar = j + 1
                        try:
                            sliceCount[str(nextSlice_label)][str(howFar)] += 1
                        except KeyError:
                            try:
                                sliceCount[str(nextSlice_label)][str(howFar)] = 1
                            except KeyError:
                                sliceCount[str(nextSlice_label)] = {}
                                sliceCount[str(nextSlice_label)][str(howFar)] = 1
                j += 1
    if meth == 'iq':
        for i, slicey in enumerate(theSlices):
            #if i > 10:
            #    break
            if slicey == ['start'] or slicey == ['end'] or theSlices[i + 1] == ['end']:
                continue
            theKey = slicey['key']
            theTonic = str(theKey).split(' ')[0]
            theKeyPC = pitch.Pitch(theTonic).pitchClass
            keyTransPCs = [(n - theKeyPC)%12 for n in slicey['pcset']]
            rightChord = chord.Chord(sorted(keyTransPCs))
            slicey_label = (rightChord.pitchNames,slicey['bassSD'])
            if str(slicey_label) != voicing:
                continue
            #If it's a series of identical, repeated chords, only count the last one
            theKey = theSlices[i+1]['key']
            theTonic = str(theKey).split(' ')[0]
            theKeyPC = pitch.Pitch(theTonic).pitchClass
            keyTransPCs = [(n - theKeyPC)%12 for n in theSlices[i+1]['pcset']]
            rightChord = chord.Chord(sorted(keyTransPCs))
            nextsorted_sliceylabel = (rightChord.pitchNames,theSlices[i+1]['bassSD'])
            if nextsorted_sliceylabel == voicing:
                continue
            j = 0
            while j < dist:
                #Get the next slice j away
                if theSlices[i + j + 1] == ['end']:
                    nextSlice_label = ['end']
                    break
                #Reject it if it's too small
                if len(theSlices[i + j + 1]['pcset']) >= 3:
                    #Find its bass and type.  For now, allow self-bass motions
                    theKey = theSlices[i+j+1]['key']
                    theTonic = str(theKey).split(' ')[0]
                    theKeyPC = pitch.Pitch(theTonic).pitchClass
                    keyTransPCs = [(n - theKeyPC)%12 for n in theSlices[i+j+1]['pcset']]
                    rightChord = chord.Chord(sorted(keyTransPCs))
                    nextSlice_label = (rightChord.pitchNames, theSlices[i+j+1]['bassSD'])
                    howFar = j + 1
                    try:
                        sliceCount[str(nextSlice_label)][str(howFar)] += 1
                    except KeyError:
                        try:
                            sliceCount[str(nextSlice_label)][str(howFar)] = 1
                        except KeyError:
                            sliceCount[str(nextSlice_label)] = {}
                            sliceCount[str(nextSlice_label)][str(howFar)] = 1
                j += 1
    #now put the bigramTally in some kind of csv table
    cols = set()
    for row in sliceCount:
        for col in sliceCount[row]:
            cols.add(col)
    fieldnames = ['nth slice...'] + list(cols)
    #populate row labels
    for row in sliceCount:
        sliceCount[row]['nth slice...'] = row
    #write the CSV
    if meth == 'aj':
        fileName = Template('$voic vcg slices 1122.csv')
        csvName = fileName.substitute(voic = str(voicing))
    if meth == 'iq':
        fileName = Template('$voic pcSD slices 1122.csv')
        csvName = fileName.substitute(voic = str(voicing))
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in sliceCount:
        dw.writerow(sliceCount[row])

def binnedPCSDafter(voicing,bassSize,dist):
    """
    An impressive but useless code. Takes as input some ([PCset],bassSD) like "(['F', 'G', 'B'], 7)"
    It then:
    1. Bins via superset, looking for cases with big bass jumps st [i+1] is superset of [i]; skips [i]
        Also skips where singleton followed by bass motion 0
    2. Glues via probability, looking for cases where p([i+1]+[i]) > p([i+1)] and p([i]) (or either, if one is a singleton)
        *This is a fairly weak criterion; consider adding cases where p([i]) < p([i]+[i+1]) < p([i+1])
    3. Searches through the binned, glued slices for the requested chord
        Outputs a list of all the things that come n slices after it (for n<= dist)
    Problem: Destination chord categories (and cc in gen.) are TOO SMALL -- tons of them
    """
    from string import Template
    sliceCount = {}
    theImportantSlices = []
    skippedThings = 0
    corpusSize = 0
    binnedThings = 0
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
    #Now, code ported from whatsNafter to use the binned (PCset,bassSD) data to look for succession syntax
    sliceCount = {}
    for i, slicey in enumerate(theReallyImportantSlices):
        #if i > 10:
        #    break
        if slicey == ['start'] or slicey == ['end'] or i == len(theReallyImportantSlices) - 1:
            continue
        rightChord = chord.Chord(slicey[0])
        slicey_label = (rightChord.pitchNames,slicey[1])
        if str(slicey_label) != voicing:
            continue
        #If it's a series of identical, repeated chords, only count the last one
        rightChord = chord.Chord(theReallyImportantSlices[i+1][0])
        nextsorted_sliceylabel = (rightChord.pitchNames,theReallyImportantSlices[i+1][1])
        if str(nextsorted_sliceylabel) == voicing:
            continue
        j = 0
        while j < dist:
            #Get the next slice j away
            if theReallyImportantSlices[i + j + 1] == ['end']:
                nextSlice_label = ['end']
                break
            #Reject it if it's too small
            if len(theReallyImportantSlices[i + j + 1][0]) >= 3:
                #Find its bass and type.  For now, allow self-bass motions
                rightChord = chord.Chord(theReallyImportantSlices[i+j+1][0])
                nextSlice_label = (rightChord.pitchNames, theReallyImportantSlices[i+j+1][1])
                howFar = j + 1
                try:
                    sliceCount[str(nextSlice_label)][str(howFar)] += 1
                except KeyError:
                    try:
                        sliceCount[str(nextSlice_label)][str(howFar)] = 1
                    except KeyError:
                        sliceCount[str(nextSlice_label)] = {}
                        sliceCount[str(nextSlice_label)][str(howFar)] = 1
            j += 1
    #now put the bigramTally in some kind of csv table
    cols = set()
    for row in sliceCount:
        for col in sliceCount[row]:
            cols.add(col)
    fieldnames = ['nth slice...'] + list(cols)
    #populate row labels
    for row in sliceCount:
        sliceCount[row]['nth slice...'] = row
    #write the CSV
    fileName = Template('$voic pcSD binned 1122.csv')
    csvName = fileName.substitute(voic = str(voicing))
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in sliceCount:
        dw.writerow(sliceCount[row])
    print 'Binned things?', binnedThings
    
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
    thePickles = ['1122MajModeSliceDictwSDB.pkl','1122MinModeSliceDictwSDB.pkl']
    for eachPickle in thePickles:
        theSlices = pickle.load( open (eachPickle, 'rb') )
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
    csvName = 'pcset majmin superset tallies.csv'
    x = csv.writer(open(csvName, 'wb'))
    for pair in sorted_slicecount:
        x.writerow([pair[0], pair[1]]) 
    #print "supersetted things",supersettedThings
    #now put the bigramTally in some kind of csv table
    '''
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
    '''
        
def catFinder(bassSize):
    """
    A simpified version of categoryFinder which returns [sliceProbs,theReallyImportantSlices] for use in other codes.
    """
    from string import Template
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
    return [sliceProbs,theReallyImportantSlices]
        
def supersetBigrams(thingy,dist,mode):
    """
    OK, so we can find supersets and stuff.  Now to try and assemble bigrams using that info.
    1. Get list of binned, glued, pcset probs or whatever
    2. Go through slices looking for given pcset (*not its commons supersets?)
    3. At each OC, tally the next chord at dist based on its most common superset
        Does what it's supposed to do, but most stuff still has crazy low counts
    If mode == 'probs':
    4. Output the results normalized by log(p|oc) - log(p|c)
        Does what it's supposed to do, but results are stupid: vastly over-represents extremely rare chords
    """
    theStuff = catFinder(7)
    sliceProbs = theStuff[0]
    theSlices = theStuff[1]
    sliceCount = {}
    supersettedThings = 0
    for i, slicey in enumerate(theSlices):
        #if i > 10:
        #    break
        if slicey == ['start'] or slicey == ['end']:
            continue
        if str(slicey) != thingy:
            continue
        if str(theSlices[i+1]) == thingy:
            continue
        j = 0
        while j < dist:
            #Get the next slice j away
            if theSlices[i + j + 1] == ['end']:
                nextSlice_label = ['end']
                break
            nextSlice = theSlices[i + j + 1]
            #skip if it's a singleton?
            if len(nextSlice[0]) > 1:
                #otherwise, find its most probable superset
                slicey_prob = sliceProbs[str(nextSlice)]
                bestSupersetProb = slicey_prob
                bestSuperset = nextSlice
                #Find superset entries in sliceProbs with higher prob
                for key, probvalue in sliceProbs.iteritems():
                    if probvalue < bestSupersetProb:
                        continue
                    keything = key.split('], ')[0]
                    keyparts = keything.strip('([')
                    if len(keyparts) == 1:
                        listofPCs = [int(n) for n in keyparts]
                    else:
                        pclist  = keyparts.split(', ')
                        listofPCs = [int(n) for n in pclist]
                    continueIfZero = 0
                    #even one note wrong means no!  For now, allow NEW bass note?
                    for n in nextSlice[0]:
                        if n not in listofPCs:
                            continueIfZero += 1
                            break
                    if continueIfZero == 0:
                        supersettedThings += 1
                        bestSuperset = key
                        bestSupersetProb = probvalue
                nextSlice_label = bestSuperset
                howFar = j + 1
                try:
                    sliceCount[str(howFar)][str(nextSlice_label)] += 1
                except KeyError:
                    try:
                        sliceCount[str(howFar)][str(nextSlice_label)] = 1
                    except KeyError:
                        sliceCount[str(howFar)] = {}
                        sliceCount[str(howFar)][str(nextSlice_label)] = 1
            j += 1
    print sliceCount
    if mode == 'probs':
        #now make probs from those tallies
        for slicedist in sliceCount:
            theProbs = getProbsFromFreqs(sliceCount[slicedist])
            sliceCount[slicedist] = theProbs
        print sliceCount
        #now normalize by computing log(p|oc) - log(p|c)
        for slicedist in sliceCount:
            for destchord in sliceCount[slicedist]:
                localprob = sliceCount[slicedist][destchord]
                sliceCount[slicedist][destchord] = numpy.log10(localprob) - numpy.log10(sliceProbs[destchord])
        print sliceCount
    #now put the bigramTally in some kind of csv table
    cols = set()
    for row in sliceCount:
        for col in sliceCount[row]:
            cols.add(col)
    fieldnames = ['nth slice...'] + list(cols)
    #populate row labels
    for row in sliceCount:
        sliceCount[row]['nth slice...'] = row
    #write the CSV
    if mode == 'probs':
        fileName = Template('$voic pcSD superset bi normed 1122.csv')
    else:
        fileName = Template('$voic pcSD superset bi 1122.csv')
    csvName = fileName.substitute(voic = str(thingy))
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in sliceCount:
        dw.writerow(sliceCount[row])

#supersetBigrams("([0, 2, 5, 9], 2)",25,'c')
categoryFinder(12)    
#binnedPCSDafter("(['F', 'G', 'B'], 7)",7,25)   