from music21 import *
from sets import *
import csv
import pickle
import numpy
import os
import operator
import collections

"""
Goal: Find a low-overhead way to visualize progression data about particular (high-frequency) things
1. Revised whatsNAfter shows behavior of particular voicings. Tends to pick out self-motion almost entirely.
2. Need a way of finding the next DIFFERENT thing that happens after particular voicing.  Can I adapt threader for this?
Already rejects things of size < 3, or which are identical to start slice.
Also rejects things which are supersets and subsets of start slice.
Also removes octave duplications (keeps the lowest)
For now, allows self-bass motions.  Run again after ruling those out to see difference.
"""

def getProbsFromFreqs(DictionaryOfTallies):
    totalSum = 0.0
    dictOfProbs = {}
    for key, freq in DictionaryOfTallies.iteritems():
        totalSum += float(freq)
    for key, freq in DictionaryOfTallies.iteritems():
        dictOfProbs[key] = float(freq)/totalSum
    return dictOfProbs

def csvTransposer(f,d):
    infile = open(f)
    reader = csv.reader(infile)
    cols = []
    for row in reader:
        cols.append(row)
    outfile = open(d,'wb')
    writer = csv.writer(outfile)
    for i in range(len(max(cols, key=len))):
        writer.writerow([(c[i] if i<len(c) else '') for c in cols])        

def whatsNstepsAfter(voicing,dist,bassMode,octaveMode):
    """
    This takes a voicing and tells you what comes next after 1, 2, ..., non-identical steps.
    Starts with a slice, and:
    removes octave dups (leaving lowest) if octaveMode = 'no octaves'
    iterates through until it finds the next slice such that length > 3
    not subset or superset, and bass moves (if bassMode = 'bass change')
    That's the next threaded slice; it does this to thread dist
    """
    from string import Template
    sliceCount = {}
    #Load the pickled slices that have not been bass-normalized into types
    theSlices = pickle.load( open ('quantizedChordSlicesWithBass1122.pkl', 'rb') )
    for i, slicey in enumerate(theSlices):
        #don't count what happens to start and end tokens
        if slicey == 'start' or slicey == 'end' or theSlices[i + 1] == 'end':
            continue
        else:
            sorted_slicey = sorted(slicey)
            sliceyBass = sorted_slicey[0]
            #find out the voicing this slice is
            slicey_type = [n - sliceyBass for n in sorted_slicey]
            #select only what we're interested in
            if slicey_type != voicing:
                continue
            startAt = 0
            k = 0
            while k < dist:
                theNextSlice = 0
                j = 0
                while theNextSlice == 0:
                    testSlice = theSlices[i + startAt + j + 1]
                    #watch out for end tokens
                    if testSlice == 'end':
                        theNextSlice = 'end'
                        continue
                    theNextSlice = sorted(testSlice)
                    #if selfbass not requested, reject
                    if bassMode == 'bass change':
                        if theNextSlice[0] == sliceyBass:
                            theNextSlice = 0
                            j += 1
                            continue
                    #remove octave duplications if octaveMode requests it
                    if octaveMode == 'no octaves':
                        for pitchy in theNextSlice:
                            n = 1
                            while n < 8:
                                if pitchy - 12*n in theNextSlice:
                                    try:
                                        theNextSlice.remove(pitchy)
                                    except ValueError:
                                        pass
                                n += 1
                    #Reject it if it's too small
                    if theNextSlice != 0:
                        if len(theNextSlice) < 3:
                            theNextSlice = 0
                            j += 1
                            continue
                    #else:
                    #If diffToggle == 0, the slice is skipped
                    diffToggle = 1
                    #Reject if it's a subset
                    setToggle1 = 0
                    setToggle2 = 0
                    for pitchy in theNextSlice:
                        if pitchy not in sorted_slicey:
                            setToggle1 += 1
                    #Reject if it's a superset
                    for pitchy in sorted_slicey:
                        if pitchy not in theNextSlice:
                            setToggle2 += 1
                    #Reject if pcsets identical (re-registration)
                    pcset = set()
                    for pitchy in sorted_slicey:
                        pcset.add(pitchy % 12)
                    nextpcset = set()
                    for pitchy in theNextSlice:
                        nextpcset.add(pitchy % 12)
                    if pcset == nextpcset:
                        diffToggle = 0
                    #if whatever bullshit disqualified the slice, move on                        
                    if setToggle1 == 0 or setToggle2 == 0:
                        diffToggle = 0
                    if diffToggle == 0: 
                        theNextSlice = 0
                    j += 1
                startAt += j
                #watch out for end tokens
                if theNextSlice == 'end':
                    nextSlice_label = 'end'
                else:
                    #Find its bass and type.  For now, allow self-bass motions
                    nextSlice_bass = theNextSlice[0]
                    nextSlice_type = [n - nextSlice_bass for n in theNextSlice]
                    nextSlice_label = (nextSlice_bass - sliceyBass, nextSlice_type)
                howFar = k + 1
                try:
                    sliceCount[str(nextSlice_label)][str(howFar)] += 1
                except KeyError:
                    try:
                        sliceCount[str(nextSlice_label)][str(howFar)] = 1
                    except KeyError:
                        sliceCount[str(nextSlice_label)] = {}
                        sliceCount[str(nextSlice_label)][str(howFar)] = 1
                k += 1     
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
    fileName = Template('$voic stepsquant thread 1122.csv')
    if bassMode == 'bass change' and octaveMode == 'no octaves':
        csvName = fileName.substitute(voic = str(voicing)+'bc n8')
    elif bassMode == 'bass change' and octaveMode != 'no octaves':
        csvName = fileName.substitute(voic = str(voicing)+'bc')
    elif bassMode != 'bass change' and octaveMode == 'no octaves':
        csvName = fileName.substitute(voic = str(voicing)+'n8')
    else:
        csvName = fileName.substitute(voic = str(voicing))
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in sliceCount:
        dw.writerow(sliceCount[row])                                     
                    
    
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
        
def supersetsOf(voicing):
    '''
    Given a voicing, this tallies up all the supersets of it
    Note: Does NOT assume pc-equivalence
    The idea here is to see what most commonly-deployed voicings contain whatever
    '''
    from string import Template
    sliceCount = {}
    totalChords = 0
    #Load the pickled slices that have not been bass-normalized into types
    theSlices = pickle.load( open ('pickles/quantizedChordDictSlices1015.pkl', 'rb') )
    for i, slicey in enumerate(theSlices):
        #Don't check start/end tokens
        if slicey == ['start'] or slicey == ['end']:
            continue
        #pull the sorted midi representation of the slice
        sliceMidi = slicey['midi']
        for n in sliceMidi:
            #Build the voicing on each pitch of the slice midi to check for a match
            testVoicing = [n + p for p in voicing]
            #This is just a toggle
            zeroIsMatch = 0
            for m in testVoicing:
                #If any pitch from the built voicing isn't in the slice, build a different one
                if m not in sliceMidi:
                    zeroIsMatch += 1
                    break
            #If all the pitches of the built voicing ARE in the slice, it's a superset slice
            if zeroIsMatch == 0:
                totalChords += 1
                try:
                    sliceCount[str(sliceMidi)] += 1
                except KeyError:
                    sliceCount[str(sliceMidi)] = 1
                break
    sorted_sliceCount = sorted(sliceCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print 'Total number of slices:', totalChords
    #print 'All the slices!',sorted_chordCount
    #export the tally as a csv file
    fileName = Template('supersets of $voic 1015.csv')
    csvName = fileName.substitute(voic = str(voicing))
    x = csv.writer(open(csvName, 'wb'))
    for pair in sorted_sliceCount:
        x.writerow([pair[0], pair[1]])    
        
    
def subsetsOf(voicing,siz):
    #Write a thing that checks for, say, #>2 subsets of voicing
    '''
    Given a voicing, this tallies up all the subsets of size 'siz' or greater
    Note: Does NOT assume pc-equivalence
    The idea here is to see what notes end up dropped most?  Make sense?
    '''
    from string import Template
    sliceCount = {}
    totalChords = 0
    #Load the pickled slices that have not been bass-normalized into types
    theSlices = pickle.load( open ('quantizedChordDictSlices1122.pkl', 'rb') )
    for i, slicey in enumerate(theSlices):
        #Don't check start/end tokens
        if slicey == ['start'] or slicey == ['end']:
            continue
        #pull the sorted midi representation of the slice
        sliceMidi = slicey['midi']
        #reject if too small
        if len(sliceMidi) < siz:
            continue
        #This is just a toggle
        zeroIsMatch = 0
        #Try all the transpositions of slice to pitches of voicing
        for n in voicing:
            #Transpose the slice to start at each pitch of voicing
            testSubs = [p + n for p in sliceMidi]
            for m in testSubs:
                #At that trans., if any pitch from slice isn't in voicing, move on
                if m not in voicing:
                    zeroIsMatch += 1
                    break
            if zeroIsMatch == 0:
                totalChords += 1
                try:
                    sliceCount[str(sliceMidi)] += 1
                except KeyError:
                    sliceCount[str(sliceMidi)] = 1
                break
    sorted_sliceCount = sorted(sliceCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print 'Total number of slices:', totalChords
    #print 'All the slices!',sorted_chordCount
    #export the tally as a csv file
    fileName = Template('subsets of $voic 1122.csv')
    csvName = fileName.substitute(voic = str(voicing))
    x = csv.writer(open(csvName, 'wb'))
    for pair in sorted_sliceCount:
        x.writerow([pair[0], pair[1]])  
            

def whereIs(voicing):
    from string import Template
    sliceCount = {}
    totalChords = 0
    #Load the pickled slices that have not been bass-normalized into types
    theSlices = pickle.load( open ('quantizedChordDictSlices1122.pkl', 'rb') )
    for i, slicey in enumerate(theSlices):
        if slicey == ['start'] or slicey == ['end']:
            continue
        if slicey['midi'] == voicing:
            soloName = slicey['solo']
            totalChords += 1
            try:
                sliceCount[soloName] += 1
            except KeyError:
                sliceCount[soloName] = 1
    sorted_sliceCount = sorted(sliceCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print 'Total number of slices:', totalChords
    #print 'All the slices!',sorted_chordCount
    #export the tally as a csv file
    fileName = Template('$voic solos 1122.csv')
    csvName = fileName.substitute(voic = str(voicing))
    x = csv.writer(open(csvName, 'wb'))
    for pair in sorted_sliceCount:
        x.writerow([pair[0], pair[1]])
        
def binthethings(bassSize):
    """
    1. If you have large bass motion and some superset relation, get rid of the first
    2. Also, if a singleton is followed by bass motion zero: get rid of it
    3. Tally up most common voicings
    4. Now, any time there's a big bass motion, consider gluing together if:
            One is a singleton and the other is less common than the sum of the parts (DONE)
            No singleton and the resulting thing is more common than either of the parts (DONE)
            ALSO MAYBE
            get a list of all the gluings such that p(p) < p(p v q) < p (q)
            see if those are cases where we want to glue or not
    """
    from string import Template
    sliceCount = {}
    theImportantSlices = []
    skippedThings = 0
    corpusSize = 0
    binnedThings = 0
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
            #If the combination never occurs, don't combine and move on
            try:
                testProb = sliceProbs[str(sortedSlice_type)]
            except KeyError:
                theReallyImportantSlices.append(slicey)
                continue
            #Deal with singletons, which always have higher p
            #If both are singletons, move on:
            if len(slicey['midi']) == 1 and len(theImportantSlices[i+1]['midi']) == 1:
                continue
            #If the first is a singleton and second more probable than comb., move on
            elif len(slicey['midi']) == 1 and len(theImportantSlices[i+1]['midi']) > 1:
                if testProb < sliceProbs[str(theImportantSlices[i+1]['midi'])]:
                    continue
            #If the second is a singleton and first more probable than comb., move on
            elif len(theImportantSlices[i+1]['midi']) == 1 and len(slicey['midi']) > 1:
                if testProb < sliceProbs[str(slicey['midi'])]:
                    continue
            #Otherwise, if p(comb) is less than either by themselves, move on
            elif testProb < sliceProbs[(str(slicey['midi']))] or testProb < sliceProbs[str(theImportantSlices[i+1]['midi'])]:
                continue
            #Once we rule out those cases, we know we want to combine.
            combinedSlice = {}
            combinedSlice['bass'] = sortedSlice[0]
            combinedSlice['midi'] = sortedSlice_type
            theReallyImportantSlices.append(combinedSlice)
            skipNext = 1
            binnedThings += 1
    #Now, from the binned/REALLY important ones, find voicing probs
    binnedCount = {}
    for slicey in theReallyImportantSlices:
        try:
            binnedCount[str(slicey['midi'])] += 1
        except KeyError:
            binnedCount[str(slicey['midi'])] = 1
    binnedProbs = getProbsFromFreqs(binnedCount)
    sorted_binnedProbs = sorted(binnedProbs.iteritems(), key=operator.itemgetter(1), reverse=True)
    #export the probs as a csv file
    fileName = Template('$voic motion-collapsed binned probs.csv')
    csvName = fileName.substitute(voic = str(bassSize))
    x = csv.writer(open(csvName, 'wb'))
    for pair in sorted_binnedProbs:
        x.writerow([pair[0], pair[1]])   
    print 'Skipped this many:',skippedThings
    print 'Corpus has this many slices in general:',corpusSize
    print 'These many were binned',binnedThings
    print 'The collapsed voicing probs are ready!'               
    
def tallyVoicings(cmin):
    allChords = csv.reader(open('50 smushed chords.csv','rb'))
    allChordsList = []
    voicingTally = collections.Counter()
    for row in allChords:
        allChordsList.append(row)
    for v in allChordsList:
        dab = []
        for n in v:
            dab.append(int(n))
        if len(dab) < cmin:
            continue
        sorted_dab = sorted(dab)
        dab = [x - sorted_dab[0] for x in sorted_dab]
        voicingTally[str(dab)] += 1
    sorted_vcgTally = sorted(voicingTally.iteritems(), key=operator.itemgetter(1), reverse=True)
    y = csv.writer(open('50ms smushed vcgTally.csv','wb'))
    for pair in sorted_vcgTally:
        y.writerow([pair[0],pair[1]])
    
            
tallyVoicings(3)
#binthethings(7)
#whatsNAfter([0,3,5,10],25,'aj')            
#subsetsOf([0,4,7,10,16],3)
#supersetsOf([0,1,5,8])
#whereIs([0,8,13,19])
#whatsNAfter([0,4,10,17],50)#,'something','no octaves')
#whatsNstepsAfter([0,10,15],50,'something','no octaves')