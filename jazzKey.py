from __future__ import absolute_import
from __future__ import print_function
import midi
#import music21
from imghdr import what
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

#path for the edited jazz midi files
#path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIsandbox/'
#path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIQuantized/'
#listing = os.listdir(path)

def getProbsFromFreqs(DictionaryOfTallies):
    totalSum = 0.0
    dictOfProbs = {}
    for key, freq in six.iteritems(DictionaryOfTallies):
        totalSum += float(freq)
    for key, freq in six.iteritems(DictionaryOfTallies):
        dictOfProbs[key] = float(freq)/totalSum
    return dictOfProbs
    
def chordSlicesWithKey():
    """
    This is basically just a fancy slicer/pickle producer
    Starts from whatever collection of MIDI files you want
    1. Finds the best key guess for each individual midi file (1 per file!)
        *Consider changing key-finding window size
    2. For each slice, a dict is created with all its info (see below for dict keys)
    3. Slice dicts are stored separately for major and minor mode stuff
    Outputs two pickles and two voicing type tallies (major and minor)
    """
    totalChordsMajor = 0
    totalChordsMinor = 0
    chordCountMajor = {}
    chordCountMinor = {}
    theSlicesMajor = list()
    theSlicesMinor = list()
    problemFiles = []
    for f in listing:
        address = path + f
        print('current file:',address)
        try:
            oneSolo = converter.parse(address)
        except:
            problemFiles.append(f)
            print('Problem with',f)
            pass
        else:
            theKey = analysis.discrete.analyzeStream(oneSolo, 'bellman')
            theTonic = str(theKey).split(' ')[0]
            theMode = str(theKey).split(' ')[1]
            theKeyPC = pitch.Pitch(theTonic).pitchClass
            #print 'current key:', theKey, theKeyPC, theMode
            theSoloChords = oneSolo.chordify().flat.getElementsByClass('Chord')
            if theMode == 'major':
                theSlices = theSlicesMajor
                chordCount = chordCountMajor
                totalChords = totalChordsMajor
            elif theMode == 'minor':
                theSlices = theSlicesMinor
                chordCount = chordCountMinor
                totalChords = totalChordsMinor
            else:
                print('WHAT MODE??', theKey, address)
                continue
            startToken = ['start']
            theSlices.append(startToken)
            for someChord in theSoloChords:
                midiListing = [p.midi for p in someChord.pitches]
                #print 'midi listing:', midiListing
                bassNoteMidi = someChord.bass().midi
                bassNotePC = someChord.bass().pitchClass
                bassNoteSD = (bassNotePC - theKeyPC)%12
                #print 'bass note:', bassNoteMidi
                distAboveBass = [n - bassNoteMidi for n in midiListing]
                #print "intervals above bass:", distAboveBass
                sorted_distAboveBass = sorted(distAboveBass)
                thisSlice = {}
                thisSlice['voicing_type'] = sorted_distAboveBass
                thisSlice['bassMIDI'] = bassNoteMidi
                thisSlice['bassSD'] = bassNoteSD
                thisSlice['key'] = str(theKey)
                thisSlice['normalform'] = someChord.normalForm
                thisSlice['solo'] = f
                thisSlice['pcset'] = someChord.orderedPitchClasses
                transPCs = chord.Chord([n - theKeyPC for n in someChord.orderedPitchClasses])
                thisSlice['transpc'] = transPCs.pitchNames
                theSlices.append(thisSlice)
                try:
                    chordCount[str((someChord.orderedPitchClasses,bassNoteSD))] += 1
                except KeyError:
                    chordCount[str((someChord.orderedPitchClasses,bassNoteSD))] = 1
                totalChords += 1
            endToken = ['end']
            theSlices.append(endToken)
            if theMode == 'major':
                theSlicesMajor = theSlices
                chordCountMajor = chordCount
                totalChordsMajor = totalChords
            elif theMode == 'minor':
                theSlicesMinor = theSlices
                chordCountMinor = chordCount
                totalChordsMinor = totalChords
    #pickle the slices
    fpPickleMaj = '1122MajModeSliceDictwSDB.pkl'
    #fpPickleMaj = 'combinedMajSliceDictwSDB.pkl'
    fpPickleMin = '1122MinModeSliceDictwSDB.pkl'
   # fpPickleMin = 'combinedMinSliceDictwSDB.pkl'
    pickle.dump(theSlicesMajor, open(fpPickleMaj, "wb"))
    pickle.dump(theSlicesMinor, open(fpPickleMin, "wb"))
    #tally up the frequencies for each chord
    sorted_chordCountMaj = sorted(six.iteritems(chordCountMajor), key=operator.itemgetter(1), reverse=True)
    sorted_chordCountMin = sorted(six.iteritems(chordCountMinor), key=operator.itemgetter(1), reverse=True)
    print('Total number of Major slices:', totalChordsMajor)
    print('Total number of Minor slices:', totalChordsMinor)
    #export the tally as a csv file
    csvNameMaj = '1122MajModepcSTallywSDB.csv'
    #csvNameMaj = 'combinedMajorChordTallywSDB.csv'
    csvNameMin = '1122MinModepcSTallywSDB.csv'
    #csvNameMin = 'combinedMinorChordTallywSDB.csv'
    xmaj = csv.writer(open(csvNameMaj, 'w'))
    for pair in sorted_chordCountMaj:
        xmaj.writerow([pair[0],pair[1]])
    x = csv.writer(open(csvNameMin, 'w'))
    for pair in sorted_chordCountMin:
        x.writerow([pair[0], pair[1]])
    print('problem files are:', problemFiles)
    
def keyGetter():
    #look through the already-processed pickles to assemble a list of the keys for each solo
    #make listofKeys = [filename,int(tonic)] so we can subtract tonic out in midi later
    listofKeys = []
    theMajPickle = '1122MajModeSliceDictwSDB.pkl'
    theSlices = pickle.load(open(theMajPickle, 'r'))
    for i,slice in enumerate(theSlices):
        if slice != ['start']:
            continue
        theTonic = theSlices[i+1]['key'].split(' ')[0]
        theKeyPC = pitch.Pitch(theTonic).pitchClass
        theSolo = theSlices[i+1]['solo'].replace('_quant','')
        theOrigSolo = theSolo.replace('quant','')
        if '_q' in theOrigSolo:
            theOrigSolo = theSolo.replace('_q','')
        #if 'q' in theOrigSolo:
            #theOrigSolo = theSolo.replace('q','')
        listofKeys.append([theOrigSolo,theKeyPC])
    theMinPickle = '1122MinModeSliceDictwSDB.pkl'
    theSlices = pickle.load( open (theMinPickle, 'r') )
    for i, slice in enumerate(theSlices):
        if slice != ['start']:
            continue
        theTonic = theSlices[i+1]['key'].split(' ')[0]
        theKeyPC = pitch.Pitch(theTonic).pitchClass
        theSolo = theSlices[i+1]['solo'].replace('_quant','')
        theOrigSolo = theSolo.replace('quant','')
        if '_q' in theOrigSolo:
            theOrigSolo = theSolo.replace('_q','')
        #if 'q' in theOrigSolo:
            #theOrigSolo = theSolo.replace('q','')
        listofKeys.append([theOrigSolo,str((int(theKeyPC)+3)%12)])
    #fieldnames = ['ms window end','weighted MIDI','ordered PCs','file']
    csvName = 'solokeyoffsets_a_C.csv'
    file = open(csvName, 'w')
    lw = csv.writer(file)
    #lw.writerow(fieldnames)
    for row in listofKeys:
        lw.writerow(row)
    print(listofKeys)
    
def headKeys():
    """
    This makes a key-finding guess for the heads in MIDIheads.  Outputs a csv for it all.
    Should still hand check all of these!  I'll ultimately use them to make jazzy key profile vectors.
    """
    path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIheads_q/'
    listing = os.listdir(path)
    problemFiles = []
    headAndKey = []
    for f in listing:
        address = path + f
        print('current file:',address)
        try:
            oneSolo = converter.parse(address)
        except:
            problemFiles.append(f)
            print('Problem with',f)
            continue
        else:
            theKey = analysis.discrete.analyzeStream(oneSolo, 'bellman')
            theTonic = str(theKey).split(' ')[0]
            theMode = str(theKey).split(' ')[1]
            theKeyPC = pitch.Pitch(theTonic).pitchClass
            headAndKey.append([f,theKey])
    print("Any problems:",problemFiles)
    print(headAndKey)
    #write the CSV
    file = open('jazz_headKeys_1122.csv', 'w')
    lw = csv.writer(file)
    for row in headAndKey:
        lw.writerow(row)
        
def keyProfiles():
    """
    Inputs the head .mid files and the list of identified keys
    Transposes each track relative to ``tonic" (defensible??)
    Outputs weighted pc vector for each head
    """
    fp = open('jazz_headKeys_1122.csv','r')
    keyDict = {}
    dta = csv.reader(fp)
    for row in dta:
        keyDict[row[0]] = row[1]
    #print keyDict
    path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIheads_q/'
    listing = os.listdir(path)
    problemFiles = []
    headpcVec = []
    for f in listing:
        address = path + f
        print('current file:',address)
        try:
            oneSolo = converter.parse(address)
        except:
            problemFiles.append(f)
            print('Problem with',f)
            continue
        theKey = keyDict[f]
        theTonic = str(theKey).split(' ')[0]
        #theMode = str(theKey).split(' ')[1]
        theKeyPC = pitch.Pitch(theTonic).pitchClass
        pcVec = []
        for j in range(12):
            pcVec.append(0.0)
        theSoloChords = oneSolo.flat.getElementsByClass(note.Note)
        for k in range(len(theSoloChords.secondsMap)):
            transPC = (theSoloChords.secondsMap[k]['element'].pitch.pitchClass - theKeyPC)%12
            pcVec[transPC] += theSoloChords.secondsMap[k]['durationSeconds']
        headpcVec.append([f,pcVec])
    #output list of pcVecs as csv
    file = open('jazz_head_transpcVecs.csv', 'w')
    lw = csv.writer(file)
    for row in headpcVec:
        lw.writerow(row)

"""
1. Iterate through each solo
2. For each track, iterate a series of sliding windows through
   For each window, tally up the durationally-weighted pcvector for the window
   For each window size, calculate average perplexity
   CALLS: entrop(solo=filename), midiTimeWindows
3. From average perplexity vs. window size distribution, choose window size such that perplexity ~8.1
   This is chosen to roughly match existing scale/key profiles (esp. b-b)
4. Also from avg perp vs. window size dist, choose pane size such that perplexity first starts to rise
   Trigger: increase by 30% over perplexity of smallest window size (consider better justification?)
5. Using window and pane size, slide through track for local keyfinding
   CALLS: jazzKeyFinder(filepath)
6. Re-run midiTimeWindows, transposing each pane by cross-referencing local key list/dict
OUTPUT: csv of durationally-weighted pcVecs for locally-transposed windows
"""
path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIunquant/'
listing = os.listdir(path)
def midiTimeWindows(windowWidth,incUnit,solos=all):
    #windowWidth is obvious; incUnit how large the window slide step is
    #numTunes = 0
    #numShortTracks = 0
    #numTracks = 0
    #if solos != all:
        #listing = [solos]
    #we'll make a list: [millisecs at end of window, music21 chord, set of midi numbers,  pcs in order, file name]
    msandmidi = []
    #Load the pickled slices that have not been bass-normalized into types
    for n, testFile in enumerate(listing):
        if solos != all:
            if testFile != solos:
                continue
        #print path + testFile
        #if n > 50:
            #continue
            #break
        #numTunes += 1
        #for use with import midi
        pattern = midi.read_midifile(path + testFile)
        #this line makes each tick count cumulative
        pattern.make_ticks_abs()
        #print pattern.resolution, testFile
        #print len(pattern)
        for i,track in enumerate(pattern):
            #numTracks += 1
            if len(track) < 50:
                #numShortTracks += 1
                continue
            #how many tempo events?
            tempEvents = 0
            noteEvents = 0
            for thing in track:
                #print thing
                if thing.__class__ == midi.events.NoteOnEvent:
                    noteEvents += 1
                if thing.__class__ == midi.events.SetTempoEvent:
                    microspt = thing.get_mpqn() / pattern.resolution
                    #print microspt
                    tempEvents +=1
            if noteEvents == 0:
                #numShortTracks += 1
                continue
            if tempEvents == 0:
                microspt = 500000 / pattern.resolution
            if tempEvents > 1:
                print('hey, extra tempo event?')
                break
            #windowWidth = 100 #number of milliseconds wide each bin will be
            windows = []
            #Generate a window starting at each incUnit until last window exceeds track end
            startTime = 0
            #print track[-1]
            while startTime + windowWidth < track[-1].tick* microspt/1000:
                windows.append(collections.Counter())
                startTime += incUnit
            for m, thing in enumerate(track):
                #Now put each event into all the right windows
                #if m > 50:
                    #break
                absTicks = thing.tick * microspt/1000
                if thing.__class__ == midi.events.NoteOnEvent and thing.get_velocity() != 0:
                    #figure out how long it is by looking for off event
                    for s in range(m,len(track)):
                        if track[s].__class__ == midi.events.NoteOnEvent and track[s].get_velocity() == 0 and track[s].get_pitch() == thing.get_pitch():
                            endTick = track[s].tick* microspt/1000
                            diffTicks = endTick - absTicks
                            break
                        if track[s].__class__ == midi.events.NoteOffEvent and track[s].get_pitch() == thing.get_pitch():
                            endTick = track[s].tick* microspt/1000
                            diffTicks = endTick - absTicks
                            break
                        if s == len(track):
                            print('No note end!',testFile)
                    for j in range(len(windows)):
                        #weight considering four cases.  First, if the note off starts and ends inside the first window
                        if j*incUnit < absTicks < j*incUnit + windowWidth:
                            if endTick < j*incUnit + windowWidth:
                                windows[j][thing.get_pitch()] += int(round(diffTicks))
                            #next, if it starts in one and stretches to some future window
                            if endTick > j*incUnit + windowWidth:
                                windows[j][thing.get_pitch()] += int(round(j*incUnit + windowWidth - absTicks))
                        if j*incUnit > absTicks:
                            #if it started in some past window and ends in some future one
                            if endTick > j*incUnit + windowWidth:
                                windows[j][thing.get_pitch()] += windowWidth
                            #and last: if it started in some past window and ends in this one
                            if j*incUnit < endTick < j*incUnit + windowWidth:
                                windows[j][thing.get_pitch()] += int(round(endTick - j*incUnit))
                        #Once the note has ended, stop looking for places to stick it
                        if j*incUnit > endTick:
                            break
            for j in range(len(windows)):
                '''
                if sum(windows[j].values()) == 0:#skip the empty windows
                    continue
                '''
                msandmidi.append([(j)*incUnit,windows[j]])
    #print msandmidi
    '''
    #package up a csv
    #fieldnames = ['ms window end','weighted MIDI','ordered PCs','file']
    fileName = Template('$siz $inc ms inc 1122.csv')
    csvName = fileName.substitute(siz = str(windowWidth), inc = str(incUnit))
    file = open(csvName, 'w')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    for row in msandmidi:
        lw.writerow(row)
    #pickle that shit
    if transpose==True:
        fpPickle = Template('$win ms pcCount trans.pkl')
    elif transpose != True:
        fpPickle = Template('$win ms midcount overlap.pkl')
    pickleName = fpPickle.substitute(win = windowWidth)
    pickle.dump(msandmidi, open(pickleName, "wb"))
    '''
    return msandmidi

def perplexity(solo=all):
    #go from 50ms to 60*1000 ms by doubling
    windowSize = 12.5
    PerpAtSize = []
    while windowSize < 60000:
        windowSize = windowSize*2
        #print windowSize
        if windowSize <= 1000:
            incUnit = 25
        elif 1000 < windowSize < 10000:
            incUnit = 250
        else:
            incUnit = 1000
        if solo != all:
            msandmidi = midiTimeWindows(windowSize, incUnit, solos=solo)
        else:
            msandmidi = midiTimeWindows(windowSize,incUnit)
        perplexities = []
        for i, row in enumerate(msandmidi):
            #if i == 0:
                #continue
            if sum(row[1].values()) == 0:#skip the empty windows
                continue
            pcVector = []
            for j in range(12):
                pcVector.append(0.01)
            for mid, counts in six.iteritems(row[1]):
                pcVector[mid%12] += counts
            perplexities.append(scipy.exp2(scipy.stats.entropy(pcVector,base=2)))
            #print windowSize,pcVector, entropies[-1]
        PerpAtSize.append([windowSize,scipy.average(perplexities)])
        #now write the body of the table
    '''
    if solo != all:
        fileName = Template('$sol overlap window pc avg perp.csv')
        csvName =fileName.substitute(sol = solo.split('.')[0])
    else:
        csvName ='overlap window pc avg entropy.csv'
    file = open(csvName, 'w')
    lw = csv.writer(file)
    for row in PerpAtSize:
        lw.writerow(row)
    '''
    return PerpAtSize

def jazzKeyFinder(fp,windowSize,paneSize):
    #an empty list for the small, final key slices
    panesAndKeys = []
    #fn = path + fp
    #the size of window for each keyfitting attempt 
    #windowSize = 3200
    #the size of each pane; it'll fall into 8 possible key windows
    openCSV = open('bbkeyweights.csv','r')
    allProfiles = csv.reader(openCSV)
    keyProfiles = [] #stick each possible key profile vector in this list?
    for row in allProfiles:
        prof = []
        for entry in row:
            prof.append(int(entry))
        keyProfiles.append(prof)
    #paneSize = windowSize/8
    #get midi counters for each pane
    msandmidi = midiTimeWindows(paneSize, paneSize, solos=fp)
    keyConfDict = {}
    bestKeyDict = {}
    for i, row in enumerate(msandmidi):
        #if i == 0:
            #continue
        pcVector = []
        for j in range(12):
            pcVector.append(0)
        #Consider cases where there's not a full window left in msandmidi
        if i + int(math.floor(windowSize/paneSize)) > len(msandmidi) - 1:
            j = i
            while j < len(msandmidi):
                for mid, counts in six.iteritems(msandmidi[j][1]):
                    pcVector[mid%12] += counts
                j += 1
        #Also consider cases where there's not a full window left in a single track.
        #We still need to assign a key! But don't draw from next piece.
        elif msandmidi[i + int(math.floor(windowSize/paneSize))][0] < row[0]:
            j = i
            while j < i + int(math.floor(windowSize/paneSize)):
                for mid, counts in six.iteritems(msandmidi[j][1]):
                    pcVector[mid%12] += counts
                j += 1
        else:#here's for all the windows that do fit inside a single track
            for k in range(int(math.floor(windowSize/paneSize))):
                for mid, counts in six.iteritems(msandmidi[i+k][1]):
                    pcVector[mid%12] += counts
        #skip empty windows!
        pcVecSum = 0
        for n in range(12):
            pcVecSum += pcVector[n]
        if pcVecSum == 0:
            continue
        #now pcVector has the profile for a full window starting from pane i
        #take the pcVector from before and minimize its (cosine?) distance from one of these keyProfile vectors
        leastDist = 'na'
        for n,keyProf in enumerate(keyProfiles):
            #print n
            dot = 0
            magp = 0
            magk = 0
            for m in range(12):
                dot += keyProf[m]*pcVector[m]
                magp += math.pow(int(pcVector[m]),2)
                magk += math.pow(int(keyProf[m]),2)
            cosDist = 1 - dot/(math.sqrt(magp)*math.sqrt(magk))
            #print dot, math.sqrt(magp)*math.sqrt(magk), cosDist
            if leastDist == 'na':
                leastDist = cosDist
                secondLeastDist = 2
                thirdLeastDist = 3
                bestKey = n
                secondBestKey = 'na'
            elif leastDist > cosDist:
                thirdLeastDist = secondLeastDist
                thirdBestKey = secondBestKey
                secondLeastDist = leastDist
                secondBestKey = bestKey
                leastDist = cosDist
                bestKey = n
            elif secondLeastDist > cosDist > leastDist:
                thirdBestKey = secondBestKey
                thirdLeastDist = secondLeastDist
                secondBestKey = n
                secondLeastDist = cosDist
            elif thirdLeastDist > cosDist > secondLeastDist:
                thirdBestKey = n
                thirdLeastDist = cosDist
        #print bestKey, secondBestKey, thirdBestKey, i*paneSize
        #if the second best key is rel, skip it to get the confidence value
        if abs(bestKey - secondBestKey) == 9:
            bestKey_conf = 1 - leastDist/thirdLeastDist
        #if the second best key is NOT rel, then use if for conf value
        elif abs(bestKey - secondBestKey) != 9:
            bestKey_conf = 1 - leastDist/secondLeastDist#reasonable confidence measure? (Albrecht/Shanahan)
        #now, for each pane, see if it's the most confident key-guess yet
        for k in range(int(math.floor(windowSize/paneSize))):
            try:
                if keyConfDict[str(int((i+k)*paneSize))] < bestKey_conf:
                    keyConfDict[str(int((i+k)*paneSize))] = bestKey_conf
                    bestKeyDict[str(int((i+k)*paneSize))] = bestKey
            except KeyError:
                bestKeyDict[str(int((i+k)*paneSize))] = bestKey
                keyConfDict[str(int((i+k)*paneSize))] = bestKey_conf
    '''
    Stick the keyConf/AMBG rejection criteria HERE
    Add some major/minor escape clause
    if b > 12 and sb < 12 or if b < 12 and sb > 12:
    if bestKey - secondBestKey == +/-9, then maj/minor, which is OK
    '''
    for tms, conf in six.iteritems(keyConfDict):
        if conf < 0.3:
            bestKeyDict[tms] = 'AMBG'
    keyTally = collections.Counter()
    for tms, ky in six.iteritems(bestKeyDict):
        keyTally[str(ky)] += 1
    print(sorted(six.iteritems(keyTally), key=operator.itemgetter(1),reverse=True))
    #sorted_bestKeyDict = sorted(bestKeyDict.iteritems(), key=operator.itemgetter(0))
    #print sorted_bestKeyDict
    return bestKeyDict 
        
def transPCVecs(twindow,relTo='Do'):
    '''
    Widen windows over which keyfinding (and chords?) look.
    Calls jazzKeyFinder and midiTimeWindows to output a csv of locally-transposed, weighted pc vectors
    For each track, finds window and pane sizes which match perplexity constraints
    Normalizes window size to be closest integer multiple of pane size
    Runs keyfinding, tranposes relative to relTo ('la' or 'Do' supported)
    Outputs 'iwpcVecs twindow+relTo lcltrans.csv'
    '''
    #csvName ='iwpcVecs_lcltrans_wind.csv'
    fileName = Template('iwpcVecs $tw clcltrans.csv')
    csvName =fileName.substitute(tw = str(twindow)+relTo)
    file = open(csvName, 'w')
    lw = csv.writer(file)
    fileName2 = Template('iwpcVecs $tw cKeys.csv')
    csvName2 =fileName2.substitute(tw = str(twindow)+relTo)
    file2 = open(csvName2,'w')
    lw2 = csv.writer(file2)
    for m, testFile in enumerate(listing):
        #if m > 1:
            #break
        PerpAtSize = perplexity(solo=testFile)
        minDist = 'na'
        #this is a reasonable target perplexity, taken from the b-b profile
        scalePerp = 8.1
        #find what window size yields the closest perplexity
        for size,perp in PerpAtSize:
            fitDist = abs(scalePerp - perp)
            if minDist == 'na':
                minDist = fitDist
                bestSize = size
            elif minDist > fitDist:
                minDist = fitDist
                bestSize = size
        #for the pane size, look for when perplexity increases 30% over smallest window
        startPerp = PerpAtSize[0][1]
        for size,perp in PerpAtSize:
            if (perp - startPerp)/startPerp > 0.3:
                paneSize = size
                break
        '''
        Originally: window size whatever avg window closest to perp 8.1
        Now: same, except bump up windows < 16 secs
        '''
        while bestSize < 16000:
            bestSize = bestSize*2
        print("Best window size for",testFile,"is",bestSize,"with paneSize",paneSize)
        #send window and pane size through local key finder
        keyDict = jazzKeyFinder(testFile, bestSize, paneSize)
        #use resulting ms-indexed keys to transpose tracks
        #unTrans = midiTimeWindows(2*bestSize, paneSize, solos=testFile)
        #for window size, choose closest integer multiple of paneSize
        twindow = round(twindow/paneSize)*paneSize
        unTrans = midiTimeWindows(twindow, paneSize, solos=testFile)
        windKeys = []
        for tms, pccol in unTrans:
            try:
                lclKey = keyDict[str(int(tms))]
            except KeyError:
                continue
            if lclKey == 'AMBG':
                continue
            #print tms
            #if i == 0:
                #continue
            if sum(pccol.values()) == 0:#skip the empty windows
                continue
            pcVector = []
            #print pccol
            for j in range(12):
                pcVector.append(0.0)
            if relTo=='Do':#trans major/minor the same
                if lclKey > 11:
                    lclKey -= 12
                for mid, counts in six.iteritems(pccol):
                    pcVector[(mid - lclKey)%12] += counts
            if relTo=='La':#trans minor rel to la and major rel to do
                if lclKey <= 11:#major key cases
                    for mid, counts in six.iteritems(pccol):
                        pcVector[(mid  - lclKey)%12] += counts
                elif lclKey > 11:#minor key cases
                    lclKey -= 12
                    for mid, counts in six.iteritems(pccol):
                        #here: the -3 is for trans rel to la, not do
                        pcVector[(mid  - lclKey - 3)%12] += counts
            lw.writerow(pcVector)
            windKeys.append([tms,lclKey])
        windKeys[-1] = [tms, lclKey,'track end']
        for row in windKeys:
            lw2.writerow(row)

def chordFinder(wsize,relTo='La',withTimeStamps=False):
    """
    We also have a lot of things that look like chords + single scale step
    Need to gather chords based on closeness of note onsets
    Go track by track, running jazzKeyFinder and midiTimeWindows with very small size panes (50ms, non-overlapping) and large windows
    For now, output three csvs:
         1. csv of all the scale degree set slices (sdcvecs)
         2. csv of all the 50ms timestamped keys (keys)
         3. csv of locally-transposed midi voicings, st 0<bass<12 (So: voiced scale degrees)
    """
    #Make a keyDict
    #csvName ='iwpcVecs_lcltrans_wind.csv'
    fileName = Template('sdcVecs $tw 2xwind TS.csv')
    fileName2 = Template('$tw 2xwind keys TS.csv')
    fileName3 = Template('$tw 2xwind transMIDI TS.csv')
    csvName = fileName.substitute(tw = str(wsize)+relTo)
    csvName2 = fileName2.substitute(tw = str(wsize)+relTo)
    csvName3 = fileName3.substitute(tw = str(wsize)+relTo)
    lw = csv.writer(open(csvName,'w',newline='\n'))
    lw2 = csv.writer(open(csvName2,'w',newline='\n'))
    lw3 = csv.writer(open(csvName3,'w',newline='\n'))
    for m, testFile in enumerate(listing):
        #if m > 1:
            #break
        PerpAtSize = perplexity(solo=testFile)
        minDist = 'na'
        #this is a reasonable target perplexity, taken from the b-b profile
        scalePerp = 8.1
        #find what window size yields the closest perplexity
        for size,perp in PerpAtSize:
            fitDist = abs(scalePerp - perp)
            if minDist == 'na':
                minDist = fitDist
                bestSize = size
            elif minDist > fitDist:
                minDist = fitDist
                bestSize = size
        #for the pane size, look for when perplexity increases 30% over smallest window
        startPerp = PerpAtSize[0][1]
        for size,perp in PerpAtSize:
            if (perp - startPerp)/startPerp > 0.3:
                paneSize = size
                break
        '''
        Originally: window size whatever avg window closest to perp 8.1
        Now: window size 2 times whatever that was/is, and it has to start bigger than 16s
        Increase window size even more and look for improvement?
        '''
        while bestSize < 16000:
            bestSize = bestSize*2
        print("Best window size for",testFile,"is",2*bestSize,"with paneSize",paneSize)
        #send window and pane size through local key finder
        keyDict = jazzKeyFinder(testFile, 2*bestSize, paneSize)
        #output csv of timestamped keys
        sorted_keyDict = sorted(six.iteritems(keyDict),key=operator.itemgetter(0))
        for tms, ky in sorted_keyDict:
            lw2.writerow([tms,ky])
        #use resulting ms-indexed keys to transpose tracks
        #for window size, choose super small chord-capturing size (50ms)
        unTrans = midiTimeWindows(wsize, wsize, solos=testFile)
        for tms, pccol in unTrans:
            try:
                lclKey = keyDict[str(int(tms))]
            except KeyError:
                for pane, ky in six.iteritems(keyDict):
                    if int(pane) < int(tms) < int(pane)+paneSize:
                        lclKey = ky
                        break
            if lclKey == 'AMBG':
                continue
            #print tms
            #if i == 0:
                #continue
            if sum(pccol.values()) == 0:#skip the empty windows
                continue
            pcVector = []
            transMIDIvec = []
            #print pccol
            for j in range(12):
                pcVector.append(0.0)
            if withTimeStamps==True:
                pcVector.append(tms)
                pcVector.append(testFile)
            if relTo=='Do':#trans major/minor the same
                if lclKey > 11:
                    lclKey -= 12
                for mid, counts in six.iteritems(pccol):
                    pcVector[(mid - lclKey)%12] += counts
                    if mid - lclKey < 0:
                        transMIDIvec.append(mid - lclKey + 12)
                    else:
                        transMIDIvec.append(mid - lclKey)
            if relTo=='La':#trans minor rel to la and major rel to do
                if lclKey <= 11:#major key cases
                    for mid, counts in six.iteritems(pccol):
                        pcVector[(mid  - lclKey)%12] += counts
                    if mid - lclKey < 0:
                        transMIDIvec.append(mid - lclKey + 12)
                    else:
                        transMIDIvec.append(mid - lclKey)
                elif lclKey > 11:#minor key cases
                    lclKey -= 12
                    for mid, counts in six.iteritems(pccol):
                        #here: the -3 is for trans rel to la, not do
                        pcVector[(mid  - lclKey - 3)%12] += counts
                        if mid - lclKey -3 < 0:
                            transMIDIvec.append(mid - lclKey -3 + 12)
                        else:
                            transMIDIvec.append(mid - lclKey -3)
            sorted_MV = sorted(transMIDIvec)
            if withTimeStamps==True:
                tmsMV = [tms,testFile]
                for mv in sorted_MV:
                    tmsMV.append(mv)
                lw3.writerow(tmsMV)
            else:
                lw3.writerow(sorted_MV)
            '''Below turns MIDI vecs into voicing classes, moving them all down so 0 < lowest voice < 11
            while sorted_MV[0] > 11:
                for m in range(len(sorted_MV)):
                    sorted_MV[m] -= 12
            '''
            lw.writerow(pcVector)
            
def tallySDSets(cmin,probs=False):
    """
    Takes output of chordFinder (csv of locally-transposed, 50ms SDvecs)
    Tallies up all the SD sets.  IF probs=False, gives tallies; if True, gives probs
    Outputs a csv of them
    Returns a dict
    """
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind.csv','r',newline='\n'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
    SDSets = collections.Counter()
    for midvec in allMidiList:
        SDs = set([])
        for n in range(12):
            if midvec[n] != '0.0':
                SDs.add(n)
        if len(SDs) >= cmin:
            SDSets[str(sorted(SDs))] += 1
    if probs==True:
        SDSets_probs = getProbsFromFreqs(SDSets)
        sorted_SDSets = sorted(six.iteritems(getProbsFromFreqs(SDSets)),key=operator.itemgetter(1),reverse=True)
        fp = Template('50ms $cm SDSets Probs.csv')
    else:
        sorted_SDSets = sorted(six.iteritems(SDSets), key=operator.itemgetter(1),reverse=True)
        fp = Template('50ms $cm SDSets.csv')
    '''
    csvName = fp.substitute(cm=cmin)
    x = csv.writer(open(csvName, 'w',newline='\n'))
    for pair in sorted_SDSets:
        x.writerow([pair[0], pair[1]])
    '''
    return SDSets_probs
        
def nAfterSDS(sds,numWin,supersets=False,probs='None',filtp=0.0):
    """
    Takes a given scale degree set and tallies up what happens with next numWin windows
    By default, doesn't count scale degree supersets, but it will consider all SDSS as equivalent 
    if you set supersets=True
    Also compares to unigram probs if probs='Rel'
    Just absolute, distance-based probs if probs='Abs'
    Tallies only if probs='None'
    filtp sets a unigram prob cutoff for stuff that gets counted.  Below it, no stats kept (not very helpful)
    If sds is a list, it assumes a single scale degree set and tracks what follows
    If sds is a tuple, 
    TODO:
        EMERGENCY: This is running right across track ends!!
         1. Add parameter for increasing window size (combining 50ms panes) (easy? But watch out for endings)
         2. Add mode for tracking equivalence classes (my hand-picked I, ii, V) (harder; make a kludge?)
    """
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind.csv','r',newline='\n'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
    SDSetsT = collections.Counter()
    for i, midvec in enumerate(allMidiList):
        SDset = set([])
        for n in range(12):
            if midvec[n] != '0.0':
                SDset.add(n)
        '''
        if sds is a set, then it's a single scale degree set, and leave the old methods alone
        if sds is a tuple (sds, dict), look for all ocs that are in dict[sds]
        Ex: ('ii',{'ii': [[0,2,5],[0,2,5,9],[2,5,9]],'V': [[5,7,11],[2,5,7,11],[2,7,11]], 'I': [[0,4,7],[0,3,7],[0,4,11],[0,3,10],[0,4,7,11],[0,3,7,10]]})
        '''
        if type(sds) == list:
            #print('Running single origin chord mode')
            if sorted(SDset) == sds or (supersets == True and SDset.issuperset(sds) == True):
                j = i+1
                while j < i + numWin + 1 and i + j < len(allMidiList):
                    nextSet = set([])
                    for m in range(12):
                        if allMidiList[j][m] != '0.0':
                            nextSet.add(m)
                    if len(nextSet) < 3:
                        j += 1
                        continue
                    try:
                        SDSetsT[j-i][str(sorted(nextSet))] += 1
                    except TypeError:
                        SDSetsT[j-i] = collections.Counter()
                        SDSetsT[j-i][str(sorted(nextSet))] += 1
                    j += 1
        elif type(sds) == tuple:
            #print('Trying manually-defined category mode')
            catDict = sds[1]#all of our categories in one dict, indexed by label
            ocs = catDict[sds[0]]#these are the possible origin chords in category
            #print('possible origin chords in category: ',ocs)
            if sorted(SDset) in ocs:
                j = i+1
                while j < i + numWin + 1 and i + j < len(allMidiList):
                    nextSet = set([])
                    #print(allMidiList[j])
                    for m in range(12):
                        if allMidiList[j][m] != '0.0':
                            nextSet.add(m)
                    if len(nextSet) < 3:
                        j += 1
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
                    j += 1
    #print(SDSetsT[0])
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
                #print("log of 4 is",log(4))
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
            fileName = Template('$voic SDs prog 50ms.csv') if supersets == False else Template('$voic SDSS prog 50ms.csv')
        elif probs=='Rel':
            fileName = Template('$voic SDs prog rlogprobs 50ms.csv') if supersets == False else Template('$voic SDSS prog logprobs 50ms.csv')
        elif probs=='Abs':
            fileName = Template('$voic SDs prog probs 50ms.csv') if supersets == False else Template('$voic SDSS prog logprobs 50ms.csv')
        csvName = fileName.substitute(voic = str(sds))
    elif type(sds) == tuple:
        if probs=='None':
            fileName = Template('$voic SDs prog 50ms.csv') if supersets == False else Template('$voic SDSS prog 50ms.csv')
        elif probs=='Rel':
            fileName = Template('$voic SDs prog rlogprobs 50ms.csv') if supersets == False else Template('$voic SDSS prog logprobs 50ms.csv')
        elif probs=='Abs':
            fileName = Template('$voic SDs prog probs 50ms.csv') if supersets == False else Template('$voic SDSS prog logprobs 50ms.csv')
        csvName = fileName.substitute(voic = sds[0])
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in SDSetsT:
        dw.writerow(SDSetsT[row])
        
def sdsSupersetVoicings(sds):
    """
    For a given scale degree set SDS, locate all relevant SD supersets
    Iterate through transMIDI voiced scale degree set data (0<bass<12, upper scale degree voicings)
    """
    allPanes = csv.reader(open('50Do 2xwind transMIDI.csv','r'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
    SDSetsT = collections.Counter()
    for transmid in allMidiList:
        SDset = set([])
        for n in range(len(transmid)):
            SDset.add(int(transmid[n])%12)
        if SDset.issuperset(sds) == True:
            SDSetsT[str(transmid)] += 1
    #write the CSV
    fileName = Template('$voic voiced SDSS 50ms.csv')
    csvName = fileName.substitute(voic = str(sds))
    lw = csv.writer(open(csvName, 'w'))
    sorted_SDSets = sorted(six.iteritems(SDSetsT), key=operator.itemgetter(1),reverse=True)
    for pair in sorted_SDSets:
        lw.writerow([pair[0], pair[1]])

def voicingAsSDS(voicing):
    """
    answers question: for a given (untransposed) voicing, what are its most common SDSS deployments?
    this can be done from the locally-transposed MIDI data (NOT the pc or sds data)
    """
    allPanes = csv.reader(open('50Do 2xwind transMIDI.csv','r'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
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
        
def csvTransposer(f,d):
    infile = open(f)
    reader = csv.reader(infile)
    cols = []
    for row in reader:
        cols.append(row)
    outfile = open(d,'w',newline='\n')
    writer = csv.writer(outfile)
    for i in range(len(max(cols, key=len))):
        writer.writerow([(c[i] if i<len(c) else '') for c in cols]) 

def syntax_backwards(sds,numWin,supersets=False,probs='None',filtp=0.0):
    """
    Takes a given scale degree set and tallies up what happens in the PRECEDING numWin windows
    By default, doesn't count scale degree supersets, but it will consider all SDSS as equivalent if you set supersets=True
    Also compares to unigram probs if probs='Rel'
    Just absolute, distance-based probs if probs='Abs'
    Tallies only if probs='None'
    filtp sets a unigram prob cutoff for stuff that gets counted.  Below it, no stats kept (not very helpful)
    If sds is a list, it assumes a single scale degree set and tracks what follows
    If sds is a tuple, manual given categories are imposed!
    TODO:
        EMERGENCY:
        1.  This is running right across track ends!!
        2.  Add parameter for increasing window size (combining 50ms panes) (easy? But watch out for endings)
    """
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind.csv','r',newline='\n'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
    SDSetsT = collections.Counter()
    for i, midvec in enumerate(allMidiList):
        SDset = set([])
        for n in range(12):
            if midvec[n] != '0.0':
                SDset.add(n)
        '''
        if sds is a set, then it's a single scale degree set, and leave the old methods alone
        if sds is a tuple (sds, dict), look for all ocs that are in dict[sds]
        Ex: ('ii',{'ii': [[0,2,5],[0,2,5,9],[2,5,9]],'V': [[5,7,11],[2,5,7,11],[2,7,11]], 'I': [[0,4,7],[0,3,7],[0,4,11],[0,3,10],[0,4,7,11],[0,3,7,10]]})
        '''
        if type(sds) == list:#This seems to work backwards correctly
            #print('Running single origin chord mode')
            if sorted(SDset) == sds or (supersets == True and SDset.issuperset(sds) == True):
                j = i-1
                while j > i - numWin - 1:
                    if j < 0:
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
                    if j < 0:
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
    print(SDSetsT)
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
    file = open(csvName, 'w',newline='\n')
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
    allChords = csv.reader(open('50ms 3 SDSets.csv','r',newline='\n'))
    allChordsList = []
    for i, row in enumerate(allChords):
        if i > numchords - 1:
            break
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
    for chd in allChordsList:
        if backwards == False:
            nAfterSDS(chd, numwind, probs=prb, filtp=0.0)
        else:
            syntax_backwards(chd, numwind, probs=prb, filtp=0.0)
            
def sdsTracker(sds):
    """
    Search for a scale degree set sds; find the tracks and timestamps where they occur
    Output a (sorted) list of which tracks and the timestamps within them
    """
    from string import Template
    sliceCount = {}
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind TS.csv','r',newline='\n'))
    allMidiList = []
    allTMS = []
    tms = {}#a dict; each file name will be a key, and the value is a list of times where the sds occurs
    for row in allPanes:
        allMidiList.append(row[0:-2])#just the pitches
        allTMS.append(row[-2:])#just the timestamp and file name
    #print(allMidiList)
    #SDSetsT = collections.Counter()
    #print(sds)
    for i, midvec in enumerate(allMidiList):
        SDset = set([])
        for n in range(12):
            if midvec[n] != '0.0':
                SDset.add(n)
        if sorted(SDset) == sds:
            try:
                tms[allTMS[i][1]].append(int(allTMS[i][0]))#add a timestamp to the dict entry for this track
            except KeyError:#unless there's no dict entry for that track... in which case it should be a list
                tms[allTMS[i][1]] = []
    #once all entries are added to the tms dict, listify it and sort by total tally
    outputList = []
    for key, value in tms.items():
        outputList.append([len(value),key,sorted(value)])
    #print(tms)
    print(outputList)
    #put the (sorted) outputList in a csv
    fileName = Template('$voic SDs locations.csv')
    csvName = fileName.substitute(voic = str(sds))
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    sol = sorted(outputList,key=operator.itemgetter(0),reverse=True)
    for row in sol:
        lw.writerow(row)

#sdsTracker([0,2,7])
#runtopNchords(1836, 100, prb='Abs',backwards=True)#1836 should be ALL of them.  Run these forwards, too, and zip/send to Jeremy
#syntax_backwards([0,2,5], 100, probs='Rel', filtp=0.0)
#syntax_backwards(('ii',{'ii': [[0,2,5],[0,2,5,9],[2,5,9]],'V': [[5,7,11],[2,5,7,11],[2,7,11]], 'I': [[0,4,7],[0,3,7],[0,4,11],[0,3,10],[0,4,7,11],[0,3,7,10]]}),100,supersets=False,probs='Rel',filtp=0.0)
#sdsSupersetVoicings([0,2,7])
#nAfterSDS(('V',{'ii': [[0,2,5],[0,2,5,9],[2,5,9]],'V': [[5,7,11],[2,5,7,11],[2,7,11]], 'I': [[0,4,7],[0,3,7],[0,4,11],[0,3,10],[0,4,7,11],[0,3,7,10]]}),100,supersets=False,probs='Abs',filtp=0.0)
nAfterSDS([0,4,5,9],100,supersets=False,probs='Abs',filtp=0.0)
csvTransposer('[0, 4, 5, 9] SDs prog probs 50ms.csv', '[0, 4, 5, 9] SDs prog probs 50msTRANS.csv')   
#tallySDSets(1,probs=True)             
#chordFinder(50, relTo='Do',withTimeStamps=True)
#tallySDSets(3,probs=True)
#voicingAsSDS([0,5,10])
'''
transPCVecs(800, relTo='La')
transPCVecs(1200, relTo='La')
transPCVecs(1200, relTo='Do')
transPCVecs(800, relTo='La')
'''  