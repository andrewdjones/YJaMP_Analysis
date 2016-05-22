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

def getProbsFromFreqs(DictionaryOfTallies):
    #Simple script to turn a dict of tallies into a dict of normalized probabilities
    totalSum = 0.0
    dictOfProbs = {}
    for key, freq in six.iteritems(DictionaryOfTallies):
        totalSum += float(freq)
    for key, freq in six.iteritems(DictionaryOfTallies):
        dictOfProbs[key] = float(freq)/totalSum
    return dictOfProbs

def csvTransposer(f,d):
    #Basic: does what it says, and leaves csv f alone after creating T(f) = d
    infile = open(f)
    reader = csv.reader(infile)
    cols = []
    for row in reader:
        cols.append(row)
    outfile = open(d,'w',newline='\n')
    writer = csv.writer(outfile)
    for i in range(len(max(cols, key=len))):
        writer.writerow([(c[i] if i<len(c) else '') for c in cols]) 
    
def chordSlicesWithKey(path):
    """
    DEPRECATED: chordFinder and midiTimeWindows do better keyfinding, now, and csvs are more easily shareable
    
    This is basically just a fancy slicer/pickle producer
    Starts from whatever collection of MIDI files you want
    1. Finds the best key guess for each individual midi file (1 per file!)
    2. For each slice, a dict is created with all its info (see below for dict keys)
    3. Slice dicts are stored separately for major and minor mode stuff
    Outputs two pickles and two voicing type tallies (major and minor)
    """
    
    listing = os.listdir(path)
    totalChordsMajor = 0
    totalChordsMinor = 0
    chordCountMajor = {}#tally, format: chordCount[sorted pitch class set, bass note] = tally
    chordCountMinor = {}#ditto
    theSlicesMajor = []#list of dicts; each dict holds info about a successive chord
    theSlicesMinor = []#ditto
    problemFiles = []
    
    for f in listing:
        address = path + f
        print('current file:',address)
        
        #use music21 (slow) parser
        try:
            oneSolo = converter.parse(address)
        except:
            problemFiles.append(f)
            print('Problem with',f)
            pass
        else:
            #built-in keyfinding
            theKey = analysis.discrete.analyzeStream(oneSolo, 'bellman')
            theTonic = str(theKey).split(' ')[0]
            theMode = str(theKey).split(' ')[1]
            theKeyPC = pitch.Pitch(theTonic).pitchClass
            #print 'current key:', theKey, theKeyPC, theMode
            
            #built-in chord finding
            theSoloChords = oneSolo.chordify().flat.getElementsByClass('Chord')
            
            #separate major mode from minor, toss out errors
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
            
            #start token, dict entry for each chord, end token
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
                
                #tally up the pitch class sets and bass note as entity
                try:
                    chordCount[str((someChord.orderedPitchClasses,bassNoteSD))] += 1
                except KeyError:
                    chordCount[str((someChord.orderedPitchClasses,bassNoteSD))] = 1
                totalChords += 1
            endToken = ['end']
            theSlices.append(endToken)
            
            #now reassign the new slice list in place of the old one
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
    '''
    DEPRECATED: chordFinder and midiTimeWindows do better (non-music21) keyfinding
    
    look through the already-processed pickles (from chordSlicesWithKey)
    assemble a list of the keys for each solo
    make listofKeys = [filename,int(tonic)] to subtract tonic out in midi later
    '''
    
    #Get pickled data
    listofKeys = []#[filename, int(tonic)]
    theMajPickle = '1122MajModeSliceDictwSDB.pkl'
    theSlices = pickle.load(open(theMajPickle, 'r'))
    
    #toss start and end slices, and pull key info from the rest
    for i,slice in enumerate(theSlices):
        if slice != ['start']:
            continue
        theTonic = theSlices[i+1]['key'].split(' ')[0]
        theKeyPC = pitch.Pitch(theTonic).pitchClass
        #string formatting kludge for quantized/edited file names
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
    
    #stick the pitch classes corresponding to keys in a csv
    csvName = 'solokeyoffsets_a_C.csv'
    file = open(csvName, 'w')
    lw = csv.writer(file)
    for row in listofKeys:
        lw.writerow(row)
    print(listofKeys)

#############################################################################
'''
headKeys and keyProfiles provide a test of bellman-budge vs. jazz key profiles
1. headKeys uses bellman-budge to guess the key of all the jazz heads in YJaMP
2. I manually check the key assignments to make sure they seem reasonable
3. keyProfiles calculates the transposed key profile for each head
Conclusion: head key profiles match bellman-budge profiles pretty closely
'''
#----------------------------------------------------------------------------   
def headKeys():
    '''
    This makes a key-finding guess for the heads in MIDIheads directory.  Outputs a csv for it all.
    Hand check all of these
    Then keyProfiles can extract jazz key profile vectors for comparison.
    '''
    path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIheads_q/'
    listing = os.listdir(path)
    problemFiles = []
    headAndKey = []#format: [filename, key]
    
    #use music21 parser on manually-identified heads; keyfind
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
    
    #write key assignments to CSV
    file = open('jazz_headKeys_1122.csv', 'w')
    lw = csv.writer(file)
    for row in headAndKey:
        lw.writerow(row)
        
def keyProfiles():
    '''
    Inputs the head .mid files and the list of identified keys from headKeys
    Transposes each track relative to ``tonic" (hypothesized, classicized)
    Outputs weighted pc vector for each head
    If these look like classical key vectors, then keyfinding assumptions justifiable
    (Update: They do and are.)
    '''
    #get the key assignments for the heads
    fp = open('jazz_headKeys_1122.csv','r')
    keyDict = {}
    dta = csv.reader(fp)
    for row in dta:
        keyDict[row[0]] = row[1]
    #print keyDict
    
    #pull the heads themselves and use music21 to build pc vectors
    path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIheads_q/'
    listing = os.listdir(path)
    problemFiles = []
    headpcVec = []#format: [filepath, weighted pc vector]
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
        
        #put the transposed pitches and durations into headpcVec
        pcVec = []#will be 12-entry vector of durationally-weighted pitch classes
        for j in range(12):
            pcVec.append(0.0)#floats
        theSoloChords = oneSolo.flat.getElementsByClass(note.Note)#notes
        #times and pitch classes
        for k in range(len(theSoloChords.secondsMap)):
            transPC = (theSoloChords.secondsMap[k]['element'].pitch.pitchClass - theKeyPC)%12
            pcVec[transPC] += theSoloChords.secondsMap[k]['durationSeconds']
        headpcVec.append([f,pcVec])
        
    #output list of head pcVecs as csv
    file = open('jazz_head_transpcVecs.csv', 'w')
    lw = csv.writer(file)
    for row in headpcVec:
        lw.writerow(row)
#############################################################################

#############################################################################
'''
chordFinder(wsize):
1. Iterate through each solo
2. For each track, iterate a series of sliding windows through midi data
   For each window, tally up the durationally-weighted pcvector for the window
   For each window size, calculate average perplexity
   CALLS: perplexity(solo=filename), jazzKeyFinder, midiTimeWindows
3. From average perplexity vs. window size distribution, choose window size such that perplexity ~8.1
   This is chosen to roughly match existing scale/key profiles (esp. bellman-budge)
4. Also from avg perp vs. window size dist, choose pane size such that perplexity first starts to rise
   Trigger: increase by 30% over perplexity of smallest window size (consider better justification?)
5. Using window and pane size, slide through track for local keyfinding
   CALLS: jazzKeyFinder(filepath)
Then, separately:
6. Re-run midiTimeWindows, transposing each pane by cross-referencing local key list/dict
OUTPUT: csv of durationally-weighted pcVecs for locally-transposed windows
The results get used in scripts like catTPD.py, for tallying SDsets and building probability dists
'''
#----------------------------------------------------------------------------
def midiTimeWindows(windowWidth,incUnit,solos=all):
    '''
    windowWidth in millisecs; incUnit how large the window slide step is (also ms)
    Lots of processing to make list:
    [millisecs at end of window, music21 chord, set of midi numbers, pcs in order, file name]
    '''
    #use unquantized data, since we're bypassing music21's quantization requirements
    path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIunquant/'
    listing = os.listdir(path)
    
    #for each solo, put the timestamp and midi contents for each window in msandmidi
    msandmidi = []
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
    '''
    return msandmidi

def perplexity(solo=all,sendData=False):
    '''
    Takes solo (or all, if solo==all) and calculates average window perplexity
    Calculates for range 25ms to 60*1000 ms by doubling
    (Calls midiTimeWindows for segmentation)
    Returns list of [[window size, avg perplexity],[wsize,avg perp],...]
    Writes to csv if sendData==True
    '''
    windowSize = 12.5#start size is twice this
    PerpAtSize = []#each list entry will be of form [window size, avg perplexity]
    while windowSize < 60000:
        windowSize = windowSize*2
        #print windowSize
        
        #Let sliding increment change as window size increases
        if windowSize <= 1000:
            incUnit = 25
        elif 1000 < windowSize < 10000:
            incUnit = 250
        else:
            incUnit = 1000
            
        #Get a list of the midiTimeWindows for windows of windowSize incrementing by incUnit
        if solo != all:#single track mode
            msandmidi = midiTimeWindows(windowSize, incUnit, solos=solo)
        else:#all corpus mode
            msandmidi = midiTimeWindows(windowSize,incUnit)
            
        #iterate through particular midiTimeWindows listing and calculate perplexities
        perplexities = []#list of consecutive window perplexities, 2^(entropy)
        for i, row in enumerate(msandmidi):
            #if i == 0:
                #continue
            if sum(row[1].values()) == 0:#skip the empty windows
                continue
            pcVector = []#will be 12-entry pitch class vector for window
            for j in range(12):
                pcVector.append(0.01)#slight smoothing
            for mid, counts in six.iteritems(row[1]):
                pcVector[mid%12] += counts
            perplexities.append(scipy.exp2(scipy.stats.entropy(pcVector,base=2)))
            #print windowSize,pcVector, entropies[-1]
            
        #put average perplexity in PerpAtSize
        PerpAtSize.append([windowSize,scipy.average(perplexities)])
    
    #write to csv
    if sendData:
        if solo != all:
            fileName = Template('$sol overlap window pc avg perp.csv')
            csvName =fileName.substitute(sol = solo.split('.')[0])
        else:
            csvName ='overlap window pc avg entropy.csv'
        file = open(csvName, 'w')
        lw = csv.writer(file)
        for row in PerpAtSize:
            lw.writerow(row)
    return PerpAtSize

def jazzKeyFinder(fp,windowSize,paneSize):
    '''
    Called by chordFinder, this slides a window of windowSize by paneSize increments
    Calculates the pitch class profile for each window
    Compares pcprofs to bellman-budge key profile
    Each pane inherits key from its window of best key fit
    '''
    
    #an empty list for the small, final key slices
    panesAndKeys = []
    
    #Get the bellman-budge key profiles for each key
    openCSV = open('bbkeyweights.csv','r')
    allProfiles = csv.reader(openCSV)
    keyProfiles = [] #stick each possible key profile vector in this list
    for row in allProfiles:
        prof = []
        for entry in row:
            prof.append(int(entry))
        keyProfiles.append(prof)
        
    #get midi counters for each pane
    msandmidi = midiTimeWindows(paneSize, paneSize, solos=fp)
    keyConfDict = {}#keyed by timestamp, gives the confidence of best key assignment for each pane
    bestKeyDict = {}#keyed by timestamp, gives best key assignment for each pane
    for i, row in enumerate(msandmidi):
        #if i == 0:
            #continue
        
        #put all the pitch content from appropriate panes into window pcVector
        pcVector = []#12-entry pitch class vector to keep track of window contents
        for j in range(12):
            pcVector.append(0)
        #Consider cases where there's not a full window's worth of panes left in msandmidi list
        if i + int(math.floor(windowSize/paneSize)) > len(msandmidi) - 1:
            j = i
            while j < len(msandmidi):
                for mid, counts in six.iteritems(msandmidi[j][1]):
                    pcVector[mid%12] += counts
                j += 1
        #Also consider cases where there's not a full window left in a single track.
        #We still need to assign a key! But don't draw from next piece.
        elif msandmidi[i + int(math.floor(windowSize/paneSize))][0] < row[0]:#compare consecutive timestamps
            j = i
            while j < i + int(math.floor(windowSize/paneSize)):
                for mid, counts in six.iteritems(msandmidi[j][1]):
                    pcVector[mid%12] += counts
                j += 1
        #here's for all the windows that do fit inside a single track
        else:
            for k in range(int(math.floor(windowSize/paneSize))):
                for mid, counts in six.iteritems(msandmidi[i+k][1]):
                    pcVector[mid%12] += counts
                    
        #skip empty windows
        pcVecSum = 0
        for n in range(12):
            pcVecSum += pcVector[n]
        if pcVecSum == 0:
            continue
        
        #now pcVector has the profile for a full window starting from pane i
        #minimize pcVector's cosine distance from one of the keyProfile vectors
        leastDist = 'na'
        for n,keyProf in enumerate(keyProfiles):
            #kludgy cosine dist for clarity
            dot = 0
            magp = 0
            magk = 0
            for m in range(12):
                dot += keyProf[m]*pcVector[m]
                magp += math.pow(int(pcVector[m]),2)
                magk += math.pow(int(keyProf[m]),2)
            cosDist = 1 - dot/(math.sqrt(magp)*math.sqrt(magk))
            #print dot, math.sqrt(magp)*math.sqrt(magk), cosDist
            
            #iterative complexity to keep top three key assignment options
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
        
        #if the second best key is the relative major/minor, skip it to get the confidence value
        if abs(bestKey - secondBestKey) == 9:
            bestKey_conf = 1 - leastDist/thirdLeastDist
        #if the second best key is NOT rel, then use if for confidence value
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
    
    #Reject key assignment if confidence < 30%
    for tms, conf in six.iteritems(keyConfDict):
        if conf < 0.3:
            bestKeyDict[tms] = 'AMBG'
            
    #print a tally of most common kekys (which isn't kept)
    keyTally = collections.Counter()
    for tms, ky in six.iteritems(bestKeyDict):
        keyTally[str(ky)] += 1
    print(sorted(six.iteritems(keyTally), key=operator.itemgetter(1),reverse=True))
    #sorted_bestKeyDict = sorted(bestKeyDict.iteritems(), key=operator.itemgetter(0))
    #print sorted_bestKeyDict
    return bestKeyDict 

def chordFinder(wsize,relTo='La',withTimeStamps=False):
    '''
    We also have a lot of things that look like chords + single scale step
    Need to gather chords based on closeness of note onsets
    Go track by track, running jazzKeyFinder and midiTimeWindows with very small size panes (50ms, non-overlapping) and large windows
    For now, output three csvs:
         1. csv of all the scale degree set slices (sdcvecs)
         2. csv of all the 50ms timestamped keys (keys)
         3. csv of locally-transposed midi voicings, st 0<bass<12 (So: voiced scale degrees)
    '''
    #Set up the output csvs
    fileName = Template('sdcVecs $tw 2xwind TS.csv')
    fileName2 = Template('$tw 2xwind keys TS.csv')
    fileName3 = Template('$tw 2xwind transMIDI TS.csv')
    csvName = fileName.substitute(tw = str(wsize)+relTo)
    csvName2 = fileName2.substitute(tw = str(wsize)+relTo)
    csvName3 = fileName3.substitute(tw = str(wsize)+relTo)
    lw = csv.writer(open(csvName,'w',newline='\n'))
    lw2 = csv.writer(open(csvName2,'w',newline='\n'))
    lw3 = csv.writer(open(csvName3,'w',newline='\n'))
    
    #give a directory for the unquantized midi files
    path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIunquant/'
    listing = os.listdir(path)
    
    #iterate through solos in listing
    for m, testFile in enumerate(listing):
        #if m > 1:
            #break
            
        #set window and pane sizes from a wide range of perplexities
        PerpAtSize = perplexity(solo=testFile)#list of lists [window size, avg perplexity]
        minDist = 'na'
        scalePerp = 8.1#this is a reasonable target perplexity, taken from the bellman-budge profile
        #find what window size yields the closest perplexity to target
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
            
        """
        Proceed to perform keyfinding and transposition on track at best time window/pane scales
        Originally: window size whatever avg window closest to perp 8.1
        Now: window size 2 times whatever that was/is, and it has to start bigger than 16s
        """
        while bestSize < 16000:
            bestSize = bestSize*2
        print("Best window size for",testFile,"is",2*bestSize,"with paneSize",paneSize)
        
        #send window and pane size through local key finder
        keyDict = jazzKeyFinder(testFile, 2*bestSize, paneSize)
        #output csv of timestamped keys
        sorted_keyDict = sorted(six.iteritems(keyDict),key=operator.itemgetter(0))
        for tms, ky in sorted_keyDict:
            lw2.writerow([tms,ky])#write timestamped keys to file
        
        """
        use resulting ms-indexed keys to transpose tracks
        for window size, choose super small chord-capturing size (50ms)
        NB: this size set by hand, unlike the keyfinding pane size above, set by perplexity
        That's so we can be sure to capture "phonetic-type" data (viz. linguistic models)
        """
        unTrans = midiTimeWindows(wsize, wsize, solos=testFile)#untransposed midi data
        for tms, pccol in unTrans:
            #look up the right key for each window by timestamp
            try:
                lclKey = keyDict[str(int(tms))]
            except KeyError:#window falls between key assignments; inherit
                for pane, ky in six.iteritems(keyDict):
                    if int(pane) < int(tms) < int(pane)+paneSize:
                        lclKey = ky
                        break
            if lclKey == 'AMBG':
                continue
            if sum(pccol.values()) == 0:#skip the empty windows
                continue
            
            #transpose weighted pitch classes in window by local key (lclkey)
            pcVector = []#tranposed scale degree classes
            transMIDIvec = []#transposed pitches (midi)
            for j in range(12):
                pcVector.append(0.0)
            if withTimeStamps==True:
                pcVector.append(tms)
                pcVector.append(testFile)
            if relTo=='Do':#transpose major/minor the same
                if lclKey > 11:#recall: 0-11 major keys, 12-23 minor keys
                    lclKey -= 12
                for mid, counts in six.iteritems(pccol):
                    pcVector[(mid - lclKey)%12] += counts
                    if mid - lclKey < 0:#light fudging in case transposition goes below midi 0
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
                lw3.writerow(tmsMV)#write list of [timestamp, filepath, trans midi] to file
            else:
                lw3.writerow(sorted_MV)#write list of [trans midi] windows to file
            """
            DEPRECATED:
            Turns MIDI vecs into voicing classes, moving them all down so 0 < lowest voice < 11
            while sorted_MV[0] > 11:
                for m in range(len(sorted_MV)):
                    sorted_MV[m] -= 12
            """
            #write tranposed scale degree class vector to file
            lw.writerow(pcVector)
            
#############################################################################
def transPCVecs(twindow,relTo='Do'):
    '''
    DEPRECATED: less effective version of chordFinder()?
    
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
        

#csvTransposer('25 SDs prog probs 50ms.csv', '25 SDs prog probs 50msTRANS.csv')   
#chordFinder(50, relTo='Do',withTimeStamps=True)
#transPCVecs(800, relTo='La')
#transPCVecs(1200, relTo='La')
#transPCVecs(1200, relTo='Do')
#transPCVecs(800, relTo='La')