import midi
import music21
from imghdr import what
import os
import numpy
from string import Template
import csv
import operator
import scipy.stats
import collections
import matplotlib.pyplot as plt
from music21.ext.jsonpickle.util import itemgetter


path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIunquant/'
listing = os.listdir(path)
#testFile = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIQuantized/Alex_1_1.mid'
#testFile = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIQuantized/Julian_5_6.mid'

def getProbsFromFreqs(DictionaryOfTallies):
    totalSum = 0.0
    dictOfProbs = {}
    for key, freq in DictionaryOfTallies.iteritems():
        totalSum += float(freq)
    for key, freq in DictionaryOfTallies.iteritems():
        dictOfProbs[key] = float(freq)/totalSum
    return dictOfProbs
    

def getInteronsets(mode='unquant'):
    """
    Hunts through all the tracks (as of Sept 2015) and tallies interonsets
    Does this with unquantized (default) or quantized (any other mode) files
    Counts each verticality as one onset.  Un-quant midi won't have any, right?
    """
    diffCount = collections.Counter()
    for j, testFile in enumerate(listing):
        #if j > 0:
            #break
        pattern = midi.read_midifile(path + testFile)
        #this line makes each tick count cumulative
        pattern.make_ticks_abs()
        #print pattern.resolution, testFile
        for i,track in enumerate(pattern):
            if len(track) < 50:
                continue
            #how many tempo events?
            tempEvents = 0
            noteEvents = 0
            for j, thing in enumerate(track):
                if thing.__class__ == midi.events.NoteOnEvent:
                    noteEvents += 1
                if thing.__class__ == midi.events.SetTempoEvent:
                    mspt = thing.get_mpqn() / pattern.resolution
                    #print mspt
                    tempEvents +=1
            if noteEvents == 0:
                continue
            if tempEvents == 0:
                mspt = 500000 / pattern.resolution
                #NOTE: this is MICRO seconds per tick, not milli
            if tempEvents > 1:
                print 'hey, extra tempo event?'
                break
            for j, thing in enumerate(track):
                #if i > 50:
                    #break
                if j == 0:
                    continue
                if thing.__class__ == midi.events.NoteOnEvent:
                    prevTime = 'null'
                    k = j-1
                    while prevTime == 'null':
                        if track[k].__class__ == midi.events.NoteOnEvent:
                            prevTime = track[k].tick
                        k -= 1
                        if k < 0:
                            break
                    if prevTime == 'null':
                        continue
                    timeDiff = round((thing.tick - prevTime)*mspt/1000)
                    #time diff is now to nearest MILLIsecond
                    #print thing.tick, prevTime
                    diffCount[str(timeDiff)] += 1
    #that should do it.  Now package up a csv of the counter dist
    interonsets = []
    counts = []
    diffCount_list = []
    for key,count in diffCount.iteritems():
        diffCount_list.append([float(key),count])
    sorted_diffCount = sorted(diffCount_list)
    print sorted_diffCount
    for key,count in sorted_diffCount:
        if count < 50:
            continue
        interonsets.append(float(key))
        counts.append(count)
    plt.plot(interonsets,counts)
    print  'plot did not hang'
    plt.xlabel('interonset (millisecs)')
    plt.ylabel('counts')
    plt.savefig('C:/Users/Andrew/Documents/DissNOTCORRUPT/testio.png')
    plt.show()
    '''
    csvName = 'unquant_interonsets.csv'
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    for key, count in diffCount.iteritems():
        lw.writerow([key,count])
    '''

"""
OK, things to do:
1. Strip out tracks which contain no notes (DONE)
2. From the remaining tracks, determine the number of milli(micro?)secs per tick (DONE)
3. Get a list of the note event in absolute ticks (DONE)
4. Translate those absolute ticks into elapsed milli/microseconds (DONE)
5. Figure out the distribution of note lengths; choose a good one for windowing (SKIPPED)
6. Make a list of time slices in which we can look for notes (DONE)
7. Export time slices as a csv.  Mimic ycac?
"""
def midiTimeWindows(windowWidth):
    tempEventDict = {}
    #numTunes = 0
    #numShortTracks = 0
    #numTracks = 0
    #we'll make a list: [millisecs at end of window, music21 chord, set of midi numbers,  pcs in order, file name]
    msandmidi = []
    for j, testFile in enumerate(listing):
        #print testFile
        if j > 0:
            #continue
            break
        #numTunes += 1
        #for use with import midi
        pattern = midi.read_midifile(path + testFile)
        #this line makes each tick count cumulative
        pattern.make_ticks_abs()
        #print pattern.resolution, testFile
        for i,track in enumerate(pattern):
            #numTracks += 1
            if len(track) < 50:
                #numShortTracks += 1
                continue
            #how many tempo events?
            tempEvents = 0
            noteEvents = 0
            for i, thing in enumerate(track):
                if thing.__class__ == midi.events.NoteOnEvent:
                    noteEvents += 1
                if thing.__class__ == midi.events.SetTempoEvent:
                    mspt = thing.get_mpqn() / pattern.resolution
                    #print mspt
                    tempEvents +=1
            if noteEvents == 0:
                #numShortTracks += 1
                continue
            if tempEvents == 0:
                mspt = 500000 / pattern.resolution
            if tempEvents > 1:
                print 'hey, extra tempo event?'
                break
            #windowWidth = 100 #number of milliseconds wide each bin will be
            windows = [set([])]
            for i, thing in enumerate(track):
                #if i > 50:
                    #break
                #print thing
                j = int(numpy.floor(thing.tick * mspt/1000 * 1/windowWidth)) #which window?
                if len(windows) < j + 2:
                    for n in range(j+2 - len(windows)):
                        newWindow = set([])
                        for item in windows[-1]:
                            newWindow.add(item)
                        windows.append(newWindow)
                #print windows
                '''
                SOMETHING WRONG IN WHILE??
                while len(windows) < j + 1: #window might not exist yet
                    #print windows[-1]
                    print windows
                    windows.append(windows[-1]) #copy contents of last window: all still sound!
                '''
                if thing.__class__ == midi.events.NoteOnEvent:
                    if thing.get_velocity() == 0: #weirdly-notated noteOffEvent
                        windows[j+1].remove(thing.get_pitch()) #remove from NEXT window
                        #print 'removing pitch',thing.get_pitch(),'from window',j,windows[j]
                    else:
                        windows[j].add(thing.get_pitch()) #note turns on in proper window
                        windows[j+1].add(thing.get_pitch()) #also add it to next window
                        #print 'adding pitch',thing.get_pitch(),'to window',j,windows[j]
                if thing.__class__ == midi.events.NoteOffEvent:
                    windows[j+1].remove(thing.get_pitch()) #remove from window
                    #print 'removing pitch',thing.get_pitch(),'from window',j
            for j in range(len(windows)):
                if len(windows[j]) == 0:#skip the empty windows
                    continue
                pitchClasses = set([])
                for mid in windows[j]:
                    if mid%12 in pitchClasses:
                        continue
                    pitchClasses.add(mid%12)
                msandmidi.append([(j+1)*windowWidth,windows[j],sorted(pitchClasses),testFile])
    #print msandmidi
    #package up a csv
    """
    for j, testFile in enumerate(listing):
        #print testFile
        if j < 51:
            continue
            #break
        #numTunes += 1
        #for use with import midi
        pattern = midi.read_midifile(path + testFile)
        #this line makes each tick count cumulative
        pattern.make_ticks_abs()
        #print pattern.resolution, testFile
        for i,track in enumerate(pattern):
            #numTracks += 1
            if len(track) < 50:
                #numShortTracks += 1
                continue
            #how many tempo events?
            tempEvents = 0
            noteEvents = 0
            for i, thing in enumerate(track):
                if thing.__class__ == midi.events.NoteOnEvent:
                    noteEvents += 1
                if thing.__class__ == midi.events.SetTempoEvent:
                    mspt = thing.get_mpqn() / pattern.resolution
                    #print mspt
                    tempEvents +=1
            if noteEvents == 0:
                #numShortTracks += 1
                continue
            if tempEvents == 0:
                mspt = 500000 / pattern.resolution
            if tempEvents > 1:
                print 'hey, extra tempo event?'
                break
            #windowWidth = 100 #number of milliseconds wide each bin will be
            windows = [set([])]
            for i, thing in enumerate(track):
                #if i > 50:
                    #break
                #print thing
                j = int(numpy.floor(thing.tick * mspt/1000 * 1/windowWidth)) #which window?
                if len(windows) < j + 2:
                    for n in range(j+2 - len(windows)):
                        newWindow = set([])
                        for item in windows[-1]:
                            newWindow.add(item)
                        windows.append(newWindow)
                #print windows
                '''
                SOMETHING WRONG IN WHILE??
                while len(windows) < j + 1: #window might not exist yet
                    #print windows[-1]
                    print windows
                    windows.append(windows[-1]) #copy contents of last window: all still sound!
                '''
                if thing.__class__ == midi.events.NoteOnEvent:
                    if thing.get_velocity() == 0: #weirdly-notated noteOffEvent
                        windows[j+1].remove(thing.get_pitch()) #remove from NEXT window
                        #print 'removing pitch',thing.get_pitch(),'from window',j,windows[j]
                    else:
                        windows[j].add(thing.get_pitch()) #note turns on in proper window
                        windows[j+1].add(thing.get_pitch()) #also add it to next window
                        #print 'adding pitch',thing.get_pitch(),'to window',j,windows[j]
                if thing.__class__ == midi.events.NoteOffEvent:
                    windows[j+1].remove(thing.get_pitch()) #remove from window
                    #print 'removing pitch',thing.get_pitch(),'from window',j
            for j in range(len(windows)):
                if len(windows[j]) == 0:#skip the empty windows
                    continue
                pitchClasses = set([])
                for mid in windows[j]:
                    if mid%12 in pitchClasses:
                        continue
                    pitchClasses.add(mid%12)
                msandmidi.append([(j+1)*windowWidth,windows[j],sorted(pitchClasses),testFile])
    """
    #print msandmidi
    #package up a csv
    fieldnames = ['ms window end','MIDI set','ordered PCs','file']
    fileName = Template('$siz window tslices 1122.csv')
    csvName = fileName.substitute(siz = str(windowWidth))
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    for row in msandmidi:
        lw.writerow(row)
    return msandmidi

def getCollections():
    #go from 50ms to 60*1000 ms by 50s increments
    windowSize = 25
    EntropyatSize = []
    while windowSize < 60000:
        windowSize = windowSize*2
        print windowSize
        msandmidi = midiTimeWindows(windowSize)
        PCsetTally = {}
        for i, row in enumerate(msandmidi):
            if i == 0:
                continue
            try:
                PCsetTally[str(row[1])] += 1
            except KeyError:
                PCsetTally[str(row[1])] = 1
        distProbs = getProbsFromFreqs(PCsetTally)
        distProbsList = []
        #now write the body of the table
        #use a different CSV writer object
        for key,value in distProbs.iteritems():
            distProbsList.append(value)
        distEntropy = scipy.stats.entropy(distProbsList)
        EntropyatSize.append([windowSize,distEntropy,len(distProbsList)])
    csvName ='2 window midi entropy scales.csv'
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    for row in EntropyatSize:
        lw.writerow(row)
    
            
#getCollections()
#midiTimeWindows(100)
#getInteronsets()
