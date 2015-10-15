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
def midiTimeWindows(windowWidth,incUnit,solos=all):
    #windowWidth is obvious; incUnit how large the window slide step is
    #numTunes = 0
    #numShortTracks = 0
    #numTracks = 0
    #if solos != all:
        #listing = [solos]
    #we'll make a list: [millisecs at end of window, music21 chord, set of midi numbers,  pcs in order, file name]
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
                print 'hey, extra tempo event?'
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
                    for j in range(len(windows)):
                        #deal with note on events case
                        #put it in each window
                        if j*incUnit < absTicks < j*incUnit + windowWidth:
                            windows[j][thing.get_pitch()] += 1
                        if j*incUnit > absTicks:
                            #first too-late window
                            for k in range(j,len(windows)):
                                #Add to all remaining; we'll turn it off later
                                windows[k][thing.get_pitch()] += 1
                            break
                #deal with note off events cases
                elif thing.__class__ == midi.events.NoteOnEvent and thing.get_velocity() == 0:
                    for j in range(len(windows)):
                        if j*incUnit > absTicks:
                            #first window AFTER
                            for k in range(j,len(windows)):
                                #turn off in all windows after so it's not counted
                                del windows[k][thing.get_pitch()]
                            break
                elif thing.__class__ == midi.events.NoteOffEvent:
                    for j in range(len(windows)):
                        if j*incUnit > absTicks:
                            #first window AFTER
                            for k in range(j,len(windows)):
                                #turn off in all windows after so it's not counted
                                del windows[k][thing.get_pitch()]
                            break
            for j in range(len(windows)):
                if sum(windows[j].values()) == 0:#skip the empty windows
                    continue
                '''
                pitchClasses = set([])
                for mid in windows[j]:
                    if windows[j][mid] == 0:
                        continue
                    if mid%12 in pitchClasses:
                        continue
                    pitchClasses.add(mid%12)
                '''
                msandmidi.append([(j)*incUnit,windows[j]])
    #print msandmidi
    #package up a csv
    #print msandmidi
    '''
    #package up a csv
    fieldnames = ['ms window end','MIDI multiset','ordered PCs','file']
    fileName = Template('$siz $inc ms inc 1122.csv')
    csvName = fileName.substitute(siz = str(windowWidth), inc = str(incUnit))
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    for row in msandmidi:
        lw.writerow(row)
    '''
    return msandmidi


def entrop(solo=all):
    #go from 50ms to 60*1000 ms by doubling
    windowSize = 25
    EntropyatSize = []
    while windowSize < 60000:
        windowSize = windowSize*2
        print windowSize
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
        entropies = []
        for i, row in enumerate(msandmidi):
            if i == 0:
                continue
            pcVector = []
            for j in range(12):
                pcVector.append(0.01)
            for mid, counts in row[1].iteritems():
                pcVector[mid%12] += counts
            entropies.append(scipy.exp2(scipy.stats.entropy(pcVector,base=2)))
            #print windowSize,pcVector, entropies[-1]
        EntropyatSize.append([windowSize,scipy.average(entropies)])
        #now write the body of the table
    if solo != all:
        fileName = Template('$sol overlap window pc avg ent.csv')
        csvName =fileName.substitute(sol = solo.split('.')[0])
    else:
        csvName ='overlap window pc avg entropy.csv'
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    for row in EntropyatSize:
        lw.writerow(row)
        
def slidingEntropy(solo,windowSize, incUnit):
    """
    GOAL: input solo and windowSize; outputs entropy of pc vector as window increments
    """
    msandmidi  = midiTimeWindows(windowSize, incUnit, solos = solo)
    entropies = []
    for i, row in enumerate(msandmidi):
        if i == 0:
            continue
        pcVector = []
        for j in range(12):
            pcVector.append(0.01)
        for mid, counts in row[1].iteritems():
            pcVector[mid%12] += counts
        entropies.append([row[0],scipy.exp2(scipy.stats.entropy(pcVector,base=2))])
    fileName = Template('$sol overlap window pc ent.csv')
    csvName =fileName.substitute(sol = solo.split('.')[0])
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    for row in entropies:
        lw.writerow(row)
              
#entrop(solo='Alex_6_6_blueingreen.mid')
slidingEntropy('Alex_6_6_blueingreen.mid', 1000, 25)
#midiTimeWindows(200, 25)