import midi
import os
from string import Template
import csv

def chordSmusher(wsize):
    """
    Hunt through raw midi data in file path
    Whenever a note onset has subsequent note onsets within wsize, smush them together
    Repeat process until encountering an interonset > wsize
    That smushed entity is a chord; put it in a consecutive list and move on.
    Outputs: List of smushed chords with their onset times (in milliseconds)
    """
    path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/MIDIunquant/'
    listing = os.listdir(path)
    smushedChords = []
    tmsList = []
    for testFile in listing:
        #read the midi
        pattern = midi.read_midifile(path + testFile)
        #this line makes each tick count cumulative
        pattern.make_ticks_abs()
        #print pattern.resolution, testFile
        for track in pattern:
            #skip any weird pedalling tracks
            if len(track) < 50:
                continue
            #how many tempo events? We'll need this to get real timestamps.
            tempEvents = 0
            noteEvents = 0
            for thing in track:
                if thing.__class__ == midi.events.NoteOnEvent:
                    noteEvents += 1
                if thing.__class__ == midi.events.SetTempoEvent:
                    #NOTE: this is MICROseconds per tick
                    mcspt = thing.get_mpqn() / pattern.resolution
                    tempEvents +=1
            if noteEvents == 0:
                continue
            if tempEvents == 0:
                mcspt = 500000 / pattern.resolution #MICRO seconds per tick
            if tempEvents > 1:
                #haven't encountered these, but good to keep an eye out
                print 'hey, extra tempo event?'
                break
            #convert 50ms into ticks for the given track
            wsizeTicks = 1000*wsize/mcspt
            #tells code how many notes were smushed and should be skipped
            skipToggle = 0
            for j, thing in enumerate(track):
                if j == 0:
                    continue
                if thing.__class__ == midi.events.NoteOnEvent:
                    if skipToggle != 0:#that is, if we need to skip some
                        skipToggle -= 1
                        continue
                    currentIO = 0
                    currentTick = thing.tick
                    k = j+1
                    midiList = set([])
                    midiList.add(thing.get_pitch())
                    while currentIO < wsizeTicks:
                        if k >= len(track):
                            break
                        if track[k].__class__ != midi.events.NoteOnEvent:
                            k += 1
                            continue
                        currentIO = track[k].tick - currentTick#interonset in ticks
                        if currentIO < wsizeTicks:
                            midiList.add(track[k].get_pitch())
                            skipToggle += 1
                        k += 1
                    smushedChords.append(midiList)
                    tmsList.append(currentTick*mcspt/1000)
    #now stick the timestamp (millisecs) and midi notes in a csv
    fileName = Template('$wsiz smushed tms and chords.csv')
    csvName = fileName.substitute(wsiz = str(wsize))
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    for j,row in enumerate(smushedChords):
        rowList = [tmsList[j]]
        for setelem in row:
            rowList.append(setelem)
        lw.writerow(rowList)

chordSmusher(50)