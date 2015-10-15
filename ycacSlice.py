from music21 import *
from sets import *
import csv
import pickle
import os
from music21.pitch import AccidentalException
import operator

#First, tell python to look in the directory for all the piano roll midi files
path = 'C:/Python27/Lib/site-packages/music21/corpus/YCAC_all/'
listing = os.listdir(path)

def ycacVoicings():
    """
    Looks through the ycac csv slices and tallies up the most common voicings.
    """
    from string import Template
    sliceCount = {}
    problems = 0
    for f in listing:
        fileName = path + f
        print 'Running ', fileName
        #Load the pickled slices that have not been bass-normalized into types
        openCSV = open(fileName,'r')
        allSlices = csv.reader(openCSV)
        listOfRows = []
        for row in allSlices:
            listOfRows.append(row)
        for i, row in enumerate(listOfRows):
            #print row
            if i == 0:
                continue
            if i == len(listOfRows) - 1:
                break
            theChord = row[1]
            chordList = theChord.split('chord.Chord ')[1].split(' ')
            chordNotes = []
            for j, noteThing in enumerate(chordList):
                if j == len(chordList) - 1:
                    break
                chordNotes.append(noteThing)
            lastNote = chordList[-1].split('>')[0]
            chordNotes.append(lastNote)
            try:
                someChord = chord.Chord(chordNotes)
            except AccidentalException:
                problems += 1
                continue
            midiChord = [p.midi for p in someChord.pitches]
            sorted_slicey = sorted(midiChord)
            sliceyBass = sorted_slicey[0]
            #find out the voicing this slice is
            slicey_type = [n - sliceyBass for n in sorted_slicey]
            try:
                sliceCount[str(slicey_type)] += 1
            except KeyError:
                sliceCount[str(slicey_type)] = 1
    #tally up the frequencies for each chord
    sorted_chordCount = sorted(sliceCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print 'All the slices!',sorted_chordCount
    #export the tally as a csv file
    csvName = 'ycacVoicings.csv'
    x = csv.writer(open(csvName, 'wb'))
    for pair in sorted_chordCount:
        x.writerow([pair[0], pair[1]])

def whatsNAfter(voicing,dist):
    """
    This takes a voicing and tells you what comes next after 1, 2, ..., dist slices.
    """
    from string import Template
    sliceCount = {}
    problems = 0
    for f in listing:
        fileName = path + f
        #Load the pickled slices that have not been bass-normalized into types
        openCSV = open(fileName,'r')
        allSlices = csv.reader(openCSV)
        listOfRows = []
        for row in allSlices:
            listOfRows.append(row)
        for i, row in enumerate(listOfRows):
            #print row
            if i == 0:
                continue
            if i == len(listOfRows) - 1:
                break
            theChord = row[1]
            chordList = theChord.split('chord.Chord ')[1].split(' ')
            chordNotes = []
            for j, noteThing in enumerate(chordList):
                if j == len(chordList) - 1:
                    break
                chordNotes.append(noteThing)
            lastNote = chordList[-1].split('>')[0]
            chordNotes.append(lastNote)
            #print chordNotes
            try:
                someChord = chord.Chord(chordNotes)
            except AccidentalException:
                problems += 1
                continue
            midiChord = [p.midi for p in someChord.pitches]
            sorted_slicey = sorted(midiChord)
            sliceyBass = sorted_slicey[0]
            #find out the voicing this slice is
            slicey_type = [n - sliceyBass for n in sorted_slicey]
            #print slicey_type
            if slicey_type != voicing:
                continue
            j = 0
            while j < dist:
                #Get the next slice j away, unless out of range
                if i + j + 1 > len(listOfRows) - 1:
                    break
                theNextRow = listOfRows[i+j+1]
                theNextChord = theNextRow[1]
                nextChordList = theNextChord.split('chord.Chord ')[1].split(' ')
                nextChordNotes = []
                for k, noteThing in enumerate(nextChordList):
                    if k == len(nextChordList) - 1:
                        break
                    nextChordNotes.append(noteThing)
                lastNote = nextChordList[-1].split('>')[0]
                nextChordNotes.append(lastNote)
                try:
                    someNextChord = chord.Chord(nextChordNotes)
                except AccidentalException:
                    problems += 1
                    continue
                midiNextChord = [p.midi for p in someNextChord.pitches]
                nextSlice = sorted(midiNextChord)
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
                #Reject if the result is too small
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
    fileName = Template('$voic nonselfset voicing paths ycac.csv')
    csvName = fileName.substitute(voic = str(voicing))
    file = open(csvName, 'wb')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in sliceCount:
        dw.writerow(sliceCount[row])
    print 'slice count',sliceCount
    print 'problem chords', problems
        
        
# ycacVoicings()       
whatsNAfter([0,4,10],50)