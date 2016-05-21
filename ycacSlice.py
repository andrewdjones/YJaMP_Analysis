from __future__ import absolute_import
from __future__ import print_function
from music21 import *
#from sets import *
import csv
import six
from six.moves import range
#import pickle
import os
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform

#First, tell python to look in the directory for all the YCAC midi files
path = 'C:/Python27/Lib/site-packages/music21/corpus/YCAC_all/'
listing = os.listdir(path)

def ycacVoicings():
    """
    Looks through the ycac csv slices and tallies up the most common voicings.
    """
    from string import Template
    sliceCount = {}#dict of [voicing slices] = tally
    problems = 0#things that don't behave
    
    #for each csv, process the legacy format and tally stats
    for f in listing:
        #Get the raw csv rows
        fileName = path + f
        print('Running ', fileName)
        openCSV = open(fileName,'r')
        allSlices = csv.reader(openCSV)
        listOfRows = []
        for row in allSlices:
            listOfRows.append(row)
            
        #extract pitches from weird format    
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
            
            #If we get here, music21 recognizes those pitches as a chord voicing we can tally
            midiChord = [p.midi for p in someChord.pitches]
            sorted_slicey = sorted(midiChord)
            sliceyBass = sorted_slicey[0]
            #find out the voicing this slice is by moving the bass note to midi 0
            slicey_type = [n - sliceyBass for n in sorted_slicey]
            try:
                sliceCount[str(slicey_type)] += 1
            except KeyError:
                sliceCount[str(slicey_type)] = 1
                
    #sort the frequencies for each chord and output the tally as a csv
    sorted_chordCount = sorted(six.iteritems(sliceCount), key=operator.itemgetter(1), reverse=True)
    #print('All the slices!',sorted_chordCount)
    csvName = 'ycacVoicings.csv'
    x = csv.writer(open(csvName, 'w',newline='\n'))
    for pair in sorted_chordCount:
        x.writerow([pair[0], pair[1]])

def whatsNAfter(voicing,dist):
    """
    This takes a voicing and tells you what comes next after 1, 2, ..., dist slices.
    YCAC analog to (better documented, more complete) routine nAfterSDS (used primarily for YJaMP)
    """
    from string import Template
    sliceCount = {}#will be a dict of dicts sliceCount[destination chord][distance] = tally
    problems = 0#hopefully 0; data that has formatting issues
    for f in listing:
        
        #Get raw csv rows
        fileName = path + f
        openCSV = open(fileName,'r')
        allSlices = csv.reader(openCSV)
        listOfRows = []
        for row in allSlices:
            listOfRows.append(row)
            
        #extract data from weird format
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
            
            #if we make it here, music21 can recognize the chord
            midiChord = [p.midi for p in someChord.pitches]
            sorted_slicey = sorted(midiChord)
            sliceyBass = sorted_slicey[0]
            #find out the voicing this slice is
            slicey_type = [n - sliceyBass for n in sorted_slicey]
            #print slicey_type
            if slicey_type != voicing:
                continue
            
            #if we make it here, the voicing is of the type we'd like to tally stats for
            j = 0
            while j < dist:
                #Get the next slice j away, unless out of range
                if i + j + 1 > len(listOfRows) - 1:
                    break
                #more kludgy legacy format, thanks to CW and IQ
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
                
                #now set up a series of rejection conditions
                #If diffToggle == 0, the next slice is skipped
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
                        
                #if it passes all rejection conditions
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
                
    #now put the dict of dicts in some kind of csv table
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
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in sliceCount:
        dw.writerow(sliceCount[row])
    print('slice count',sliceCount)
    print('problem chords', problems)
      
def PCAforYCAC(oc,n):
    """
    oc is an origin chord successions csv file (tallied by IQ from YCAC) with normalized probs out to 50 slices
    outputs n basis components (and successions rotated into new basis coords?)
    """
    import numpy
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    #Get ycac data
    originpath = 'C:/Users/Andrew/Documents/IQ Regime Paper/successions/'+oc+'.csv'
    allDists = csv.reader(open(originpath,'r'))
    listOfRows = []
    for row in allDists:
        listOfRows.append(row)
    #get the list of slice distances for which there's data (probably 50)
    slicedist = [int(x) for x in listOfRows[0][1:]]
    #print(slicedist)
    #turn the csv strings into floats to get slicedist-dim probability distributions
    distprobs = []
    for i in range(1,len(listOfRows)-1):
        distprobs.append([float(x) for x in listOfRows[i][1:]])

    #convert distprobs into numpy array and run PCA
    probarr = numpy.array(distprobs)
    pca = PCA(n_components = n)
    pca.fit(probarr)
    print(pca.components_)
    
    #plot principal components and variance capture
    plt.subplot(121)
    for y in range(n):
        plt.plot(slicedist, pca.components_[y],label='PCA '+str(y+1)+', '+"{0:.3f}".format(pca.explained_variance_ratio_[y]))#all the distributions
    plt.legend(loc="upper left",bbox_to_anchor=(1.05, 1.))
    plt.title(str(oc)+' PCA')
    plt.xlabel('Salami slices')
    plt.ylabel('Distance-based probability')
    #plt.axis([0,50,-1,1])#set axis dimensions
    #print what percentage of the variance is explained by each of the n components
    print((pca.explained_variance_ratio_))
    #display the plot
    plt.show()
    
#ycacVoicings()       
#PCAforYCAC('A_(EA)',5)