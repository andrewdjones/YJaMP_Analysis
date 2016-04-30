from __future__ import absolute_import
from __future__ import print_function
#from music21 import *
#from sets import *
import csv
import six
from six.moves import range
#import pickle
import os
#from music21.pitch import AccidentalException
#import operator

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
        print('Running ', fileName)
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
    sorted_chordCount = sorted(six.iteritems(sliceCount), key=operator.itemgetter(1), reverse=True)
    print('All the slices!',sorted_chordCount)
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
    print('slice count',sliceCount)
    print('problem chords', problems)
      
#Have: "succession" data for trans pcsets with sd in bass for 50 consecutive slices (as csvs)
#Want: to turn (project?) a collection of those 50-dim distribution vectors into a lower-dim basis of temporal regimes
#Use: scikit PCA, which takes a numpy array and fits PCA on it.  Can provide num components to fit to... should I?
def PCAforYCAC(oc,n):
    """
    oc is an origin chord successions csv file (tallied by IQ from YCAC) with normalized probs out to 50 slices
    relies on scikit-learn
    outputs n basis components (and successions rotated into new basis coords?)
    """
    import numpy
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    originpath = 'C:/Users/Andrew/Documents/IQ Regime Paper/successions/'+oc+'.csv'
    #Load the succession data for oc from csv
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
    #print(distprobs[1])
    #convert into numpy array for PCA
    probarr = numpy.array(distprobs)
    #run PCA; don't forget to set n_components
    pca = PCA(n_components = n)
    pca.fit(probarr)
    print(pca.components_)
    #plot however many components you want to see
    for y in range(n):
        plt.plot(slicedist, pca.components_[y])#all the distributions
    #plt.axis([0,50,-1,1])#set axis dimensions
    #print what percentage of the variance is explained by each of the n components
    print((pca.explained_variance_ratio_))
    #display the plot
    plt.show()
    
def PCAforYJaMP(oc,n,mode='Rel'):
    import numpy
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    if mode=='Rel':
        originpath = 'C:/Users/Andrew/workspace/DissWork/'+oc+' SDs prog rlogprobs 50msTRANS.csv'
    elif mode=='Abs':
        originpath = 'C:/Users/Andrew/workspace/DissWork/'+oc+' SDs prog probs 50msTRANS.csv'
    #Load the succession data for oc from csv
    allDists = csv.reader(open(originpath,'r',newline='\n'))
    listOfRows = []
    for row in allDists:
        listOfRows.append(row)
    #get the list of slice distances for which there's data (probably 50)
    slicedist = [int(x) for x in listOfRows[0][1:]]
    #print(slicedist)
    #turn the csv strings into floats to get slicedist-dim probability distributions
    for row in listOfRows:
        for j in range(len(row)):
            if row[j]=='':
                row[j]=0
    #print(listOfRows[1])
    distprobs = []
    for i in range(1,len(listOfRows)):
        distprobs.append([float(x) for x in listOfRows[i][1:]])
    #print(distprobs[1])
    #convert into numpy array for PCA
    probarr = numpy.array(distprobs)
    #run PCA; don't forget to set n_components
    pca = PCA(n_components = n)
    pca.fit(probarr)
    '''This sends out the transformed data '''
    transformed_data = pca.fit(probarr).transform(probarr)
    print(transformed_data)
     #write the CSV
    csvName = oc+' Rel transformed data.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    for row in transformed_data:
        lw.writerow(row)
    print(len(probarr),len(transformed_data))
    #print(pca.components_)
    #plot however many components you want to see
    plt.subplot(121)
    for y in range(n):
        plt.plot(slicedist, pca.components_[y],label='PCA '+str(y+1)+', '+str(pca.explained_variance_ratio_[y]))#all the distributions
    plt.legend(loc="upper left",bbox_to_anchor=(1.05, 1.))
    plt.title(str(oc)+' PCA')
    plt.xlabel('50ms time windows')
    plt.ylabel('Unigram-relative log probability')
    #plt.axis([0,50,-1,1])#set axis dimensions
    #print what percentage of the variance is explained by each of the n components
    print((pca.explained_variance_ratio_))
    #display the plot
    plt.show()

def PCAexample():
    '''
    A bad example for putting in a "simple" discussion in chapter 4
    '''
    import numpy
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    someChords = []
    for i in range(5):
        someChords.append([2*i,3*i+2,i+5,i+9])
    print(someChords)
    #convert into numpy array for PCA
    probarr = numpy.array(someChords)
    #run PCA; don't forget to set n_components
    pca = PCA(n_components = 3)
    pca.fit(probarr)
    print(pca.components_)
    transformed_data = pca.fit(probarr).transform(probarr)
    print(transformed_data)
    plt.subplot(121)
    for y in range(3):
        plt.plot([0,1,2,3], pca.components_[y],label='PCA '+str(y+1)+', '+str(pca.explained_variance_ratio_[y]))#all the distributions
    for y in range(5):
        plt.plot([0,1,2,3],someChords[y])
    plt.legend(loc="upper left",bbox_to_anchor=(1.05, 1.))
    plt.title(' PCA')
    plt.xlabel('Sequential chord step')
    plt.ylabel('Pitch')
    #plt.axis([0,50,-1,1])#set axis dimensions
    #print what percentage of the variance is explained by each of the n components
    print((pca.explained_variance_ratio_))
    #display the plot
    plt.show()
    
"""
Scale up PCA:
Take each top-100 scale degree set
Run PCA on its TPDs
Score (positive) PCA1 and (opposite-signed) PCA2
Track the top PCA2-scored chords
Weighted graph similarity/clustering?
"""
    
# ycacVoicings()       
#PCAforYCAC('A_(EA)',5)
PCAforYJaMP('V',5,mode='Rel')
#PCAexample()