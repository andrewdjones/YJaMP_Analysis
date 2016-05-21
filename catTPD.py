from __future__ import absolute_import
from __future__ import print_function
import midi
import os
import numpy as np
from string import Template
import csv
import operator
import scipy.stats
import collections
import math
from numpy import log, log10
import six
from six.moves import range

def getProbsFromFreqs(DictionaryOfTallies):
    '''
    Simple script to turn a dict of tallies into a dict of normalized probabilities
    '''
    totalSum = 0.0
    dictOfProbs = {}
    for key, freq in six.iteritems(DictionaryOfTallies):
        totalSum += float(freq)
    for key, freq in six.iteritems(DictionaryOfTallies):
        dictOfProbs[key] = float(freq)/totalSum
    return dictOfProbs

def tallySDSets(cmin,probs=False):
    '''
    Takes output of chordFinder (csv of locally-transposed, 50ms scale degree vectors)
    Tallies up all the SD sets with cardinality >= cmin.
    If probs=False, gives tallies; if True, gives probs
    Outputs a csv of (SDset, tallies or probs)
    Returns a dict
    '''
    #Get the scale degree vector data csv
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind TS.csv','r',newline='\n'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row[0:12])#cut off any additional annotation past column 12
    #print(allMidiList)
    
    #Add the vectors to a counter
    SDSets = collections.Counter()
    for midvec in allMidiList:
        SDs = set([])#avoid multisets
        for n in range(12):
            if midvec[n] != '0.0':#any nonzero duration counts
                SDs.add(n)
        if len(SDs) >= cmin:#cardinality restriction
            SDSets[str(sorted(SDs))] += 1
            
    #sort by descending frequency and prepare for csv output
    if probs==True:
        SDSets_probs = getProbsFromFreqs(SDSets)
        sorted_SDSets = sorted(six.iteritems(getProbsFromFreqs(SDSets)),key=operator.itemgetter(1),reverse=True)
        fp = Template('50ms $cm SDSets Probs.csv')
    else:
        sorted_SDSets = sorted(six.iteritems(SDSets), key=operator.itemgetter(1),reverse=True)
        fp = Template('50ms $cm SDSets.csv')
    """Suppress csv writing, for now
    csvName = fp.substitute(cm=cmin)
    x = csv.writer(open(csvName, 'w',newline='\n'))
    for pair in sorted_SDSets:
        x.writerow([pair[0], pair[1]])
    """
    if probs==True: return SDSets_probs

def nAfterSDS(sds,numWin,supersets=False,probs='None',filtp=0.0):
    '''
    Takes a given scale degree set and tallies up what happens with next numWin time windows
    By default, doesn't count scale degree supersets, but it will consider all SDSS as equivalent 
    if supersets==True
    Also normalizes by unigram probabilities if probs='Rel'; just absolute, distance-based probs if probs='Abs'
    Tallies only if probs='None'
    filtp sets a unigram prob cutoff for stuff that gets counted.  Below it, no stats kept (not very helpful)
    If sds is a list, it assumes a single scale degree set and tracks what follows
    If sds is a tuple, first entry is chord to test; second entry is dict of manually-imposed categories
    '''
    
    #Get the scale degree vector data
    allPanes = csv.reader(open('sdcVecs 50Do 2xwind TS.csv','r',newline='\n'))
    allMidiList = []
    for row in allPanes:
        allMidiList.append(row)
    #print(allMidiList)
    
    #Start the tally
    #SDSetsT is a counter of the form SDSetsT[time distance][destination chord] = tallies
    SDSetsT = collections.Counter()
    for i, midvec in enumerate(allMidiList):
        #first, assemble each consecutive set for testing
        SDset = set([])
        for n in range(12):
            #print(n)
            if midvec[n] != '0.0':
                SDset.add(n)
                
        """
        if sds is a list, then it's a single scale degree set
        if sds is a tuple (sds, dict), look for all ocs that are in dict[sds]
        Ex: ('ii',{'ii': [[0,2,5],[0,2,5,9],[2,5,9]],'V': [[5,7,11],[2,5,7,11],[2,7,11]], 'I': [[0,4,7],[0,3,7],[0,4,11],[0,3,10],[0,4,7,11],[0,3,7,10]]})
        """
        if type(sds) == list:
            print('Running single origin chord mode')
            #if the chord being tested is of the right kind, assemble stats
            if sorted(SDset) == sds or (supersets == True and SDset.issuperset(sds) == True):
                j = i+1
                while j < i + numWin + 1 and j < len(allMidiList):#time constraints
                    if allMidiList[j][-1] != allMidiList[i][-1]:#if we change tracks, don't count progressions
                        break
                    nextSet = set([])#subsequent chord at time distance j
                    for m in range(12):
                        if allMidiList[j][m] != '0.0':
                            nextSet.add(m)
                    if len(nextSet) < 3:#skip small cardinality chords
                        j += 1
                        continue
                    try:
                        SDSetsT[j-i][str(sorted(nextSet))] += 1
                    except TypeError:
                        SDSetsT[j-i] = collections.Counter()
                        SDSetsT[j-i][str(sorted(nextSet))] += 1
                    j += 1
        elif type(sds) == tuple:
            print('Running manually-defined category mode')
            catDict = sds[1]#all of our categories in one dict, indexed by label
            ocs = catDict[sds[0]]#these are the possible origin chords in category
            #print('possible origin chords in category: ',ocs)
            #Same procedure as above, but allowing stats for any chord in category
            if sorted(SDset) in ocs:
                j = i+1
                while j < i + numWin + 1 and j < len(allMidiList):
                    if allMidiList[j][-1] != allMidiList[i][-1]:
                        break
                    nextSet = set([])
                    #print(allMidiList[j])
                    for m in range(12):
                        if allMidiList[j][m] != '0.0':
                            nextSet.add(m)
                    if len(nextSet) < 3:
                        j += 1
                        continue
                    #make sure to label destination chord with its proper category from catDict
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
    #for each time distance, turn absolute tallies into unigram-relative log probs
    if probs=='Rel':
        uni_probs = tallySDSets(3,probs=True)#unigram probs
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
                #calculate unigram-relative log probability
                logprob = log(probs_at_dist[sd])-log(uni_probs[sd])
                SDSetsT[row][sd] = logprob
                
    #Easy version: each distance gets its own probability normalization, no unigram business
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
            fileName = Template('$voic SDs prog rlogprobs 50ms.csv') if supersets == False else Template('$voic SDSS prog rlogprobs 50ms.csv')
        elif probs=='Abs':
            fileName = Template('$voic SDs prog probs 50ms.csv') if supersets == False else Template('$voic SDSS prog probs 50ms.csv')
        csvName = fileName.substitute(voic = str(sds))
    elif type(sds) == tuple:
        if probs=='None':
            fileName = Template('$voic SDs prog 50ms.csv') if supersets == False else Template('$voic SDSS prog 50ms.csv')
        elif probs=='Rel':
            fileName = Template('$voic SDs prog rlogprobs 50ms.csv') if supersets == False else Template('$voic SDSS prog rlogprobs 50ms.csv')
        elif probs=='Abs':
            fileName = Template('$voic SDs prog probs 50ms.csv') if supersets == False else Template('$voic SDSS prog probs 50ms.csv')
        csvName = fileName.substitute(voic = sds[0])
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(fieldnames)
    dw = csv.DictWriter(file, fieldnames)
    for row in SDSetsT:
        dw.writerow(SDSetsT[row])

def plot_dendrogram(model, **kwargs):
    '''
    taken from online example in sklearn fork
    turns hierarchical model into dendrogram (originally)
    for now, I have it set to return a linkage matrix, instead
    '''
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.datasets import load_iris
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances
    from matplotlib import pyplot as plt
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    return linkage_matrix
"""This part plots the dendrogram
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
"""

def agglomClusCat(distmat,k,crit):
    '''
    For all the TPD matrices captured by pairwise distmat, uses sklearn to hierarchically cluster
    k is number of clusters
    crit is criterion for fcluster ('distance' best option)
    '''
    import sklearn
    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster.hierarchy import fcluster
    from sklearn.datasets import load_iris
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances
    from matplotlib import pyplot as plt
    import itertools
    
    #put the calculated (generalized Manhattan) inter-matrix distances into array of floats
    diMat = []
    dists = csv.reader(open(distmat, 'r',newline='\n'))
    for row in dists:
        diMat.append(row)
    disArr = np.array(diMat)#pairwise dist mat as strings
    diArr = disArr.astype(float)#now as floats
    #distMat_cond = squareform(diArr)#turns redundant, square into condensed, triangular
    
    #set and fit the agglomerative clustering model
    mclus = AgglomerativeClustering(n_clusters = k, affinity='precomputed',linkage='complete')
    clusfit = mclus.fit(diArr)
    labels = clusfit.labels_
    #print(labels)
    
    #From PCA-based data, pull in the string names of chords in order
    chdnames = csv.reader(open('n10_PCA/562TPDmatrixSim kmed 200_n10PCA.csv', 'r',newline='\n'))
    #these for some other topN
    #chdnames = csv.reader(open('7470TPDmatrixSim kmed 50_n10PCA.csv', 'r',newline='\n'))
    #chdnames = csv.reader(open('2510TPDmatrixSim kmed 500_n10PCA.csv', 'r',newline='\n'))
    chdnamesit = []
    for row in chdnames:
        chdnamesit.append(row)
    chdnameslst = []
    for i,chd in enumerate(chdnamesit):
        if i<2: continue
        chdnameslst.append(chd[0])
    #print(chdnameslst)
    
    #now make a dendrogram and/or flat clustering assignments
    #plot_dendrogram(clusfit,labels=chdnameslst,show_leaf_counts=True,leaf_font_size=8,leaf_rotation=45)#labels=clusfit.labels_
    clusters = fcluster(plot_dendrogram(clusfit,labels=chdnameslst,show_leaf_counts=True,leaf_font_size=8,leaf_rotation=45),k,criterion=crit)
    assigns = []
    for i in range(200):
        assigns.append([clusters[i],chdnameslst[i]])
    sassigns = sorted(assigns,key=operator.itemgetter(0))
    
    #send out the leaf cluster membership data
    csvName = 'truncDend_memb_test.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    for row in sassigns:
        lw.writerow(row)
        
def sdsCatAssembler(assigncsv):
    '''
    takes output csv of agglomClusCat (flat hierarchical cluster assignments)
    puts it in scale degree set dict format suitable for nAfterSDS
    '''
    c = csv.reader(open(assigncsv, 'r',newline='\n'))
    cs = {}
    for row in c:
        if row[0] not in cs: cs[row[0]] = []
        chd = []
        #print(row[0].split(','))
        for j,char in enumerate(row[1].split(',')):
            if j == 0:
                chd.append(int(char[1:]))
                continue
            if j == len(row[1].split(','))-1:
                chd.append(int(char[0:-1]))
                break
            chd.append(int(char))
        #print(chd)
        cs[row[0]].append(chd)
    return cs

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
        
def PCAforYJaMP(oc,n,mode='Rel',sendData = False):
    '''
    takes temporal probability distributions (like from nAfterSDS, but transposed) in format
    (rows, cols) = (chords, time slices)
    reduces column basis to n principal components
    plots components
    outputs transformed data iff sendData == True
    (Rel and Abs refer to normalization, but really just control file naming consistency)
    '''
    import numpy
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    #Get the probability data
    if mode=='Rel':
        originpath = 'C:/Users/Andrew/workspace/DissWork/'+oc+' SDs prog rlogprobs 50msTRANS.csv'
        #originpath = 'C:/Users/Andrew/workspace/DissWork/'+oc+' SDs prog rlogprobs 50msTRANS.csv'
    elif mode=='Abs':
        originpath = 'C:/Users/Andrew/workspace/DissWork/'+oc+' SDs prog probs 50msTRANS.csv'
    #Load the succession data for oc from csv
    allDists = csv.reader(open(originpath,'r',newline='\n'))
    listOfRows = []
    for row in allDists:
        listOfRows.append(row)
    #get the list of slice distances for which there's data (probably 100, but allows others)
    slicedist = [int(x) for x in listOfRows[0][1:]]
    #print(slicedist)
    #turn the csv strings into floats to get slicedist-dim probability distributions
    for row in listOfRows:
        for j in range(len(row)):
            if row[j]=='':
                row[j]=0.0
    distprobs = []
    for i in range(1,len(listOfRows)):
        distprobs.append([float(x) for x in listOfRows[i][1:]])
        
    #convert into numpy array and run PCA
    probarr = numpy.array(distprobs)
    pca = PCA(n_components = n)
    pca.fit(probarr)
    #This sends out the transformed data
    if sendData:
        transformed_data = pca.fit(probarr).transform(probarr)
        print(transformed_data)
        #write the CSV
        if mode=='Rel':
            csvName = oc+' Rel ABBREV transformed data_test.csv'
        if mode=='Abs':
            csvName = oc +' Abs PCA-transformed data_test.csv'
        file = open(csvName, 'w',newline='\n')
        lw = csv.writer(file)
        for row in transformed_data:
            lw.writerow(row)
        #print(len(probarr[0]),len(transformed_data[0]))
    
    #plot principal components and variance capture
    #print(pca.components_)
    plt.subplot(121)
    for y in range(n):
        plt.plot(slicedist, pca.components_[y],label='PCA '+str(y+1)+', '+"{0:.3f}".format(pca.explained_variance_ratio_[y]))#all the distributions
    plt.legend(loc="upper left",bbox_to_anchor=(1.05, 1.))
    plt.title(str(oc)+' PCA')
    plt.xlabel('50ms time windows')
    plt.ylabel('Distance-based probability')
    #plt.axis([0,50,-1,1])#set axis dimensions
    #print what percentage of the variance is explained by each of the n components
    print((pca.explained_variance_ratio_))
    #display the plot
    plt.show()
    
#agglomClusCat('200 nDistMat AbsP Syntax Forwards_PCA.csv',140,'distance')
#nAfterSDS(('25',sdsCatAssembler('truncDend_memb.csv')),100,supersets=False,probs='Abs',filtp=0.0)      
#csvTransposer('25 SDs prog probs 50ms.csv', '25 SDs prog probs 50msTRANS.csv')   
#PCAforYJaMP('25',5,mode='Abs',sendData=False)
'''        
To formalize cat membership, collapse cats across corpus, and get new, simplified TPDs
1. Get cat membership from agglom clus into simple csv (agglomClusCat)
2. Use csvs to define dicts for nAfterSDS (sdsCatAssembler)
3. Run nAfterSDS on each category separately to get TPDs relative to one another (nAfterSDS)
4. To get PCA basis reductions of the nAfterSDS output data, transpose the csv (csvTransposer)
5. Run PCA to reduce basis, plot components, and output transformed data (PCAforYJaMP)
'''