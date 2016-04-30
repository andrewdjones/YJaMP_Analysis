from __future__ import absolute_import
from __future__ import print_function
import csv
import six
from six.moves import range
#import pickle
import os
import numpy as np
import random
from scipy.spatial import distance
from scipy.spatial.distance import pdist, cdist, squareform

def getProbsFromFreqs(DictionaryOfTallies):
    totalSum = 0.0
    dictOfProbs = {}
    for key, freq in six.iteritems(DictionaryOfTallies):
        totalSum += float(freq)
    for key, freq in six.iteritems(DictionaryOfTallies):
        dictOfProbs[key] = float(freq)/totalSum
    return dictOfProbs

def cluster(distances, k=3):

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)
   
    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point. 
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        
    return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    #print(costs,costs.argmin(axis=0, fill_value=10e9))
    return costs.argmin(axis=0, fill_value=10e9)

def TPDentropyCluster(fld,k,emRow = 'reject'):
    '''
    Input: the collection of sds temporal probability distributions (TPDs), abs(P) vs. time
    For each time window, calculate the entropy of the origin chord's tpds
    Output a vector with one entropy entry per time window
    Run k-medoids on entropy vectors to cluster by "chord regime type"
    (i.e., does sds participate in mostly local, mostly long range, some syntactic progs, etc.)
    NB: k-medoids converges LOCALLY
    '''
    import scipy.stats
    listing = os.listdir(fld)
    entVecList = []
    sdsList = []
    for f in listing:
        address = fld + f
        allDists = csv.reader(open(address,'r',newline='\n'))
        listOfRows = []
        #each row is a time window
        for row in allDists:
            listOfRows.append(row)#should be 100 of these
        emptyRows = 0
        #turn all the empty entries into 0 abs probs
        for row in listOfRows[1:]:
            for j in range(len(row)):
                if row[j]=='':
                    row[j]=0
            if emRow == 'reject':
                sumRow=0.0
                for j in range(1,len(row)):
                    sumRow += float(row[j])
                if sumRow == 0.0:
                    emptyRows += 1
                    break
        if emptyRows != 0:
            #print('skipping '+f)
            continue
        #now get the tpds
        distprobs = []
        #print(f)
        for i in range(1,101):
            distprobs.append([float(x) for x in listOfRows[i][1:]])
        entVec = []
        for tpd in distprobs:
            entAtDist = scipy.stats.entropy(tpd,base=2)
            entVec.append(entAtDist)
        sdsList.append(f)
        entVecList.append(entVec)
    #print(entVecList)
    #now, k-medoids cluster based on entVecList
    distMat = pdist(entVecList, 'cosine')#condensed
    distMat_sq = squareform(distMat)#redundant, square
    #print(trackList)
    clus_and_med = cluster(distMat_sq,k)
    meds = [sdsList[med] for med in clus_and_med[1]]
    clus = []
    for l,sds in enumerate(sdsList):
        clus.append([sds,clus_and_med[0][l],sdsList[clus_and_med[0][l]]])
    csvName = 'kmedoids test.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(meds)
    for row in clus:
        lw.writerow(row)  

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
             
def bigramTPDcluster(fld,k):
    '''
    Input: the collection of sds temporal probability distributions (TPDs), abs(P) vs. time
    Pull each individual (100-dim) TPD and label it as an OC-DC bigram vector
    Run k-medoids on all bigram vectors to cluster by "progression similarity type"
    (i.e., OC1 goes to DC1 in the same way that OC2 -> DC2)
    NB: k-medoids converges LOCALLY
    '''
    import scipy.stats
    listing = os.listdir(fld)
    bigramTPDList = []
    bgList = []
    for f in listing:
        address = fld + f
        transMat = csvTransposer(address,'tempTRANSmat.csv')
        allDests = csv.reader(open('tempTRANSmat.csv','r',newline='\n'))
        listOfRows = []
        #each row is a destination chord
        for row in allDests:
            listOfRows.append(row)
        #turn all the empty entries into 0 abs probs
        for row in listOfRows:
            for j in range(101):
                if row[j]=='':
                    row[j]=0
        #now get the tpds
        for i in range(1,len(listOfRows)):
            bigramTPDList.append([float(x) for x in listOfRows[i][1:]])
            bgList.append([f,listOfRows[i][0]])#labels; same order as actual TPD list
    #now, k-medoids cluster based on entVecList
    distMat = scipy.spatial.distance.pdist(bigramTPDList, 'cosine')#condensed
    distMat_sq = scipy.spatial.distance.squareform(distMat)#redundant, square
    clus_and_med = cluster(distMat_sq,k)
    meds = [bgList[med] for med in clus_and_med[1]]
    clus = []
    for l,bg in enumerate(bgList):
        clus.append([bg,clus_and_med[0][l],bgList[clus_and_med[0][l]]])
    csvName = 'bigramTPD kmedoids test.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(meds)
    for row in clus:
        lw.writerow(row) 
        
def rareSDSFixer(fld):
    """
    Of necessity, some sds are so rare that they won't have a full 100-tw TPD
    Need to take the csvs and fill in the missing t-windows with 0 entries
    TODO: Think about what the dummy rows would/should be for u-relative probs?
    """
    listing = os.listdir(fld)
    for f in listing:
        address = fld + f
        allDists = csv.reader(open(address,'r',newline='\n'))
        lstOfRows = []
        #each row is a time window
        for row in allDists:
            lstOfRows.append(row)#should be 100 of these
        nonHDR = lstOfRows[1:]
        for row in nonHDR:
            row[0] = int(row[0])
        listOfRows = sorted(nonHDR,key=lambda r: r[0],reverse=True)
        listOfRows.insert(0,lstOfRows[0])
        #print(listOfRows)
        #turn all the empty entries into 0 abs probs
        expandedList = []
        for i,row in enumerate(listOfRows):
            if i==0:
                firstRow = [str(0)]
                for m in range(1,len(row)):
                    firstRow.append(row[m])
                expandedList.append(firstRow)
                continue
            if i > 101:
                break
            if int(row[0]) - int(expandedList[-1][0]) == -1:
                expandedList.append(row)
            else:
                while int(row[0])-int(expandedList[-1][0]) < -1:
                    dummyRow = [str(int(expandedList[-1][0])-1)]
                    for k in range(len(expandedList[0])-1):
                        dummyRow.append(0.0)
                    expandedList.append(dummyRow)
                expandedList.append(row)
        while int(expandedList[-1][0]) > -100:
            dummyRow = [str(int(expandedList[-1][0])-1)]
            for k in range(len(expandedList[0])-1):
                dummyRow.append(0.0)
            expandedList.append(dummyRow)
        file = open('Abs Syntax Backwards_rev/'+f, 'w',newline='\n')
        lw = csv.writer(file)
        for row in expandedList:
            lw.writerow(row)
            
def TPDmatrixSim(r,fld,topN,k,meth='naive'):
    '''
    Inputs: TPD csvs for YJaMP (in directory fld), unigram probs for c3+ (called), topN most prob chords to track
    Places all the TPDs into matrices with rows ordered by unigram prob
    Includes dummy "0 rows" to preserve matrix layout
    Calculates naive similarity metric between each two matrices
    Uses resulting distance matrix (of distances BETWEEN matrices) for clustering
    Outputs: origin chord clusters and prototypes
    '''
    allChords = csv.reader(open('50ms 3 SDSets.csv','r',newline='\n'))
    matList = []#here are all the oc tpd matrices so far assembled
    distMat = np.zeros((topN,topN))
    allChordsList = []
    uniProbs = {}
    for i, row in enumerate(allChords):
        #Make a list of the topN most unigram-probable sds
        if i > topN - 1:
            break
        allChordsList.append(row[0])
        uniProbs[row[0]] = int(row[1])
    #print(allChordsList[0])#Can leave chord names as strings, right?
    listing = os.listdir(fld)
    flist = []
    #now iterate through all DC TPDs
    for f in listing:
        #Toss out those not in allChordsList
        chdStr = f.split('.')[0]#more csv kludging
        sdsStr = chdStr.split(']')[0] + ']'
        if sdsStr not in allChordsList:
            #print('skipping '+f)
            continue
        #any f reaching this point is a topN chord
        ocName = f.split(']')[0]+']'
        flist.append(ocName)
        #now, to strip out low-P destination chords
        address = fld + f
        allDists = csv.reader(open(address,'r',newline='\n'))
        lstOfRows = []
        #each row is a time window
        for row in allDists:
            lstOfRows.append(row)#should be 100 of these
        #now build the matrix
        orderedCols = [0]    
        for sds in allChordsList:
            matches = 0
            for j,dc in enumerate(lstOfRows[0]):
                if j==0:
                    continue
                if dc == sds:
                    orderedCols.append(j)
                    matches += 1
                    break
            if matches == 0:
                orderedCols.append(sds)
        dcMat = []
        for row in lstOfRows:
            goodRow = []
            for m in orderedCols:
                if type(m) == str:
                    goodRow.append('0.0')
                    continue
                goodRow.append(row[m])
            dcMat.append(goodRow)
        '''
        file = open(str(topN)+' AbsP Syntax Forwards/'+f, 'w',newline='\n')
        lw = csv.writer(file)
        for row in dcMat:
            lw.writerow(row)
        '''
        for i,mat in enumerate(matList):
            if len(matList) == 0:
                break
            #can naively compare matrices entry-by-entry, no normalization, no nothing 
            if meth=='naive':
                distMat[i][len(matList)] = naiveDistance(mat, dcMat)
                distMat[len(matList)][i] = naiveDistance(mat, dcMat)
            if meth=='cosine':
                distMat[i][len(matList)] = avgCosDistance(mat, dcMat)
                distMat[len(matList)][i] = avgCosDistance(mat, dcMat)
        matList.append(dcMat)
    print(distMat)
    '''
    file = open(str(topN)+' cDistMat AbsP Syntax Forwards.csv', 'w',newline='\n')
    lw = csv.writer(file)
    for row in distMat:
        lw.writerow(row)
    '''
    #now, flist is a list of the topN chords (in some order)
    #distMat is a redundant square matrix of naive distances between flist matrices (in SAME order)
    #can cluster based on those
    clus_and_med = cluster(distMat,k)
    meds = [flist[med] for med in clus_and_med[1]]
    clus = []
    for l,fp in enumerate(flist):
        clus.append([fp,uniProbs[fp],clus_and_med[0][l],flist[clus_and_med[0][l]]])
    if meth=='naive':
        csvName = 'TPDmatrixSim kmedoids '+str(topN)+'_n'+str(k)+'run'+str(r)+'.csv'
    elif meth=='cosine':
        csvName = 'TPDmatrixSim kmedoids '+str(topN)+'_c'+str(k)+'run'+str(r)+'.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(meds)
    lw.writerow(['origin chord','unigram tallies','cluster','medoid'])
    for row in clus:
        lw.writerow(row) 
        
def naiveDistance(mat1,mat2,headers='yes'):
    ''' Finds the summed, absolute, entry-for-entry distance between TPD mat1 and mat2'''
    summedAbsDist = 0.0
    startRow = 0
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        print('Error: matrices of different shape')
    if headers == 'yes':
        startRow = 1
    for m in range(startRow,len(mat1)):
        for n in range(startRow,len(mat1[m])):
            if mat1[m][n] == '':
                mat1[m][n] = 0.0
            if mat2[m][n] == '':
                mat2[m][n] = 0.0
            #print(mat1[m][n],mat2[m][n])
            summedAbsDist += abs(float(mat1[m][n]) - float(mat2[m][n]))
    return summedAbsDist

def avgCosDistance(mat1,mat2,headers='yes'):
    '''Compare two matrices row-for-row, calculating average cos distance over all matched rows'''
    allCosDist = 0.0
    startRow = 0
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        print('Error: matrices of different shape')
    if headers == 'yes':
        startRow = 1
    for m in range(startRow,len(mat1)):
        dot = 0
        magp = 0
        magk = 0
        for n in range(startRow,len(mat1[m])):
            if mat1[m][n] == '':
                mat1[m][n] = 0.0
            if mat2[m][n] == '':
                mat2[m][n] = 0.0
            dot += float(mat1[m][n])*float(mat2[m][n])
            magp += np.power(float(mat1[m][n]),2)
            magk += np.power(float(mat2[m][n]),2)
        allCosDist += 1 - dot/(np.sqrt(magp)*np.sqrt(magk))
    return allCosDist/len(mat1)

def matrixSimCaller(r,fld,topN,k,meth='naive'):
    '''Just runs TPDmatrixSim i times and outputs list of clusterings'''
    for j in range(r):
        TPDmatrixSim(j,fld,topN,k,meth)  

def metaCluster(fld,k):
    '''
    For a collection of (locally-convergent) clusterings in fld
    Take each origin chord and track its cluster IDs across clusterings
    Compare the resulting membership vectors and cluster by THEIR similarity
    TODO: think about how to weight meta-clustering?  Distance -> 1/#same clus assigns? Or #diff clus assigns
    I think hamming distance is ``how many entries are not the same."  Sounds right.
    '''
    listing = os.listdir(fld)
    flist = []
    #now iterate through all DC TPDs
    i=0
    clusDict = {}
    ocs = []
    clusMat = []
    allChords = csv.reader(open('50ms 3 SDSets.csv','r',newline='\n'))
    uniProbs = {}
    for row in allChords:
        #Make a list of the topN most unigram-probable sds
        uniProbs[row[0]] = int(row[1])
    uniProbs = getProbsFromFreqs(uniProbs)
    for f in listing:
        #if i: break
        address = fld + f
        allOCs = csv.reader(open(address,'r',newline='\n'))
        lstOfOCs = []
        #each row is a time window
        for row in allOCs:
            lstOfOCs.append(row)#should be topN of these
        for j,oc in enumerate(lstOfOCs):
            if j < 2: continue#cut the two header rows
            if not oc[0] in clusDict: clusDict[oc[0]] = []
            clusDict[oc[0]].append(oc[2])#stick the (int) cluster ass. in dict
    for key in clusDict.keys():
        ocs.append(key)#this tells us what the rows of clusMat refer to
        clusMat.append(clusDict[key])#this is what we'll cluster
    print('clusDict',clusDict)
    print('first row of oc list and first rowof clusMat')
    print(ocs[0],clusMat[0])
    #now, calculate hamming distances between rows/ocs
    distMat = pdist(clusMat,metric='hamming')
    distMat_sq = squareform(distMat)#redundant, square
    print(distMat_sq)
    clus_and_med = cluster(distMat_sq,k)
    meds = [ocs[med] for med in clus_and_med[1]]
    clus = []
    for l,oc in enumerate(ocs):
        clus.append([oc,uniProbs[oc],clus_and_med[0][l],ocs[clus_and_med[0][l]],distMat_sq[l,clus_and_med[0][l]]])
    csvName = 'metacluster test.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(meds)
    lw.writerow(['origin chord','uprob','cluster','medoid','distance'])
    for row in clus:
        lw.writerow(row)
        
def getSilhouettes(distmat,fld,k='single'):
    '''
    Hunts through all clusterings in fld and spits out ranked list of the silhouette dists for each
    Silhouette: avg over all points of [a(i) - b(i)/max(a(i),b(i))]
    a(i) is avg in-cluster dissimilarity
    b(i) is avg dissimilarity to next-best cluster
    '''
    import operator
    import sklearn
    from sklearn import metrics    
    diMat = []
    i=0
    dists = csv.reader(open(distmat, 'r',newline='\n'))
    for row in dists:
        diMat.append(row)
    disArr = np.array(diMat)#pairwise dist mat (gen Manh?) as strings
    diArr = disArr.astype(float)#now as floats
    listing = os.listdir(fld)
    silh = []
    for f in listing:
        #if i: break
        address = fld + f
        k1 = f.split('run')[1]
        k2 = k1.split('.')[0]
        clus = csv.reader(open(address,'r',newline='\n'))#cluster assignment csv
        clusRows = []
        for row in clus:
            clusRows.append(row)
        clusAssMat = np.empty(len(clusRows)-2)
        for i,row in enumerate(clusRows):
            if i < 2: continue
            clusAssMat[i-2] = row[2]
        #print(len(clusAssMat),clusAssMat)
        #print(len(diArr))
        #need to pull oc names and cluster labels (by number?)
        msil = sklearn.metrics.silhouette_score(diArr,clusAssMat,metric='precomputed')
        sil = sklearn.metrics.silhouette_samples(diArr,clusAssMat,metric='precomputed')
        if k != 'single':
            silh.append([f,msil,sil,k2])
        else:
            silh.append([f,msil,sil])
        i += 1
    silh.sort(key=operator.itemgetter(1),reverse=True)
    csvName = 'clus_silhouettes_redo.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    if k != 'single':
        lw.writerow(['clustering','silhouette score','each sample silhouette','k'])
    else:
        lw.writerow(['clustering','silhouette score','each sample silhouette'])
    for row in silh:
        lw.writerow(row)

def plot_dendrogram(model, **kwargs):
    #taken from online example in sklearn fork
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

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def agglomClus(distmat,k):
    '''
    For all the TPD matrices captured by pairwise distmat, uses sklearn to hierarchically cluster
    if meth=agglomerative, bottom up
    '''
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.datasets import load_iris
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances
    from matplotlib import pyplot as plt
    import itertools
    diMat = []
    i=0
    dists = csv.reader(open(distmat, 'r',newline='\n'))
    for row in dists:
        diMat.append(row)
    disArr = np.array(diMat)#pairwise dist mat (gen Manh?) as strings
    diArr = disArr.astype(float)#now as floats
    #distMat_cond = squareform(diArr)#turns redundant, square into condensed, triangular
    mclus = AgglomerativeClustering(n_clusters = k, affinity='precomputed',linkage='complete')
    clusfit = mclus.fit(diArr)
    labels = clusfit.labels_
    #print(labels)
    chdnames = csv.reader(open('n20_clustests/27TPDmatrixSim kmedoids 200_n20run1.csv', 'r',newline='\n'))
    chdnamesit = []
    for row in chdnames:
        chdnamesit.append(row)
    chdnameslst = []
    for i,chd in enumerate(chdnamesit):
        if i<2: continue
        chdnameslst.append(chd[0])
    print(chdnameslst)
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(clusfit,labels=chdnameslst,show_leaf_counts=True,leaf_font_size=8,leaf_rotation=45)#labels=clusfit.labels_
    plt.show()
    ii = itertools.count(diArr.shape[0])
    nodelst = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in clusfit.children_]
    csvName = 'agglom_testing.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    for row in nodelst:
        vals = []
        for key,value in row.items():
            vals.append(value)
        lw.writerow(vals)
        
def subCluster(n,clustr,distMat):
    '''
    Takes a clustering csv and distance matrix as inputs
    For largest two clusters (by probability mass), builds reduced/sliced distMat 
    Breaks the largest two clusters into n subclusters via k-medoids
    '''
    import operator
    import sklearn
    from sklearn import metrics
    
    #get the unigram probs
    allChords = csv.reader(open('50ms 3 SDSets.csv','r',newline='\n'))
    uniProbs = {}
    for row in allChords:
        uniProbs[row[0]] = int(row[1])
    uniProbs = getProbsFromFreqs(uniProbs)
    
    #get the distance matrix
    diMat = []
    i=0
    dists = csv.reader(open(distMat, 'r',newline='\n'))
    for row in dists:
        diMat.append(row)
    disArr = np.array(diMat)#pairwise dist mat (gen Manh?) as strings
    diArr = disArr.astype(float)#now as floats
    #print(diArr)
    
    #get the kludgy lookup list
    lkps = {}
    lkp = csv.reader(open('ndistMat_lookups.csv','r',newline='\n'))
    for row in lkp:
        lkps[row[1]] = row[0]
    
    #get the medoids and membership from prev clustering
    meds = {}#dict of medoids: each medoid keys a list of [chord, chord,...]
    medP = {}#dict of total unigram tallies keyed by medoid
    clsts = csv.reader(open(clustr, 'r',newline='\n'))
    i=0
    for row in clsts:
        i+=1
        if i < 3: continue
        if row[2] not in meds:
            meds[row[2]] = []
            medP[row[2]] = 0
        meds[row[2]].append(row[0])
        medP[row[2]] += int(row[1])
    sorted_medP = sorted(medP.items(), key=operator.itemgetter(1), reverse=True)
    
    ri = str(random.randint(0,5000))
    sils = []
    newclus = []
    #take the two biggest clusters and generate a new intra-clus disMat
    for j in range(2):
        subcl_id = []#this will be a list of row indices for new_distMat
        subcl = meds[sorted_medP[j][0]]#all the chord names in med
        for chd in subcl:
            subcl_id.append(lkps[chd])#the numerical maps for those chords
        rows = np.array(subcl_id, dtype=np.intp)
        new_distMat = diArr[np.ix_(rows, rows)]
        clus_and_med = cluster(new_distMat,n)
        new_meds = [subcl[m] for m in clus_and_med[1]]
        msil = sklearn.metrics.silhouette_score(new_distMat,clus_and_med[0],metric='precomputed')
        print(len(clus_and_med[0]),new_distMat.shape,msil)
        sils.append([ri+'_'+str(j),msil])
        #print(new_meds)
        for l,oc in enumerate(subcl):
            newclus.append([oc,uniProbs[oc],clus_and_med[0][l],subcl[clus_and_med[0][l]],new_distMat[l,clus_and_med[0][l]]])
    
    #now dump the new clusterings into a csv
    csvName = 'subClus/subcluster test'+ri+'.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(['origin chord','uprob','cluster','medoid','distance'])
    for row in newclus:
        lw.writerow(row)
    file2 = open('subClus/subClus_silh.csv','a',newline='\n')
    lw2 = csv.writer(file2)
    for row in sils:
        lw2.writerow(row)

#subCluster(10, "n10_clusTests/2472TPDmatrixSim kmedoids 200_n10run4.csv", '200 nDistMat AbsP Syntax Forwards.csv')
#agglomClus('200 nDistMat AbsP Syntax Forwards.csv',2)
#getSilhouettes('200 nDistMat AbsP Syntax Forwards.csv','C:/Users/Andrew/workspace/DissWork/clus_across_k/',k='multi')        
#metaCluster('C:/Users/Andrew/workspace/DissWork/clus_across_k/',20)
#matrixSimCaller(10, '/lustre/scratch/client/fas/quinn/adj24/Abs Syntax Forwards_rev/', 200, 10, meth='naive')
#TPDmatrixSim(0,'C:/Users/Andrew/workspace/DissWork/Abs Syntax Forwards_rev/',200,10,meth='naive')
#rareSDSFixer('C:/Users/Andrew/workspace/DissWork/Abs Syntax Backwards/')
#TPDentropyCluster('C:/Users/Andrew/workspace/DissWork/Abs Syntax Forwards_rev/',5)
#bigramTPDcluster('C:/Users/Andrew/workspace/DissWork/50 AbsP Syntax Forwards/',10)#memory hog!