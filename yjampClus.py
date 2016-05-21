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

def cluster(distances, k=3):
    '''
    Elegant kmedoids from machine_learning github fork (kmedoids.py)
    Takes (redundant, square) distance matrix
    Returns k clusters and medoids
    '''
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
    #also from machine_learning github fork
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    #also from machine_learning github fork
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    #print(costs,costs.argmin(axis=0, fill_value=10e9))
    return costs.argmin(axis=0, fill_value=10e9)

def TPDentropyCluster(fld,k,emRow = 'reject'):
    '''
    Input: the collection of scale degree set temporal probability distributions (TPDs), abs(P) vs. time
    For each time window, calculate the entropy of the origin chord's TPDs
    Output a vector with one entropy entry per time window
    Run k-medoids on entropy vectors to cluster by "chord regime type"
    (i.e., does sds participate in mostly local, mostly long range, some syntactic progs, etc.)
    NB: k-medoids converges LOCALLY
    '''
    import scipy.stats
    listing = os.listdir(fld)
    entVecList = []#list of entropy vectors, one per origin chord, each 100 time windows long
    sdsList = []#list of scale degree sets in same order as vectors in entVecList (and meant as labels for them)
    #iterate through origin chords in fld listing
    for f in listing:
        #get the raw csvrows
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
            #added option for rejecting chords with an entire empty window
            if emRow == 'reject':
                sumRow=0.0
                for j in range(1,len(row)):
                    sumRow += float(row[j])
                if sumRow == 0.0:
                    emptyRows += 1
                    break
        #don't bother tracking entropy for rare chords (which have some empty rows)
        if emptyRows != 0:
            #print('skipping '+f)
            continue
        
        #now get the TPDs for origin chords which behaved properly above
        distprobs = []
        #print(f)
        for i in range(1,101):
            distprobs.append([float(x) for x in listOfRows[i][1:]])
        entVec = []
        for tpd in distprobs:
            entAtDist = scipy.stats.entropy(tpd,base=2)
            entVec.append(entAtDist)
        sdsList.append(f)#add scale degree set label for origin chord to ordered list
        entVecList.append(entVec)#add entropy vector for origin chord to ordered list
    
    #now, k-medoids cluster based on entVecList
    #print(entVecList)
    distMat = pdist(entVecList, 'cosine')#condensed; use cosine metric, for now
    distMat_sq = squareform(distMat)#redundant, square matrix
    clus_and_med = cluster(distMat_sq,k)
    
    #put the list of cluster assignments in a csv
    meds = [sdsList[med] for med in clus_and_med[1]]
    clus = []
    for l,sds in enumerate(sdsList):
        clus.append([sds,clus_and_med[0][l],sdsList[clus_and_med[0][l]]])
    csvName = 'kmedoids_entropy_test.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(meds)
    for row in clus:
        lw.writerow(row)  
             
def bigramTPDcluster(fld,k):
    '''
    Input: the collection of sds temporal probability distributions (TPDs), abs(P) vs. time
    Pull each individual (100-dim) TPD and label it as an OC-DC bigram vector
    Run k-medoids on all bigram vectors to cluster by "progression similarity type"
    (i.e., OC1 goes to DC1 in the same way that OC2 -> DC2)
    NB: k-medoids converges LOCALLY, and this is very inefficient!
    [Likely dead end, but kept in case necessary in future]
    '''
    import numpy
    import scipy.stats
    from sklearn.decomposition import PCA
    
    listing = os.listdir(fld)
    bigramTPDList = []
    bgList = []
    for f in listing:
        #Get origin chord data
        address = fld + f
        #these start out as (row, col) = (time window, dest chord)
        transMat = csvTransposer(address,'tempTRANSmat.csv')
        allDests = csv.reader(open('tempTRANSmat.csv','r',newline='\n'))
        listOfRows = []
        #now, each row is a destination chord with (time window) cols
        for row in allDests:
            listOfRows.append(row)
        #turn all the empty entries into 0 abs probs
        for row in listOfRows:
            for j in range(101):
                if row[j]=='':
                    row[j]=0
        
        #run PCA on destination chord spectrum for given origin chord
        distprobs = []
        for i in range(1,len(listOfRows)):
            distprobs.append([float(x) for x in listOfRows[i][1:]])
        #print(len(distprobs[0]),distprobs[0])
        #convert into numpy array for PCA; num DCs rows x 100 ts cols
        probarr = numpy.array(distprobs)
        #print(probarr.shape,probarr[0])
        pca = PCA(n_components = 3)
        pca.fit(probarr)
        #put DC vectors into PCA basis
        transformed_data = pca.fit(probarr).transform(probarr)
        compn = pca.components_
        
        #to orient the components for comparison, we need at least 3 of them
        if len(compn) < 3:
            continue
        #NB!: for many chords, the components are shitty and don't tell us anything good!
        if compn[0][0] < 0:#set the first component to start positive (usually phonetic data)
            for dcrow in transformed_data:
                dcrow[0] = -1*dcrow[0]
        if compn[1][0] > 0:#set the second component to start negative (usually long-range key data)
            for dcrow in transformed_data:
                dcrow[1] = -1*dcrow[1]
        if compn[2][0] > 0:#set the third component to start negative (usually syntactic[?] data)
            for dcrow in transformed_data:
                dcrow[2] = -1*dcrow[2]
        #print(len(transformed_data),len(transformed_data[1]))
        for i,dcrow in enumerate(transformed_data):
            bigramTPDList.append(dcrow)#put PCA-basis bigram data in list
            bgList.append([f,listOfRows[i][0]])#labels; same order as actual TPD list

    #now, k-medoids cluster based on bigramTPDList
    distMat = scipy.spatial.distance.pdist(bigramTPDList, 'cosine')#condensed
    distMat_sq = scipy.spatial.distance.squareform(distMat)#redundant, square
    clus_and_med = cluster(distMat_sq,k)
    
    #send out the results via csv
    meds = [bgList[med] for med in clus_and_med[1]]
    clus = []#format: [bigram label, cluster assignment number, cluster medoid label]
    for l,bg in enumerate(bgList):
        clus.append([bg,clus_and_med[0][l],bgList[clus_and_med[0][l]]])
    csvName = 'bigramTPD kmedoids PCAtest.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(meds)
    for row in clus:
        lw.writerow(row) 
        
def rareSDSFixer(fld):
    '''
    Of necessity, some scale degree sets are so rare that they won't have a full 100-tw TPD
    Quick code to take the csvs and fill in the missing t-windows with 0 entries
    NB: used rarely, set the file path for output manually each time
    '''
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
            if i==0:#handle list of t window time labels
                firstRow = [str(0)]
                for m in range(1,len(row)):
                    firstRow.append(row[m])
                expandedList.append(firstRow)
                continue
            if i > 101:#don't keep too many
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
                
        #if the final list doesn't get to 100, add dummy 0 rows
        while int(expandedList[-1][0]) > -100:
            dummyRow = [str(int(expandedList[-1][0])-1)]
            for k in range(len(expandedList[0])-1):
                dummyRow.append(0.0)
            expandedList.append(dummyRow)
            
        #send the fixed csvs to a folder you determine
        file = open('Abs Syntax Backwards_rev/'+f, 'w',newline='\n')
        lw = csv.writer(file)
        for row in expandedList:
            lw.writerow(row)
            
def TPDmatrixSim(fld,topN,k,meth='naive',reduc='none',sendData=False):
    '''
    Inputs: TPD csvs for YJaMP (in directory fld), unigram probs for c3+ (called), topN most prob chords to track
    Places all the TPDs into matrices with rows ordered by descending unigram prob
    Includes dummy "0 rows" to preserve matrix layout across origin chords in fld
    Calculates naive or modified cosine similarity metric between each two matrices
    Uses resulting distance matrix (of distances BETWEEN matrices) for clustering
    Outputs: origin chord clusters and prototypes
    '''
    import numpy as np
    from sklearn.decomposition import PCA
    
    matList = []#here are all the oc tpd matrices so far assembled
    
    #figure out which chords are in the topN most probable for keeping
    allChords = csv.reader(open('50ms 3 SDSets.csv','r',newline='\n'))
    distMat = np.zeros((topN,topN))
    allChordsList = []#the names of the topN origin chords
    uniProbs = {}#a dict of their unigram probs
    for i, row in enumerate(allChords):
        #Make a list of the topN most unigram-probable sds
        if i > topN - 1:
            break
        allChordsList.append(row[0])
        uniProbs[row[0]] = int(row[1])    
    #print(allChordsList[0])#Can leave chord names as strings
    
    #now iterate through all DC TPDs in fld
    listing = os.listdir(fld)
    flist = []#origin chord labels/names, in order
    for f in listing:
        #Toss out those not in allChordsList (i.e, not topN prob)
        chdStr = f.split('.')[0]#more csv kludging
        sdsStr = chdStr.split(']')[0] + ']'
        if sdsStr not in allChordsList:
            #print('skipping '+f)
            continue
        #any f reaching this point is a topN chord
        ocName = f.split(']')[0]+']'
        flist.append(ocName)
        
        #get the data for the origin chord's destinations over time
        address = fld + f
        allDists = csv.reader(open(address,'r',newline='\n'))
        lstOfRows = []
        #each row is a time window
        for row in allDists:
            lstOfRows.append(row)#should be 100 of these
            
        #now build the destination chord matrix in its proper order
        dcMat = []#the correctly-ordered destionation chord matrix
        orderedCols = [0]#indices of the data rows necessary to get them in topN order    
        for sds in allChordsList:#the previously-assembled list of topN probable chords
            matches = 0
            for j,dc in enumerate(lstOfRows[0]):#pull out which columns match in order
                if j==0:#skip header row
                    continue
                if dc == sds:
                    orderedCols.append(j)#row index
                    matches += 1
                    break
            if matches == 0:
                orderedCols.append(sds)#if no match in dcs, append the missing chord name
        for j,row in enumerate(lstOfRows):
            goodRow = []
            if j==0:#treat header row differently
                for m in orderedCols:
                    if type(m) == str:#for cols missing
                        goodRow.append(m)
                        continue
                    goodRow.append(row[m])#for extant cols
            else:#non-header rows
                for m in orderedCols:#now fill each row in order
                    if type(m) == str:
                        goodRow.append('0.0')#zeros for the previously missing cols
                        continue
                    goodRow.append(row[m])#correct probs for extant cols
            dcMat.append(goodRow)
            
        #send out the ordered rows (and be careful setting the filepath)
        if sendData:
            file = open(str(topN)+' AbsP Syntax Forwards/'+f, 'w',newline='\n')
            lw = csv.writer(file)
            for row in dcMat:
                lw.writerow(row)
                
        #turn empty cells into float 0.0
        for m in range(1,len(dcMat)):
            for n in range(1,len(dcMat[m])):
                if dcMat[m][n] == '':
                    dcMat[m][n] = 0.0
        
        #run PCA
        if reduc=='PCA':
            dcMat_noheads = []#same as dcMat, but without row/col labels
            for i in range(1,len(dcMat)):
                dcMat_noheads.append([float(x) for x in dcMat[i][1:]])
            #convert into numpy array for PCA
            probarr = np.transpose(np.array(dcMat_noheads))# num dc rows by num time stamps cols, now
            pca = PCA(n_components = 3)
            pca.fit(probarr)
            #this is the PCA-transformed data (topN rows by n_components columns)
            transformed_data = pca.fit(probarr).transform(probarr)
            compn = pca.components_
            """#to orient the components for comparison, we need at least 3 of them
            if len(compn) < 3:
                continue
            """
            #NB!: for many chords, the components are shitty and don't tell us anything good!
            if compn[0][0] < 0:#set the first component to start positive (usually phonetic data)
                for dcrow in transformed_data:
                    dcrow[0] = -1*dcrow[0]
            if compn[1][0] > 0:#set the second component to start negative (usually long-range key data)
                for dcrow in transformed_data:
                    dcrow[1] = -1*dcrow[1]
            if compn[2][0] > 0:#set the third component to start negative (usually syntactic[?] data)
                for dcrow in transformed_data:
                    dcrow[2] = -1*dcrow[2]
            #now it's (n_components,topN), which is how the clustering works, but NO header rows
            dcMat = np.transpose(np.array(transformed_data))
            
        #now get the distance between this origin chord's TPD matrix and all the others so far
        for i,mat in enumerate(matList):
            if len(matList) == 0:#can't compare the first matrix to anything
                break
            if reduc=='PCA': hdr='no'
            else: hdr='yes'
            #choose a comparison method to get a distance metric between matrices
            if meth=='naive':
                distMat[i][len(matList)] = naiveDistance(mat, dcMat,headers=hdr)
                distMat[len(matList)][i] = naiveDistance(mat, dcMat, headers=hdr)
            if meth=='cosine':
                distMat[i][len(matList)] = avgCosDistance(mat, dcMat, headers=hdr)
                distMat[len(matList)][i] = avgCosDistance(mat, dcMat, headers=hdr)
        matList.append(dcMat)#now add the dcMat for this chord to the matList for later comps
    print(distMat)#this is the overall matrix-to-matrix distance matrix!
    """In case you need to write the distance (meta)matrix itself to file
    file = open(str(topN)+' nDistMat AbsP Syntax Forwards_rev.csv', 'w',newline='\n')
    lw = csv.writer(file)
    for row in distMat:
        lw.writerow(row)
    """
    
    #recall flist is a list of the topN chords (in some NOT P-BASED order)
    #distMat is a redundant square matrix of naive distances between flist matrices (in SAME order)
    #can cluster based on those
    clus_and_med = cluster(distMat,k)
    meds = [flist[med] for med in clus_and_med[1]]
    clus = []#format: [origin chord, unigram prob, cluster assignment, cluster medoid, distance to medoid]
    for l,fp in enumerate(flist):
        clus.append([fp,uniProbs[fp],clus_and_med[0][l],flist[clus_and_med[0][l]],distMat[l,clus_and_med[0][l]]])
    
    #some filepath name kludging for csv output
    if reduc=='PCA':
        k = str(k)+'PCA'
        ri = 'n10_PCA/'+str(random.randint(0,10000))
    else: ri = str(random.randint(0,10000))
    if meth=='naive':
        csvName = ri+'TPDmatrixSim kmed '+str(topN)+'_n'+str(k)+'.csv'
    elif meth=='cosine':
        csvName = ri+'TPDmatrixSim kmed '+str(topN)+'_c'+str(k)+'.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(meds)
    lw.writerow(['origin chord','unigram tallies','cluster #','medoid name','distance'])
    for row in clus:
        lw.writerow(row) 
        
def naiveDistance(mat1,mat2,headers='yes'):
    '''
    Returns the summed, absolute, entry-for-entry distance between TPD mat1 and mat2
    if headers=='yes', skips the first row and column
    '''
    summedAbsDist = 0.0#naive distance entrywise (Manhattan, strictly speaking)
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        print('Error: matrices of different shape')
    if headers == 'yes':
        startRow = 1
    elif headers == 'no':
        startRow = 0
    for m in range(startRow,len(mat1)):
        for n in range(startRow,len(mat1[m])):
            #assign floats of 0.0 to any empty cells
            if mat1[m][n] == '':
                mat1[m][n] = 0.0
            if mat2[m][n] == '':
                mat2[m][n] = 0.0
            #add entrywise distance to summedAbsDist
            summedAbsDist += abs(float(mat1[m][n]) - float(mat2[m][n]))
    return summedAbsDist

def avgCosDistance(mat1,mat2,headers='yes'):
    '''
    Compare mat1 and mat2 row-for-row, calculating average cos distance over all matched rows
    Returns average row-wise cosine distance between mat1 and mat2
    if headers=='yes', skips first row and column
    '''
    allCosDist = 0.0#summed cosine distance over all rows; divide by number of rows to get avg
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        print('Error: matrices of different shape')
    if headers == 'yes':
        startRow = 1
    elif headers == 'no':
        startRow = 0
    for m in range(startRow,len(mat1)):
        #kludgy cosine for bookkeeping clarity
        dot = 0
        magp = 0
        magk = 0
        for n in range(startRow,len(mat1[m])):
            #set empty cells to float 0.0
            if mat1[m][n] == '':
                mat1[m][n] = 0.0
            if mat2[m][n] == '':
                mat2[m][n] = 0.0
            dot += float(mat1[m][n])*float(mat2[m][n])
            magp += np.power(float(mat1[m][n]),2)
            magk += np.power(float(mat2[m][n]),2)
        allCosDist += 1 - dot/(np.sqrt(magp)*np.sqrt(magk))
    #average cosine distance
    return allCosDist/len(mat1)

def matrixSimCaller(r,fld,topN,k):
    '''
    Just runs TPDmatrixSim r times and outputs list of clusterings
    convenient for cluster computing
    '''
    for j in range(r):
        TPDmatrixSim(fld,topN,k,meth='naive',reduc='PCA')  

def metaCluster(fld,k):
    '''
    For a collection of (locally-convergent) clusterings in fld
    Take each origin chord and track its cluster IDs across clusterings
    Compare the resulting membership vectors and cluster by THEIR similarity
    Hamming distance (how many entries are not the same) seems most appropriate
    '''
    listing = os.listdir(fld)
    clusDict = {}#dict of cluster assignments across runs: clusDict[origin chord]=[assign 1, assign 2, ...]
    ocs = []#will be list of keys from clusDict
    clusMat = []#will be list of cluster assignments in order of ocs, also from clusDict
    
    #Pull chord names and probabilities for topN chords
    allChords = csv.reader(open('50ms 3 SDSets.csv','r',newline='\n'))
    uniProbs = {}
    for row in allChords:
        #Make a list of the topN most unigram-probable sds
        uniProbs[row[0]] = int(row[1])
    uniProbs = getProbsFromFreqs(uniProbs)
    
    for f in listing:
        #Get clustering data
        address = fld + f
        allOCs = csv.reader(open(address,'r',newline='\n'))
        lstOfOCs = []
        for row in allOCs:
            lstOfOCs.append(row)#should be topN of these
        
        #for each origin chord, append its assignment to the relevant clusDict entry list
        for j,oc in enumerate(lstOfOCs):
            if j < 2: continue#cut the two header rows
            if not oc[0] in clusDict: clusDict[oc[0]] = []#make one if there isn't one
            clusDict[oc[0]].append(oc[2])#stick the (int) cluster assignment in dict
            
    #build (stable, ordered) ocs and clusMat lists
    for key in clusDict.keys():
        ocs.append(key)#this tells us what the rows of clusMat refer to
        clusMat.append(clusDict[key])#this is what we'll cluster
    print('clusDict',clusDict)
    print('first row of oc list and first row of clusMat')
    print(ocs[0],clusMat[0])
    
    #now, calculate hamming distances between rows/ocs
    distMat = pdist(clusMat,metric='hamming')
    distMat_sq = squareform(distMat)#redundant, square
    print(distMat_sq)
    #kmedoids
    clus_and_med = cluster(distMat_sq,k)
    meds = [ocs[med] for med in clus_and_med[1]]
    clus = []#format: [origin chord, unigram prob, cluster assignment, medoid, distance from medoid]
    for l,oc in enumerate(ocs):
        clus.append([oc,uniProbs[oc],clus_and_med[0][l],ocs[clus_and_med[0][l]],distMat_sq[l,clus_and_med[0][l]]])
    
    #send out the csv
    csvName = 'metacluster test.csv'
    file = open(csvName, 'w',newline='\n')
    lw = csv.writer(file)
    lw.writerow(meds)
    lw.writerow(['origin chord','uprob','cluster','medoid','distance'])
    for row in clus:
        lw.writerow(row)
        
def getSilhouettes(distmat,fld,k='single'):
    '''
    !!NB: distMats are strange, platform-dependent orderings
    Depends on how the os walks the file tree!!
    Hunts through all clusterings in fld and spits out ranked list of the silhouette dists for each
    Silhouette: avg over all points of [a(i) - b(i)/max(a(i),b(i))]
    a(i) is avg in-cluster dissimilarity
    b(i) is avg dissimilarity to next-best cluster
    '''
    import operator
    import sklearn
    from sklearn import metrics  
      
    #get the point-to-point distances
    diMat = []
    dists = csv.reader(open(distmat, 'r',newline='\n'))
    for row in dists:
        diMat.append(row)
    disArr = np.array(diMat)#pairwise dist mat (generalized Manhattan) as strings
    diArr = disArr.astype(float)#now as floats
    
    #iterate through the clusterings and get silhouette data
    listing = os.listdir(fld)
    silh = []#format: [clustering file, overall silhouette, sample-wise silhouette,k (if not single)]
    i=0 
    for f in listing:
        #if i: break
        
        #get data from the clustering csv
        address = fld + f
        k1 = f.split('_n')[1]
        k2 = k1.split('PCA')[0]
        clus = csv.reader(open(address,'r',newline='\n'))#cluster assignment csv
        clusRows = []
        for row in clus:
            clusRows.append(row)
        #this will be an empty array of cluster assignments
        clusAssMat = np.empty(len(clusRows)-2)
        for i,row in enumerate(clusRows):
            if i < 2: continue
            clusAssMat[i-2] = row[2]
        #print(len(clusAssMat),clusAssMat)
        
        #from distance matrix and cluster assignment matrix, compute silhouette scores
        msil = sklearn.metrics.silhouette_score(diArr,clusAssMat,metric='precomputed')
        sil = sklearn.metrics.silhouette_samples(diArr,clusAssMat,metric='precomputed')
        if k != 'single':
            silh.append([f,msil,sil,k2])
        else:
            silh.append([f,msil,sil])
        i += 1
        
    #sort by best overall silhouette and output csv
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
    '''
    taken from online example in sklearn fork
    turns hierarchical model into dendrogram
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

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def agglomClus(distmat,k,sendData=False):
    '''
    For all the TPD matrices captured by pairwise distmat, uses sklearn to hierarchically cluster
    if meth=agglomerative, bottom up
    k number of clusters
    '''
    from scipy.cluster.hierarchy import dendrogram
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
    
    #output agglomerative mergings as csv if sendData==True
    if sendData:
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
    
    #plot a dendrogram of the agglomerative hierarchical clustering
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(clusfit,labels=chdnameslst,show_leaf_counts=True,leaf_font_size=8,leaf_rotation=45)#labels=clusfit.labels_
    plt.show()
        
def subCluster(n,clustr,distMat):
    '''
    Takes a clustering csv (clustr) and distance matrix (distMat) as inputs
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
    dists = csv.reader(open(distMat, 'r',newline='\n'))
    for row in dists:
        diMat.append(row)
    disArr = np.array(diMat)#pairwise dist mat (gen Manh?) as strings
    diArr = disArr.astype(float)#now as floats
    #print(diArr)
    
    #get the kludgy lookup list that relates chord labels to distMat rows
    lkps = {}
    lkp = csv.reader(open('ndistMat_lookups_rev.csv','r',newline='\n'))
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
    #list of medoids sorted by descending unigram probability captured
    sorted_medP = sorted(medP.items(), key=operator.itemgetter(1), reverse=True)
    
    #take the two biggest clusters and generate a new intra-clus disMat
    ri = str(random.randint(0,5000))#silly rn for csv bookkeeping
    sils = []#list silhouettes of the two biggest clusters
    newclus = []#format: [origin chord, unigram prob, cluster assign, medoid, distance to medoid]
    for j in range(2):
        subcl_id = []#this will be a list of row indices for new_distMat
        subcl = meds[sorted_medP[j][0]]#all the chord names in med
        for chd in subcl:
            subcl_id.append(lkps[chd])#the numerical maps for those chords
        rows = np.array(subcl_id, dtype=np.intp)
        new_distMat = diArr[np.ix_(rows, rows)]#distance matrix for subcluster
        
        #kmedoids on the subcluster
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

#subCluster(10, "563TPDmatrixSim kmed 200_n10.csv", '200 nDistMat AbsP Syntax Forwards_rev.csv')
#agglomClus('200 nDistMat AbsP Syntax Forwards_PCA.csv',2)
#getSilhouettes('200 nDistMat AbsP Syntax Forwards_PCA.csv','C:/Users/Andrew/workspace/DissWork/nAcrossK_PCA/',k='multi')        
#metaCluster('C:/Users/Andrew/workspace/DissWork/nAcrossK_PCA/',10)
#matrixSimCaller(10, 'C:/Users/Andrew/workspace/DissWork/Abs Syntax Forwards_rev/', 200, 10)
#TPDmatrixSim('C:/Users/Andrew/workspace/DissWork/Abs Syntax Forwards_rev/',200,10,meth='naive',reduc='PCA')
#rareSDSFixer('C:/Users/Andrew/workspace/DissWork/Abs Syntax Backwards/')
#TPDentropyCluster('C:/Users/Andrew/workspace/DissWork/Abs Syntax Forwards_rev/',5)
#bigramTPDcluster('C:/Users/Andrew/workspace/DissWork/100 AbsP Syntax Forwards/',10)#memory hog!