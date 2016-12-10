import csv

"""
Classifier models for tagging low-probability chords
based on categories assembled from flattened agglomerative clustering (yjampClus.py)
I. k-nearest neighbors
II. 
"""

###########################
#I. k-nearest neighbors stuff
###########################
def knn_testClass(r_state,verbose=False,**kwargs):
    '''Trains on top 200 chord flat clustering and attempts to label next n chords
    
    1. From flat agglom clus, use sklearn to sep training/testing sets with clus num labels
    (use PCA-red, ordered TPMs as basis)
    2. Flatten those matrices and get a sense of what happens for different k (knn)
    2. Use model to tag next n (outside the top200) chords by clus number
    '''
    import numpy as np
    from sklearn.cross_validation import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    
    #grab the flat clustering for top200 chords
    #numChords by numWindows array of sample data
    samples = []
    #numChords-length vector of cluster tags
    tags = []
    flat_clus_path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/Categories chapter/truncDend_memb.csv'
    all_labels = csv.reader(open(flat_clus_path,'r',newline='\n'))
    for i,row in enumerate(all_labels):
        if i==0: continue #skip header row
        tags.append(row[0])
        sf = csv.reader(open('dcMats_PCAord/'+row[1]+'.csv','r',newline='\n'))
        samp = np.array([r for r in sf])
        samples.append(samp.flatten())
    #print(samples[0])
    
    #auto-separate training/testing sets from full tagged sample
    x_tr,x_ts,y_tr,y_ts = train_test_split(samples,tags,test_size=0.3,random_state=r_state)
    #print(y_tr)
    
    #fit knn model, return score
    clf = KNeighborsClassifier(**kwargs)
    clf.fit(x_tr,y_tr)
    if verbose:
        print(x_ts[0],clf.predict_proba(x_ts[0]))
    #score the trained model on the held-out testing data
    return(clf.score(x_ts,y_ts))

def knn_scorer(n):
    '''Calls knn_testClass() to score n random trials with 1 < k < 10
    Output: seaborn plot of accuracy versus k
    '''
    import seaborn as sms    
    from matplotlib import pyplot as plt
    plt.subplot(111)
    
    j=0
    while j < n:
        kvals, scs = [],[]
        for k in range(1,10):
            sc = knn_testClass(j,n_neighbors=k,p=1,metric='minkowski')
            #print([k,sc])
            kvals.append(k)
            scs.append(sc)
        plt.plot(kvals, scs,label='Trial '+str(j))
        j += 1
        
    plt.legend(loc="upper left",bbox_to_anchor=(1.05, 1.))
    plt.title('Accuracy for k-nearest neighbors')
    plt.xlabel('k (neighbors compared)')
    plt.ylabel('Score')
    #display the plot
    plt.show()
    
def TPD_PCA_lowP(mx):
    '''Simple script to pull lower-prob scale degree sets for later classification
    Outputs: temporal probability matrices (TPM) with basis of top200 chords in 3-comp PCA coords
    '''
    from yjampClus import getOrderedTPDdata
    
    #figure out which chords are in the topN most probable for keeping
    allChords = csv.reader(open('50ms 3 SDSets.csv','r',newline='\n'))
    allChordsList = []#the names of the topN origin chords
    for i, row in enumerate(allChords):
        #Make a list of the top200 most unigram-probable sds
        if i < 200:
            allChordsList.append(row[0])
            continue
        #if the chord is a lowP SDS, get data for the origin chord (oc)
        originpath = 'C:/Users/Andrew/workspace/DissWork/Abs Syntax Forwards_rev/'+row[0]+' SDs prog probs 50ms.csv'
        ocTPD = getOrderedTPDdata(originpath, allChordsList, reduc='PCA',rowtype='t')
        destpath = 'C:/Users/Andrew/workspace/Disswork/dcMats_PCAord_lowP/'+row[0]+'.csv'
        lw = csv.writer(open(destpath, 'w',newline='\n'))
        for row in ocTPD:
            lw.writerow(row)
        if i > mx - 2:
            break
        #print(len(ocTPD),len(ocTPD[0]))
        
def knn_predicter(chd,**kwargs):
    '''Using knn trained on top200 SDS flat clustering tags, predicts cluster for lower-P chd'''
    import numpy as np
    from sklearn.cross_validation import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    
    #grab the flat clustering for top200 chords; this is the training set for the model
    #numChords by numWindows array of sample data
    samples = []
    #numChords-length vector of cluster tags
    tags = []
    flat_clus_path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/Categories chapter/truncDend_memb.csv'
    all_labels = csv.reader(open(flat_clus_path,'r',newline='\n'))
    for i,row in enumerate(all_labels):
        if i==0: continue #skip header row
        tags.append(row[0])
        sf = csv.reader(open('dcMats_PCAord/'+row[1]+'.csv','r',newline='\n'))
        samp = np.array([r for r in sf])
        samples.append(samp.flatten())
    #print(len(samples),samples[0])
        
    #grab the TPM data for the lowP chord to be classified
    sf = csv.reader(open('dcMats_PCAord_lowP/'+chd+'.csv','r',newline='\n'))
    samp = np.array([r for r in sf])
        
    #predict tag for chd from trained knn classifier
    clf = KNeighborsClassifier(**kwargs)
    clf.fit(samples,tags)
    return(chd,clf.predict(samp.flatten()))

###########################
#II. Decision tree and Random Forest
###########################
def decision_class(r_state,verbose=False,forest=False,**kwargs):
    """A (predictably terrible) decision tree classifier for high-P chords
    if forest==True, employs random forest (default, 10 trees)"""
    import numpy as np
    from sklearn.cross_validation import train_test_split
    from sklearn import tree
    from sklearn import ensemble
    
    #grab the flat clustering for top200 chords
    #numChords by numWindows array of sample data
    samples = []
    #numChords-length vector of cluster tags
    tags = []
    flat_clus_path = 'C:/Users/Andrew/Documents/DissNOTCORRUPT/Categories chapter/truncDend_memb.csv'
    all_labels = csv.reader(open(flat_clus_path,'r',newline='\n'))
    for i,row in enumerate(all_labels):
        if i==0: continue #skip header row
        tags.append(row[0])
        sf = csv.reader(open('dcMats_PCAord/'+row[1]+'.csv','r',newline='\n'))
        samp = np.array([r for r in sf])
        samples.append(samp.flatten())
    #print(samples[0])
    
    #auto-separate training/testing sets from full tagged sample
    x_tr,x_ts,y_tr,y_ts = train_test_split(samples,tags,test_size=0.3,random_state=r_state)
    #print(y_tr)
    
    #fit knn model, return score
    if forest:
        clf = ensemble.RandomForestClassifier(**kwargs)
    else:
        clf = tree.DecisionTreeClassifier(**kwargs)
    clf.fit(x_tr,y_tr)
    if verbose:
        #print(x_ts[0],clf.predict_proba(x_ts[0]))
        print(clf.score(x_ts,y_ts))
    #score the trained model on the held-out testing data
    return(clf.score(x_ts,y_ts))


decision_class(1,verbose=True,forest=True,min_samples_split = 5,n_estimators=100)
#knn_testClass(0,n_neighbors=1,p=1,metric='minkowski') 
#knn_scorer(10)