# This file is the implementation of the algorithm 2

from sklearn.cluster import KMeans
import numpy as np

####################### PARAMETERS #############################
# Set Algorithm 2 Inputs
#The parameter defines the ratio of how much of redundant data are to be eliminated from set A.
# As tau1 increase, more data elimination occurs.
# Similarly,tau2 defines the data elimination ratio for set B.
# If more data is eliminated from sets, the operations become fast while reducing the training accuracy.
# The other two parameters,tau3 and tau4, define thresholds to increase/decrease number of sub-sets.
# tau1 and tau2 is from [0, inf]; tau3 and tau4 is from [0, 1]
tau1=0.9
tau2=1.1
tau3=0.95
tau4=0.05

################################################################

#This function is the implementation of Algorithm 2 Step 3
#Inputs:
# set A: the class to be classified
# set B: rest of the all classes
# c: center of the cluster
# r: radius for set A
# rb: radius for set B
# Output: Eliminated sets; A and B
def eliminateWithR(A, B, c, r, rb):

    bordA=r*tau1
    bordB=rb*tau2


    elimA=[]
    elimB=[]

    dista =np.sqrt( np.power(A-c,2).sum(axis=1))
    distb =np.sqrt(np.power(B-c,2).sum(axis=1))

    if len(A)>30:
        for i in range(len(A)) :
            if (dista[i] > bordA)and (dista[i] < bordB) :
                elimA.append(A[i][:])
    else :
        elimA=A

    if len(elimA)==0:
        elimA=A

    for i in range(len(B)):
        if (distb[i] < bordB)and (distb[i] > bordA):
            elimB.append(B[i][:])

    if len(elimB)<30:
        elimB=B


    return elimA, elimB


#Starting of Algorithm 2. Step 1 implementation
#Inputs:
# set A: the class to be classified
# set B: rest of the all classes
def getPureClusters(A, B):

    stop=0
    clusterNum=2
    tol = tau3
    cont=1

    while((cont==1) and (stop==0)):

        cont=0
        kmeans = KMeans(init='k-means++', n_clusters=clusterNum, n_init=1)
        kmeans.fit(A)

        centroids = kmeans.cluster_centers_
        clusters = kmeans.labels_

        Aj=[]
        ResR=[]
        ResRB=[]
        pr=np.zeros((clusterNum,1))

        for i in range(len(centroids)):
            Aj.append([])



        for index in range(len(clusters)):
            Aj[clusters[index]].append(A[index])


        for i in range(clusterNum):

            pr[i], radi, radib = purity(Aj[i], B, centroids[i])
            ResR.append(radi)
            ResRB.append(radib)


            if (len(Aj[i])<len(A)*tau4):
                stop=1

        prSum=0
        for i in range(len(Aj)):
            prSum=prSum+(pr[i]*len(Aj[i]))

        if ((prSum/len(A)) < tol):
            cont=1

        clusterNum=clusterNum+1

    DpntClusters=[]
    for i in range(len(centroids)):
        DpntClusters.append([])

    for index in range(len(clusters)):
        DpntClusters[clusters[index]].append(A[index])



    return centroids, DpntClusters, ResR, ResRB, pr

# This function is the implementation of Algorithm 2 Step 2
# set A: the class to be classified
# set B: rest of the all classes
# c: center of the cluster
# Output: puri: purity rate of the cluster and set A-B radiuses
def purity (a, b, c):


    dista =np.sqrt(np.power(a-c,2).sum(axis=1))
    rada = np.mean(dista)
    limit = np.max(dista)

    distb =np.sqrt(np.power(b-c,2).sum(axis=1))
    insay = np.zeros((len(distb),1))

    radb = np.mean(distb)
    insay[(distb<limit)]=1

    puri =len(a)/(len(a)+insay.sum())


    return puri, rada, radb




