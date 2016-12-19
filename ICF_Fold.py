import numpy as np
from gurobipy import *
import math
import csv
import arff
import time
import ICF_Purity



# This function is for separating dataset to training and test sets
def traTestBol(a,indxForTenFold,wFold,TotalFold):

    dTest=[]
    dTrain=[]

    IOindx = np.ones(len(a))

    for i in range(0,len(indxForTenFold)):
        for j in range(0,len(indxForTenFold[i])):
            if j%TotalFold==wFold-1:
                dTest.append(a[indxForTenFold[i][j]])
                IOindx[indxForTenFold[i][j]]=0
            else:
                dTrain.append(a[indxForTenFold[i][j]])


    dTest=np.array(dTest)
    dTrain=np.array(dTrain)



    return dTrain, dTest,IOindx


def prepIndx(a):

    sinifS = int(np.max(a[:,-1]))
    rowS=len(a)
    col=len(a[0])

    indxes=[]

    for i in range(0,sinifS):
        indxes.append([])

    for i in range(0,rowS):
        indxes[int(a[i,col-1]-1)].append(i)

    return indxes

# Calculating separation function.Step 2 of Algorithm 1.
def PCFl2( Ajr, Bjr, cjr, purity):

    distb =np.sqrt(np.power(Bjr-cjr,2).sum(axis=1))
    dista =np.sqrt(np.power(Ajr-cjr,2).sum(axis=1))


    gamma=(np.max(dista)+np.min(distb))/2.0
    return { 'gamma': gamma, 'c':cjr, 'purity':purity}

# Solving P_r. LP model in Step 2 of Algorithm 1.
def PCF(Ajr, Bjr, cjr,status, purity):

    # Create optimization model
    m = Model('PCF')

    # Create variables
    gamma = m.addVar(vtype=GRB.CONTINUOUS, lb=1, name='gamma')
    w = range(nn)
    for a in range(nn):
        w[a] = m.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % a)

    ksi = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='ksi')

    m.update()
    hataA = {}
    hataB = {}

    for i in range(len(Ajr)):
        hataA[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='hataA[%s]' % i)

        m.update()
        m.addConstr(quicksum((Ajr[i][j] - cjr[j]) * w[j] for j in range(len(cjr))) + (ksi * quicksum(math.fabs(Ajr[i][j] - cjr[j]) for j in range(len(cjr)))) - gamma + 1.0 <= hataA[i])

    for z in range(len(Bjr)):
        hataB[z] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='hataB[%s]' % z)

        m.update()
        m.addConstr(quicksum((Bjr[z][r] - cjr[r]) * -w[r] for r in range(len(cjr))) - (ksi * quicksum(math.fabs(Bjr[z][q] - cjr[q]) for q in range(len(cjr)))) + gamma + 1.0 <= hataB[z])



    m.update()

    m.setObjective((quicksum(hataA[k] for k in range(len(hataA))) / len(hataA))+(quicksum(hataB[l] for l in range(len(hataB)))  / len(hataB)), GRB.MINIMIZE)

    m.update()
    # Compute optimal solution
    m.optimize()
    m.write('model.sol')
    status.append(m.Status)
    ww=[]
    for i in range(len(cjr)):
        ww.append(w[i].X)

    return {'s':status,'w': ww, 'gamma': gamma.x, 'ksi': ksi.x, 'c':cjr, 'purity':purity}

def findgj(Aj, centroids, B,status, Rs, Rbs, purity):
    gj=[]

    r=0
    for Ajr in Aj:

        if purity[r]<tolpr:

            newAjr, newB= ICF_Purity.eliminateWithR(Ajr, B, centroids[r], Rs[r], Rbs[r])
            sonuc = PCF(newAjr, newB, centroids[r],status, purity[r])
            status=sonuc['s']
            gj.append(sonuc)
        else:
            newAjr, newB= ICF_Purity.eliminateWithR(Ajr, B, centroids[r], Rs[r], Rbs[r])
            sonuc = PCFl2(newAjr, newB, centroids[r], purity[r])
            gj.append(sonuc)

        r=r+1


    #   parametreleri yazidirmak icin : print sonuc, print sonuc['ksi'], print sonuc['gamma'], print sonuc['w'], print sonuc['c']
    return status,gj


def pcfDeger(w, ksi, gamma, c, x):
    deger = np.dot(w,x-c) + ksi*np.sum(abs(x-c)) -gamma
    return deger

def pcfl2Deger(gamma,c, x):
    deger =np.sqrt(np.sum(np.square(x-c))) - gamma
    return deger

def sinifBul(data):
    sinifTahmini=[]
    g_deger=[]
    for d in data:
        t=1
        enkDeger=float('inf')
        gj_deger=[]
        for gj in g:
            gjr_deger=[]
            for gjr in gj:
                if gjr['purity']<tolpr:

                    fonkDeger = pcfDeger(gjr['w'],gjr['ksi'],gjr['gamma'],gjr['c'],d[0:-1])

                else:
                    fonkDeger= pcfl2Deger(gjr['gamma'],gjr['c'],d[0:-1])

                gjr_deger.append(fonkDeger)
                if (fonkDeger<enkDeger):
                    enkDeger=fonkDeger
                    sinifT=t
            t=t+1
            gj_deger.append(gjr_deger)
        g_deger.append(gj_deger)
        sinifTahmini.append(sinifT)
    return sinifTahmini



# Read arff file
def readArff(dosya):
    d = arff.load(open(dosya, 'rb'))
    v=[]
    for dd in d['data']:
        satir=[]
        for ddd in dd:
            satir.append(float(ddd))
        v.append(satir)
    v=np.array(v)
    return v
# Read csv file
def readData(dosya):
    dosya = open(dosya)
    okuyucu = csv.reader(dosya, quotechar=',')
    data = []

    for row in okuyucu:
        satirVeri = []
        for deger in row:
            satirVeri.append(float(deger))

        data.append(satirVeri)

    data=np.array(data)
    return data

###################### MAIN FUNCTION STARTS HERE ###############################
start_time = time.time()
egtDO=[]
testDO=[]
allFCN=[]

###################### INPUTS ###############################
data = readData('/Users/example.csv') #Dataset path should be gviven here.
#data = readArff('example.arff')
tolpr=0.60 #epsilon 1 in algorithm 1
TF=10 # Means 10 fold cross validation will be done.
######################

mm=len(data)  # row size
nn=len(data[0])-1 # feature size
sinifSayisi = int(np.max(data[:,-1])) # classes must be 1 to n in the last column ...................................

indxForTenFold = prepIndx(data)
status = []
for kfold in range(1,TF+1):

    dataTrain,dataTest, traTest =traTestBol(data,indxForTenFold,kfold,TF)
    g=[]

    for sinif in range(1,sinifSayisi+1): # For each class

        Aj = []
        Bj = []
        for d in dataTrain:
            if d[-1] == sinif:
                Aj.append(d[ 0:-1])
            else:
                Bj.append(d[ 0:-1])


        Aj=np.array(Aj)
        Bj=np.array(Bj)


        centroids, clusters, resR, resRB, purities = ICF_Purity.getPureClusters(Aj, Bj) # Call algorithm 2 here

        status,gj=findgj(clusters, centroids, Bj,status,resR, resRB, purities ) # Calling Algorithm 1, Step 1-2

        g.append(gj)

    #--------------------------------TESTING---------------------------------------------------- '''
    allFCN.append(g)

    sinifTahmini = sinifBul(data)
    gercekSinif = data[:,-1]

    #Calculating training accuracy
    EgitimDogrulukOrani= round(100.0*(np.sum(traTest*(sinifTahmini==gercekSinif)))/len(dataTrain),2)
    egtDO.append(EgitimDogrulukOrani)

    #Calculating test accuracy
    TestDogrulukOrani= round(100.0*(np.sum((1-traTest)*(sinifTahmini==gercekSinif) ))/len(dataTest),2)
    testDO.append(TestDogrulukOrani)

for f in range(1,TF+1):
     print "#######################",f,".Fold #################################"
     print f,".Fold Training Accuracy: %", egtDO[f-1]
     print f, ".Fold Test Accuracy : %" ,testDO[f-1]
     print "-----------------------------------------------------"
     j=1
     for gj in allFCN[f-1]:
        r=1
        print "For class ", j,"the classifiers are:"
        for gjr in gj:
            if gjr['purity'] < tolpr:
                print f,".Fold", j ,".class ", r ,".cluster classification function that separates A from B: gjr = w.(x-c) + ksi*|w.(x-c)|-gamma "
                print "w =", gjr['w']
                print "ksi =", gjr['ksi']
                print "gamma =", gjr['gamma']
                print "center =", gjr['c']
            else:
                print f,".Fold", j,".class ", r, ".cluster classification function that separates A from B: gjr = |x-c|_2  - gamma "
                print "gamma =", gjr['gamma']
                print "center =", gjr['c']
            print "-----------------------------------------------------------"
            r=r+1

        j=j+1

print "##################################################################"
egtDOrani=np.average(egtDO)
print "Training Accuracy : %", egtDOrani

testDOrani=np.average(testDO)
print "Test Accuracy : % ", testDOrani

print "##################################################################"


print("--- %s seconds elapsed ---" % (time.time() - start_time))




