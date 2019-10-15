# coding=utf-8
from __future__ import division

from scipy import stats
from scipy.stats import norm
import scipy.stats as st
from subprocess import call
import numpy as np
import sys
import re
import math
##import rpy2.robjects as robjects

from subprocess import Popen, PIPE


##WEB10k
#my %hsNdcgRelScore = (  "4", 15,
#						"3", 7,
#						"2", 3,
#                        "1", 1,
#                        "0", 0,
#
#### LETOR 3
#hash table for NDCG,
#my %hsNdcgRelScore = (  "2", 3,
#                        "1", 1,
#                        "0", 0,
#

def readingFile (file):

    line= None
    AntGenes = ""
    ap = []

    with open(file, 'r') as f:
        for line in f:
            m = re.search('.*\s*mean=*>(.*)\s', line)
            if m:
                ap.append(float(m.group(1)))


    return np.asarray(ap)

def getMeanRiskBaseline(mat): ##this fuction servers to get the average prediction from features

    baseline =  np.array([0.0]* mat.shape[0])
    for i in range(mat.shape[0]):
        baseline[i]=np.mean(mat[i,:])

    return baseline


def getMaxRiskBaseline(mat): ##this fuction servers to get the feature with better prediction
    baseline = np.array([0.0] * mat.shape[0])
    for i in range(mat.shape[0]):
        baseline[i] = np.max(mat[i, :])

    return baseline

def getFullBaselineByFold(coll, l2r, fold):


    fileName = "/home/daniel/Dropbox/WorkingFiles/L2R_Baselines/" + coll + "." + l2r + ".ndcg.test.Fold" + str(fold)
    ndcgBaseline = readingFile(fileName)


    return np.asarray(ndcgBaseline)

def getFullBaseline(coll, l2r, MAX_Fold):

    ndcgBaseline = []
    for fold in range(1, MAX_Fold):

        fileName = "/home/daniel/Dropbox/WorkingFiles/L2R_Baselines/" + coll + "." + l2r + ".ndcg.test.Fold" + str(fold)
        temp = readingFile(fileName)
        ndcgBaseline = ndcgBaseline + temp.tolist()

    return np.asarray(ndcgBaseline)

def getGeoRisk(mat, alpha ):
    ##### IMPORTANT
    # This function takes a matrix of number of rows as a number of queries, and the number of collumns as the number of systems.
    ##############
    numSystems = mat.shape[1]
    numQueries = mat.shape[0]
    Tj = np.array([0.0]* numQueries)
    Si = np.array([0.0] * numSystems)
    geoRisk = np.array([0.0] * numSystems)
    zRisk = np.array([0.0] * numSystems)
    mSi = np.array([0.0] * numSystems)

    for i in range(numSystems):
        Si[i]=np.sum(mat[:,i])
        mSi[i] = np.mean(mat[:, i])

    for j in range(numQueries):
        Tj[j]=np.sum(mat[j,:])

    N = np.sum(Tj)

    for i in range(numSystems):
        tempZRisk=0
        for j in range(numQueries):
            eij=Si[i]*(Tj[j]/N)
            xij_eij= mat[j,i]-eij
            if eij!=0:
                ziq = xij_eij / math.sqrt(eij)
            else:
                ziq = 0
            if xij_eij < 0:
                ziq=(1+alpha)*ziq
            tempZRisk=tempZRisk+ziq
        zRisk[i]=tempZRisk

    c = numQueries
    for i in range(numSystems):
        ncd=norm.cdf(zRisk[i] / c)
        geoRisk[i]=math.sqrt( (Si[i]/c) *ncd)

    return geoRisk

def getNdcgRelScore(dataset, label):
    web10k = np.array([0,1,3,7,15])
    letor = np.array([0, 1, 3])
    if dataset == "web10k":
        return web10k[label]
    elif dataset == "letor":
        return letor[label]

def relevanceTest(dataset, value):
    #y % hsPrecisionRel = ("4", 1,  #"3", 1, #"2", 1, # "1", 0, #"0", 0, )
    if dataset == "web10k":
        if value > 1:
            return 1
        else:
            return 0

    elif dataset == "letor":
        #my % hsPrecisionRel = ("2", 1, # "1", 1, # "0", 0 # )

        if value > 0:
            return 1
        else:
            return 0
    elif dataset == "yahoo":
        if value > 0:
            return 1
        else:
            return 0

    return 0

def getQueries(query_id_train):
    queryList = []
    queriesList = []
    # query_id_train.append(-1)
    idQ = -1
    cDoc = 0
    for i in query_id_train:
        if idQ != i:
            if (len(queryList) > 0):
                queriesList.append(queryList[:])
                del queryList[:]

        idQ = i
        queryList.append(cDoc)
        cDoc = cDoc + 1

    queriesList.append(queryList)
    return queriesList

def average_precision(arrayLabel, dataset):

    avgPrecision=0
    numRelevant=0
    iPos=0
    for prec in arrayLabel:
        if relevanceTest(dataset, prec) == 1:
            numRelevant +=1
            avgPrecision += (numRelevant / float(iPos+1))
        iPos+=1

    if numRelevant == 0:
        return 0.0

    return round(avgPrecision / float(numRelevant), 4)

def dcg(topN, arrayLabel, dataset):

    totalDocs = arrayLabel.shape[0]
    vetDCG = np.array([0.0] * totalDocs, dtype=float)
    #vetDCG = np.array([0.0]*topN, dtype=float)
    vetDCG[0] = getNdcgRelScore(dataset, arrayLabel[0])
    #totalDcos = arrayLabel.shape[0]
    for iPos in range(1, totalDocs):
    #for iPos in range(1, topN):
        r=0
        if (iPos < totalDocs ):
            r = arrayLabel[iPos]
        else:
            r = 0
        if (iPos < 2):
            vetDCG[iPos] = vetDCG[iPos-1] + getNdcgRelScore(dataset, r)
        else:
            vetDCG[iPos] = vetDCG[iPos-1] + (getNdcgRelScore(dataset, r) * math.log(2) /math.log(iPos +1))
    return vetDCG

def ndcg(arrayLabel, dataset):

    #topN = arrayLabel.shape[0]
    topN = 10
    #vetNDCG = np.array([0.0] * arrayLabel.shape[0])

    vetDCG = dcg(topN, arrayLabel, dataset)
    #print vetDCG
    stRates = np.sort(arrayLabel)[::-1]

    bestDCG = dcg(topN, stRates, dataset)

    NDCGAt10=0
    if topN > vetDCG.shape[0]:
        return 0.0
    if (bestDCG[topN-1] != 0):
            NDCGAt10 = vetDCG[topN-1]/bestDCG[topN-1]


    return round(NDCGAt10, 4)
    #iPos=0
    #while (iPos < topN and iPos < arrayLabel.shape[0] ):
    #    vetNDCG[iPos]=0
    #    if (bestDCG[iPos] != 0):
    #        vetNDCG[iPos] = vetDCG[iPos]/bestDCG[iPos]
    #    iPos+=1
    #return vetNDC

def getRisk(prec, precBaseline):
    vetRisk = []
    idQ = 0
    for my in prec:
        delta= precBaseline[idQ] - my

        idQ=idQ+1

        if(delta < 0):
            delta = 0
        vetRisk.append(delta)

    return np.asarray(vetRisk)

def gettingPValueFromTRisk (aps, anchor, base,alpha, tAnchor):
    vetDelta = []
    i = 0
    for my in aps:
        delta = my - base[i]

        i = i + 1

        if (delta < 0):
            delta = (1 + alpha) * delta

        vetDelta.append(delta)

    vetAnchor = []
    i=0
    for anc in anchor:
        delta = anc - base[i]

        i = i + 1

        if (delta < 0):
            delta = (1 + alpha) * delta

        vetAnchor.append(delta)

    p = gettingTTestR(vetAnchor, vetDelta)

   #if math.isnan(p):
    #    aux = "="
    #elif p > 0.05:
    #    aux = "="
    #else:
    #    aux = "!"

    #if tAnchor > 0:
        #aux = aux + str("{0:.3f}".format(p))

    return p

def getTRisk(aps, base, alpha):
    vetDelta= []
    i=0
    for my in aps:
        delta=my - base[i]

        #print str(delta) + "= " + str(my) + " - " + str(base[i])
        i=i+1


        if(delta < 0):
            delta=(1+alpha)*delta

        vetDelta.append(delta)

    #print "("+str(sum(aps))+" "+ str(sum(base))+") ",
    #print "("+str(sum(vetDelta))+")",
    #print alpha,
    urisk=np.mean(vetDelta)
    #print "Test:", urisk, np.mean(aps), np.mean(base), alpha, len(vetDelta), len(aps), len(base)
    #print "Trisk", np.sum((vetDelta-urisk))
    if urisk==0:
        return 0.0, vetDelta
    SEurisk= stats.sem(vetDelta)

    return urisk / SEurisk, vetDelta
    #print "My function", urisk/SEurisk
    #This part of the code is available in JForest, implemeting Trisk
    #sum = np.sum(vetDelta)
    #SSQR =0
    #for i in vetDelta:
    #    SSQR = SSQR + (i * i)

    #SQRS = sum * sum
    #c = len(vetDelta)
    #if SQRS == SSQR:
    #    print "Their fuction", 0
    #else:
    #    pairedVar= (SSQR - (SQRS / c)) / (c - 1)
    #    trisk = math.sqrt(c / pairedVar) * urisk
    #    print "Their fuction", trisk
#def modelRiskEvaluation(myModel, baselineModel, metric="trisk"):

#    result = []

#    if (metric == "trisk"):
#        result = tRisk(myModel, baselineModel, 5)
#    elif (metric == "risk"):
#        result = risk(myModel, baselineModel)

#    return result

def modelEvaluationScript_DEPRECATED(scoreList, featureFile, metric ):

    MAP = []

    if metric =="NDCG":
        indexToGet = 10
    else:
        indexToGet = 1

    scoreFile = "L2R.scoreGA"

    # print "Metric", metric, "index", indexToGet
    with open(scoreFile, "w") as outs:
        for s in scoreList:
            outs.write(str(float(s)) + "\n")

    #call(["perl", "eval_score_LBD.pl", featureFile, scoreFile, "L2R.mapGA", "1"])
    call(["perl", "/home/daniel/programs/eval_score_LBD.pl", featureFile, scoreFile, "L2R.mapGA", "1"])
    startFlag = 0
    with open("L2R.mapGA", "r") as ins:
        for line in ins:

            searchObj = re.match(r'Average' + re.escape(metric), line, re.M | re.I)
            if searchObj:
                break

            searchObj = re.match(r'qid\s' + re.escape(metric) + r'.*', line, re.M | re.I)
            if searchObj:
                startFlag = 1
                continue

            if (startFlag == 1):
                lineList = line.split("\t")

                a = lineList[indexToGet]
                b = lineList[indexToGet].rstrip()
                MAP.append(float( lineList[indexToGet].rstrip()) )

                # print ("Linha:"+line)
                #MAP = MAP + float(lineList[indexToGet].rstrip())

    return MAP

def modelEvaluation(test, score, nFeatures):

    dataset =""
    if nFeatures == 136:
        dataset="web10k"
    elif nFeatures == 4:
        dataset="letor"
    elif nFeatures == 64:
        dataset="letor"
    elif nFeatures == 700:
        dataset="web10k"
    else:
        print ("There is no evalution to this dataset, numFeatures: ", nFeatures)
        exit(0)

    listQueries = getQueries(test.q)

    lineNum = np.array(range(0, len(test.y) ), dtype=int)
    #print (len(score), len(lineNum))
    #a = np.reshape(score,-1)
    #b = np.reshape(lineNum, -1)
    #c = (   np.reshape(score,-1) , np.reshape(lineNum,-1))
    #d = np.vstack((   np.reshape(score,-1) , np.reshape(lineNum,-1)) )
    mat = (np.vstack((   np.reshape(score,-1) , np.reshape(lineNum,-1)) )).T
    apQueries = np.array([0.0]*len(listQueries), dtype=float)
    ndcgQueries = np.array([0.0] * len(listQueries), dtype=float)

    idQ=0
    for query in listQueries:
        matQuery = mat[query]
        matQuery = matQuery[np.argsort(-matQuery[:, 0], kind="mergesort")]
        labelQuery = np.array([0] * matQuery.shape[0], dtype=int)

        i=0
        for doc in matQuery:
            labelQuery[i] = test.y[int(doc[1])]
            i+=1

        apQueries[idQ]=average_precision(labelQuery, dataset)
        #print apQueries[idQ]
        ndcgQueries[idQ]=ndcg(labelQuery, dataset)
        #print "NDCG", ndcgQueries[idQ]
        idQ+=1



    return ndcgQueries, apQueries

def getConfidentValues (a):
    #error < - qt(0.975, df=length(w1$V1)-1)*sd(w1$V1) / sqrt(length(w1$V1))
    mean, sem, m = np.mean(a), st.sem(a), st.t.ppf((1 + 0.95) / 2., len(a) - 1)

    return "\pm"+str(m*sem)


def gettingTTestR (x_vet, y_vet):

    rd1 = (robjects.FloatVector(x_vet))
    rd2 = (robjects.FloatVector(y_vet))
    rvtest = robjects.r['t.test']
    pvalue = rvtest(rd1, rd2, paired=True)[2][0]
    #print rvtest(rd1, rd2, paired=True)




    #line = str(rvtest(rd1, rd2, paired=True))
    #text= line.split("\n")
    #pvalue= text[-6]





    #pvalue = rvtest(rd1, rd2, paired=True)

    #print "[", pvalue, "]"
    #print np.mean(x_vet), np.mean(y_vet), pvalue
    #pvalue = rvtest(rd1, rd2, paired=True)

    return pvalue

def obtainGeoRiskMatrix(l2rResult, coll, fold):

    l2rBaselines = ["rf", "ada", "listnet", "gbrt", "lm" ]

    if l2rResult in l2rBaselines:
        l2rBaselines.remove(l2rResult)

    finalList = []
    for l2r in l2rBaselines:
        fileName = "/home/daniel/Dropbox/WorkingFiles/L2R_Baselines/" + coll + "."+l2r+".ndcg.test.Fold"+str(fold)
        temp = readingFile(fileName)
        finalList.append(temp)


    finalList =np.asarray(finalList)

    return finalList.T

def gettingWinsLosses (myModel, baseModel):
    wins = 0
    losses = 1
    tieds = 0
    i = 0
    size = len(baseModel)

    for i in range(0, size):
        if (baseModel[i] > myModel[i]):
            losses=losses+1
        elif( baseModel[i] < myModel[i]):
            wins=wins+1
        else:
            tieds=tieds+1

    t = losses-1 + wins + tieds
    if (t !=size):
        print ("ERROR!!! The mount of wins($wins) + losses($losses) + tieds($tieds) should be igual to $size ")



    return wins/losses

def gettingLossGreater20Perc (myModel, baseModel):
    wins = 0
    losses = 1
    tieds = 0
    i = 0
    size = len(baseModel)
    losses=0
    for i in range(0, size):
        if (baseModel[i] > myModel[i]):
            perc = (baseModel[i] - myModel[i]) / baseModel[i]
            if perc >= 0.2:
                losses=losses+1
    return losses

def gettingWins (myModel, baseModel):
    wins = 0
    losses = 1
    tieds = 0
    i = 0
    size = len(baseModel)

    for i in range(0, size):
        if (baseModel[i] > myModel[i]):
            losses=losses+1
        elif( baseModel[i] < myModel[i]):
            wins=wins+1
        else:
            tieds=tieds+1

    t = losses-1 + wins + tieds
    if (t !=size):
        print ("ERROR!!! The mount of wins($wins) + losses($losses) + tieds($tieds) should be igual to $size ")


    return wins


if __name__ == "__main__":

    init = np.array([0.0500, 0.1500, 0.3000, 0.4500, 0.5500,0.4000, 0.3500, 0.3000, 0.2500, 0.2000,0.3000, 0.3000, 0.3000, 0.3000, 0.3000,0.2500, 0.2500, 0.2500, 0.2500, 0.2500,0.4000, 0.1500, 0.4000, 0.1500, 0.4000,0.2000,0.4500,0.2000, 0.4500, 0.2000,0.2542, 0.2629, 0.2802, 0.2975, 0.3061,0.2918, 0.2994, 0.3147, 0.3301, 0.3378])
    init =np.reshape(init, (8,5))
    init = init.T
    print (init)
    geoRisk= getGeoRisk(init, 1)
    print (geoRisk)

    #trisk=tRisk(init[:,1], init[:,2], 1)
    #print trisk
    #print tRiskNew(init[:,1], init[:,2], 1)
        ###queries_train = getQueries(query_id_train)
        ###queries_test = getQueries(query_id_test)
        ###score = read_score(scoreFile)
        ### ndcg_at = int(sys.argv[3])


        #ndcg , map = modelEvaluation( y_test, score, query_id_test, 136)
        #print "NDCG ", np.mean(ndcg)

        #print_metrics(rbq=rank_by_query, rj="higher", file=out_file, miss_relevant=miss_relevant, equation=equation)