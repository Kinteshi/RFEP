import sys

import numpy as np
import re
from subprocess import call
from sklearn import model_selection
from sklearn import linear_model
from ScikitLearnModificado.forest import Forest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import math
import time
from scipy.stats import norm
from cuml import LinearRegression as cuLinearRegression

def load_L2R_file(TRAIN_FILE_NAME, MASK, sparse=False):
    nLines = 0
    nFeatures = 0
    #### GETTING THE DIMENSIONALITY
    trainFile = open(TRAIN_FILE_NAME, "r")
    for line in trainFile:
        nLines = nLines + 1
    trainFile.seek(0)
    nFeatures = MASK.count('1')

    #### FILLING IN THE ARRAY
    x_train = np.zeros((nLines, nFeatures))
    y_train = np.zeros((nLines))
    q_train = np.zeros((nLines))
    maskList = list(MASK)
    numFeaturesList = []
    iL = 0
    web10k = False
    if len(MASK) == 136:
        web10k = True
    for line in trainFile:
        if web10k:
            m = re.search(r"(\d)\sqid:(\d+)\s(.*)\s#.*", line[:-1]+"#docid = G21-63-2483329\n")
        else:
            m = re.search(r"(\d)\sqid:(\d+)\s(.*)\s#.*", line)

        featuresList = (re.sub(r"\d*:", "", m.group(3))).split(" ")
        if sparse:
            numFeaturesList = (re.sub(r":.?[0-9]+.?[0-9]+\s", " ", m.group(3) + " ")).split(" ")
        y_train[iL] = m.group(1)
        q_train[iL] = m.group(2)

        colAllFeat = 0
        colSelFeat = 0
        for i in featuresList:
            if maskList[colAllFeat] == "1":
                if sparse:
                    x_train[iL][int(numFeaturesList[colAllFeat]) - 1] = float(i)
                else:
                    x_train[iL][colSelFeat] = float(i)
                colSelFeat = colSelFeat + 1
            colAllFeat = colAllFeat + 1
        iL = iL + 1

    trainFile.close()
    return x_train, y_train, q_train


def getNdcgRelScore(dataset, label):
    web10k = np.array([0, 1, 3, 7, 15])
    letor = np.array([0, 1, 3])
    if dataset == "web10k":
        return web10k[label]
    elif dataset == "letor":
        return letor[label]


def relevanceTest(dataset, value):
    # y % hsPrecisionRel = ("4", 1,  #"3", 1, #"2", 1, # "1", 0, #"0", 0, );
    if dataset == "web10k":
        if value > 1:
            return 1
        else:
            return 0

    elif dataset == "letor":
        # my % hsPrecisionRel = ("2", 1, # "1", 1, # "0", 0 # );
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


def average_precision(arrayLabel, dataset):
    avgPrecision = 0
    numRelevant = 0
    iPos = 0
    for prec in arrayLabel:
        if relevanceTest(dataset, prec) == 1:
            numRelevant += 1
            avgPrecision += (numRelevant / float(iPos + 1))
        iPos += 1

    if numRelevant == 0:
        return 0.0

    return round(avgPrecision / float(numRelevant), 4)


def dcg(topN, arrayLabel, dataset):
    totalDocs = arrayLabel.shape[0]
    vetDCG = np.array([0.0] * totalDocs, dtype=float)
    # vetDCG = np.array([0.0]*topN, dtype=float)
    vetDCG[0] = getNdcgRelScore(dataset, arrayLabel[0])
    # totalDcos = arrayLabel.shape[0]
    for iPos in range(1, totalDocs):
        # for iPos in range(1, topN):
        r = 0
        if (iPos < totalDocs):
            r = arrayLabel[iPos]
        else:
            r = 0
        if (iPos < 2):
            vetDCG[iPos] = vetDCG[iPos - 1] + getNdcgRelScore(dataset, r)
        else:
            vetDCG[iPos] = vetDCG[iPos - 1] + (getNdcgRelScore(dataset, r) * math.log(2) / math.log(iPos + 1))
    return vetDCG


def ndcg(arrayLabel, dataset):
    # topN = arrayLabel.shape[0]
    topN = 10
    # vetNDCG = np.array([0.0] * arrayLabel.shape[0])

    vetDCG = dcg(topN, arrayLabel, dataset)
    # print vetDCG
    stRates = np.sort(arrayLabel)[::-1]

    bestDCG = dcg(topN, stRates, dataset)

    NDCGAt10 = 0
    if topN > vetDCG.shape[0]:
        return 0.0
    if (bestDCG[topN - 1] != 0):
        NDCGAt10 = vetDCG[topN - 1] / bestDCG[topN - 1]

    return round(NDCGAt10, 4)


def getEvaluation(score, listQ, label, trainFile, metric, resultPrefix):
    dataset = ""
    if "web10k" in trainFile:
        dataset = "web10k"
    elif "dataset" in trainFile:
        dataset = "letor"
    elif "yahoo" in trainFile:
        dataset = "web10k"
    elif "lastfm" in trainFile:
        dataset = "web10k"
    elif "movielens" in trainFile:
        dataset = "web10k"
    elif "youtube" in trainFile:
        dataset = "web10k"
    elif "bibsonomy" in trainFile:
        dataset = "web10k"
    else:
        print("There is no evaluation to this dataset, dataFile: ", trainFile)
        exit(0)

    listQueries = getQueries(listQ)

    lineNum = np.array(range(0, len(label)), dtype=int)
    mat = (np.vstack((np.reshape(score, -1), np.reshape(lineNum, -1)))).T
    apQueries = np.array([0.0] * len(listQueries), dtype=float)
    ndcgQueries = np.array([0.0] * len(listQueries), dtype=float)

    idQ = 0
    MAP = 0
    for query in listQueries:
        matQuery = mat[query]
        matQuery = matQuery[np.argsort(-matQuery[:, 0], kind="mergesort")]
        labelQuery = np.array([0] * matQuery.shape[0], dtype=int)

        i = 0
        for doc in matQuery:
            labelQuery[i] = label[int(doc[1])]
            i += 1

        apQueries[idQ] = average_precision(labelQuery, dataset)
        # print apQueries[idQ]
        ndcgQueries[idQ] = ndcg(labelQuery, dataset)
        # print "NDCG", ndcgQueries[idQ]
        idQ += 1

    if "NDCG" in metric or "ndcg" in metric:
        for predic in ndcgQueries:
            MAP = MAP + predic

    return MAP / idQ, ndcgQueries
    # return ndcgQueries, apQueries


def ____getEvaluationOLD(scoreList, featureFile, metric, resultPrefix, printFold):
    searchObj = re.match(r".*@.*", metric, re.M | re.I)
    indexToGet = 1
    if searchObj:
        searchObj = re.search(r"(.*)@(.*)", metric, re.M | re.I)
        metric = searchObj.group(1)
        indexToGet = int(searchObj.group(2))
    scoreFile = "L2R.scoreGA"

    # print "Metric", metric, "index", indexToGet
    with open(scoreFile, "w") as outs:
        for s in scoreList:
            outs.write(str(float(s)) + "\n")
    call(["perl", "/home/daniel/programs/eval_score_LBD.pl", featureFile, scoreFile, "L2R.mapGA", "1"])
    startFlag = 0
    iAP = 0
    MAP = 0
    with open("L2R.mapGA", "r") as ins:
        for line in ins:
            # print ("line:", line)

            my_reg = r'Average' + re.escape(metric)
            searchObj = re.match(my_reg, line, re.M | re.I)
            if searchObj:
                break

            my_reg = r'qid\s' + re.escape(metric) + r'.*'
            searchObj = re.match(my_reg, line, re.M | re.I)
            if searchObj:
                startFlag = 1
                continue
                # else:
                #   searchObj=re.search( r'Avearge.*', line, re.M|re.I)
                #   startFlag=0

            if (startFlag == 1):
                lineList = line.split("\t");
                # print ("Linha:"+line)
                MAP = MAP + float(lineList[indexToGet].rstrip())
                iAP = iAP + 1

    return (MAP / iAP)


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


def getTheModel(ensemble, ntrees, frate, seed, coll):
    seed = seed + 10

    if ntrees < 100:
        ntrees = 100
    if frate < 0.3:
        frate = 0.3
    if ensemble == 1:
        clf = Forest(n_estimators=ntrees, max_features=frate, max_leaf_nodes=100, min_samples_leaf=1,
                                    random_state=seed, n_jobs=-1)

    if ensemble == 2:
        clf = ExtraTreesRegressor(n_estimators=ntrees, max_leaf_nodes=100, min_samples_leaf=1, random_state=seed,
                                  n_jobs=-1)
    if ensemble == 3:
        clf = GradientBoostingRegressor(n_estimators=ntrees, learning_rate=0.1, max_depth=2, random_state=seed,
                                        loss='ls')
    if ensemble == 4:
        clf = cuLinearRegression(fit_intercept=True, normalize=True,algorithm='eig')
        # clf = linear_model.LinearRegression(n_jobs=-1)

    if ensemble == 5:
        clf = DecisionTreeRegressor(random_state=seed)

    if ensemble == 10:
        if "200" in coll:
            clf = DecisionTreeRegressor(random_state=seed, min_samples_leaf=30, min_samples_split=50, max_depth=3)
        else:
            clf = DecisionTreeRegressor(random_state=seed, min_samples_leaf=50, min_samples_split=100, max_depth=6)

    return clf


def writeOutFeatureFile(mask, fileName):
    maskList = [int(i) for i in list(mask)]

    with open(fileName, "w") as outs:
        id = 1;
        for d in maskList:
            if d == 1:
                outs.write(str(id) + "\n")
            id = id + 1


def executeLambdaMART(trainFile, testFile, mask, fold, METRIC, NTREES):
    out = "outQuickRankFold" + str(fold)
    scoreFile = "scoreFile.Fold" + str(fold)
    # print ['/home/daniel/programs/bin/quicklearn', '--algo', "LAMBDAMART", "--train", trainFile, "--train-metric",
    #          "NDCG", "--train-cutoff", "10", "--test", testFile, "--test-metric", "NDCG", "--test-cutoff", str(NTREES),
    #          "--scores", scoreFile, "--num-trees", "100", "--shrinkage", "0.01", "--tree-depth", "4"]
    with open(out, 'w') as f:
        call(['/home/daniel/programs/quickrank/bin/quicklearn', '--algo', "LAMBDAMART", "--train", trainFile,
              "--train-metric",
              "NDCG", "--train-cutoff", "10", "--test", testFile, "--test-metric", "NDCG", "--test-cutoff", "10",
              "--scores", scoreFile, "--num-trees", str(NTREES), "--shrinkage", "0.01", "--tree-depth", "4"],
             stdout=f)

    result = np.loadtxt(scoreFile)
    return result

    # featureFile= "featureFile.Fold"+str(fold)
    # modelFile= "modelFile.Fold"+str(fold)
    #

    # call (["rm", featureFile, modelFile, scoreFile])

    # writeOutFeatureFile(mask, featureFile)
    # out1 = "out1RankLibFold"+str(fold)

    # with open(out1, 'w') as f:
    #    call(["java", "-jar", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-bag", str(NTREES), "-train", trainFile , "-test", testFile , "-ranker", str(6), "-leaf", str(50), "-shrinkage", str(0.075) ,"-save", modelFile, "-feature", featureFile], stdout=f )
    # with open(out2, 'w') as f:
    #    call(["java", "-jar", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-load", modelFile, "-rank", testFile, "-score", scoreFile, "-feature", featureFile], stdout=f )
    # result=np.loadtxt(scoreFile)
    # return result


def creatingDataset(data, label, query, fileName):
    idx = 0
    with open(fileName, "w") as outs:
        for s in data:

            line = str(int(label[idx])) + " qid:" + str(int(query[idx]))
            idx += 1
            idF = 1
            for f in s:
                line = line + " " + str(idF) + ":" + str(f)
                idF = idF + 1

            line = line + " #docid\n"
            outs.write(line)
    # print("Everythings ok...")
    return fileName


def executeRankSVM(Xtrain, yTrain, qTrain, Xtest, yTest, qTest, mask, fold):
    featureFile = "featureFile.Fold" + str(fold)
    modelFile = "modelFile.Fold" + str(fold)
    scoreFile = "scoreFile.Fold" + str(fold)
    trainFile = "trainFile.Fold" + str(fold)
    testFile = "testFile.Fold" + str(fold)

    call(["rm", featureFile, modelFile, scoreFile, trainFile, testFile])

    creatingDataset(Xtrain, yTrain, qTrain, trainFile)
    creatingDataset(Xtest, yTest, qTest, testFile)

    # writeOutFeatureFile(mask, featureFile)
    out1 = "out1RankLibFold" + str(fold)
    out2 = "out2RankLibFold" + str(fold)

    with open(out1, 'w') as f:
        call(["/home/daniel/programs/svm_rank/svm_rank_learn", "-c", str(40), "-e", str(0.001), "-l", str(1), trainFile,
              modelFile], stdout=f)
        # print (["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-round", str(10), "-train", trainFile , "-test", testFile , "-ranker", str(3), "-save", modelFile])
    with open(out2, 'w') as f:
        call(["/home/daniel/programs/svm_rank/svm_rank_classify", testFile, modelFile, scoreFile], stdout=f)
        # print (["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-load",
        # modelFile, "-rank", testFile, "-score", scoreFile])
    result = np.loadtxt(scoreFile)
    return result


def executeAdaRank(Xtrain, yTrain, qTrain, Xtest, yTest, qTest, mask, fold):
    featureFile = "featureFile.Fold" + str(fold)
    modelFile = "modelFile.Fold" + str(fold)
    scoreFile = "scoreFile.Fold" + str(fold)
    trainFile = "trainFile.Fold" + str(fold)
    testFile = "testFile.Fold" + str(fold)

    call(["rm", featureFile, modelFile, scoreFile, trainFile, testFile])

    creatingDataset(Xtrain, yTrain, qTrain, trainFile)
    creatingDataset(Xtest, yTest, qTest, testFile)

    # writeOutFeatureFile(mask, featureFile)
    out1 = "out1RankLibFold" + str(fold)
    out2 = "out2RankLibFold" + str(fold)

    with open(out1, 'w') as f:
        call(["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-round",
              str(499), "-train", trainFile, "-test", testFile, "-ranker", str(3), "-save", modelFile], stdout=f)
        # print (["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-round", str(10), "-train", trainFile , "-test", testFile , "-ranker", str(3), "-save", modelFile])
    with open(out2, 'w') as f:
        call(["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-load",
              modelFile, "-rank", testFile, "-score", scoreFile], stdout=f)
        # print (["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-load",
        # modelFile, "-rank", testFile, "-score", scoreFile])
    result = np.loadtxt(scoreFile)
    return result


###### ALL BEGNING HERE
# python ga_scikitlearn.py $trainSet $testSet  $valiSet $fold $ntrees $ensemble $subCross $cross $print_fold $paramC $measure $mask


def getGeoRisk(mat, alpha):
    ##### IMPORTANT
    # This function takes a matrix of number of rows as a number of queries, and the number of collumns as the number of systems.
    ##############
    numSystems = mat.shape[1]
    numQueries = mat.shape[0]
    Tj = np.array([0.0] * numQueries)
    Si = np.array([0.0] * numSystems)
    geoRisk = np.array([0.0] * numSystems)
    zRisk = np.array([0.0] * numSystems)
    mSi = np.array([0.0] * numSystems)

    for i in range(numSystems):
        Si[i] = np.sum(mat[:, i])
        mSi[i] = np.mean(mat[:, i])

    for j in range(numQueries):
        Tj[j] = np.sum(mat[j, :])

    N = np.sum(Tj)

    for i in range(numSystems):
        tempZRisk = 0
        for j in range(numQueries):
            eij = Si[i] * (Tj[j] / N)
            xij_eij = mat[j, i] - eij
            if eij != 0:
                ziq = xij_eij / math.sqrt(eij)
            else:
                ziq = 0
            if xij_eij < 0:
                ziq = (1 + alpha) * ziq
            tempZRisk = tempZRisk + ziq
        zRisk[i] = tempZRisk

    c = numQueries
    for i in range(numSystems):
        ncd = norm.cdf(zRisk[i] / c)
        geoRisk[i] = math.sqrt((Si[i] / c) * ncd)

    return geoRisk
