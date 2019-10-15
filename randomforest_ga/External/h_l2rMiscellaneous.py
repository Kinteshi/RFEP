# coding=utf-8
from __future__ import division

import math
import os

import re
from subprocess import call

import numpy as np
from External.h_l2rMeasures import modelEvaluation, getQueries
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


def load_L2R_file(TRAIN_FILE_NAME, MASK):
    nLines = 0
    nFeatures = 0
    #### GETTING THE DIMENSIONALITY

    trainFile = open(TRAIN_FILE_NAME, "r")
    for line in trainFile:
        nLines = nLines + 1
    trainFile.seek(0)
    nFeatures = MASK.count(1)

    #### FILLING IN THE ARRAY
    x_train = np.zeros((nLines, nFeatures))
    y_train = np.zeros((nLines))
    q_train = np.zeros((nLines))
    maskList = list(MASK)
    iL = 0
    for line in trainFile:
        m = re.search(r"(\d)\sqid:(\d+)\s(.*)\s#.*", line)

        featuresList = (re.sub(r"\d*:", "", m.group(3))).split(" ")
        y_train[iL] = m.group(1)
        q_train[iL] = m.group(2)

        colAllFeat = 0
        colSelFeat = 0
        for i in featuresList:
            if maskList[colAllFeat] == 1:
                x_train[iL][colSelFeat] = float(i)
                colSelFeat = colSelFeat + 1
            colAllFeat = colAllFeat + 1
        iL = iL + 1

    trainFile.close()
    return x_train, y_train, q_train


def read_score(path):
    path = open(path, 'r')
    lines = path.readlines()
    for idx in range(len(lines)):
        lines[idx] = float(lines[idx].strip())
    return lines


def getIdFeatureOrder(vet, per, totalFeatures):
    vetOrdinal = np.array(range(0, len(vet)))
    mat = (np.vstack((np.reshape(vet, -1), np.reshape(vetOrdinal, -1)))).T
    mat = mat[np.argsort(mat[:, 0], kind="mergesort")]

    vMask = np.array([0] * totalFeatures, dtype=int)

    id = 0
    if per == 0:
        while mat[id][0] < 1:
            vMask[int(mat[id][1])] = 1
            id = id + 1
    else:
        while id < math.ceil(per * totalFeatures):
            vMask[int(mat[id][1])] = 1
            id = id + 1
    mask = ''.join(str(x) for x in vMask)

    return mask


def executeSckitLearn(l2r, train, test, seed, nTrees):
    seed = seed + 10
    clf = None

    if (l2r == "1" or l2r == "rf"):
        print("RandomForest", seed)
        print(nTrees)
        clf = RandomForestRegressor(n_estimators=int(nTrees), max_features=0.3, max_leaf_nodes=100, min_samples_leaf=1,
                                    random_state=seed, n_jobs=-1)
    elif (l2r == "4" or l2r == "lr"):
        clf = linear_model.LinearRegression()
    elif (l2r == "3" or l2r == "gbrt"):
        # clf = GradientBoostingRegressor(n_estimators=nTrees[0], learning_rate=0.1, max_depth=2, random_state=seed,  loss='ls')
        clf = GradientBoostingRegressor(n_estimators=nTrees[0], learning_rate=nTrees[1], max_depth=2, random_state=seed,
                                        loss='ls')

    clf.fit(train.x, train.y)
    resScore = clf.predict(test.x)

    return resScore


def getL2RPrediction(l2r, fold, train, test, trainFile, testFile, parameters, mask, totalFeatures):
    if mask.count("1") == totalFeatures:
        if l2r == 6 or l2r == "lm":
            scores_test = executeExternalLib(l2r, trainFile, testFile, fold, "NDCG", parameters)
        elif l2r == 7 or l2r == "ada":
            scores_test = executeExternalLib(l2r, trainFile, testFile, fold, "NDCG", parameters)
        elif l2r == 8 or l2r == "listnet":
            scores_test = executeExternalLib(l2r, trainFile, testFile, fold, "NDCG", parameters)
        else:
            scores_test = executeSckitLearn(l2r, train, test, int(fold), parameters)
    else:
        scores_test = None
        if l2r == "6" or l2r == "lm":
            scores_test = prepareDS_callL2R(l2r, train, test, fold, "NDCG", parameters)

        else:
            scores_test = executeSckitLearn(l2r, train, test, int(fold), parameters)

    if len(scores_test) < 3:
        X_train, y_train, query_id_train = load_L2R_file(testFile)
        queriesList = getQueries(query_id_train)
        temp = np.array([0.0] * len(queriesList))
        return temp

    pred, mapPred = modelEvaluation(test, scores_test, totalFeatures)

    pred = np.asarray(pred)

    # riskBaseline= readingFile("/home/daniel/Dropbox/WorkingFiles/BaselinesResults/"+coll+".MAX.NDCG@10.result.F"+fold)
    # winLoss = gettingWinsLosses (pred, riskBaseline)
    # trisk = modelRiskEvaluation(pred, riskBaseline, "trisk")
    # return np.average(pred), trisk, winLoss
    return pred


# output = check_output(['perl', '/home/daniel/Dropbox/WorkingFiles/perl_Scripts/gettingBaseLineManySeeds.pl', str(fold), str(nTrees), str(nExecs), coll, mask, str(l2r), "0", "NDCG@10" ])
# print output
# pred = []
# listLines= output.splitlines()
# for l in listLines:
#    m = re.search('=+>(.*)', l)
#    if m:
#        found = m.group(1)
#        pred.append(float(found))

def createNewDataset(outF, train):
    if (train.x).ndim > 1:
        with open(outF, 'w') as file:
            for i in range(0, len(train.y)):
                feature = ""
                idF = 1
                for fe in train.x[i]:
                    feature = feature + str(idF) + ":" + str(fe) + " "
                    idF = idF + 1

                file.write(str(int(train.y[i])) + " qid:" + str(int(train.q[i])) + " " + feature + "#\n")
    else:
        with open(outF, 'w') as file:
            for i in range(0, len(train.y)):
                file.write(str(int(train.y[i])) + " qid:" + str(int(train.q[i])) + " 1:" + str(train.x[i]) + " #\n")


def executeExternalLib(l2r, fTrain, fTest, fold, METRIC, parameters):
    out = "outPutL21Program.fold" + str(fold)
    out1 = "outPutL21Program1.fold" + str(fold)
    fScore = "scoreFile.Fold" + str(fold)
    fModel = "modelFile.Fold" + str(fold)
    if l2r == "6" or l2r == "lm":
        try:
            with open(out, 'w') as f:
                call(['/home/daniel/programs/quickrank/bin/quicklearn', '--algo', "LAMBDAMART", "--train", fTrain,
                      "--train-metric",
                      "NDCG", "--train-cutoff", "10", "--test", fTest, "--test-metric", METRIC.upper(), "--test-cutoff",
                      "10",
                      "--scores", fScore, "--num-trees", str(parameters[0]), "--shrinkage", "0.075", "--num-leaves",
                      str(parameters[1])], stdout=f)
        except:
            result = [0] * 10
            return result;
    if l2r in ["7", "ada", "8", "listnet"]:
        if l2r in ["7", "ada"]:
            l2r = "3"
        elif l2r in ["8", "listnet"]:
            l2r = "7"
        try:
            command = []
            with open(out, 'w') as f:
                call(["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar",
                      "-round", str(parameters[0]), "-tolerance", str(parameters[1]), "-train", fTrain, "-test", fTest,
                      "-ranker", l2r, "-save", fModel],
                     stdout=f)
            command.append(
                ["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar",
                 "-round", str(parameters[0]), "-tolerance", str(parameters[1]), "-train", fTrain, "-test", fTest,
                 "-ranker", l2r, "-save", fModel])
            # print (["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-round", str(10), "-train", trainFile , "-test", testFile , "-ranker", str(3), "-save", modelFile])
            with open(out1, 'w') as f:
                call(["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar",
                      "-load", fModel, "-rank", fTest, "-score", fScore], stdout=f)

            command.append(
                ["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar",
                 "-load", fModel, "-rank", fTest, "-score", fScore])
            call(['rm', fModel, out1])

            # print (["java", "-jar", "-Xms5002M", "-Xmx5024M", "/home/daniel/programs/RankLib-v2.1/bin/RankLib.jar", "-load",
        except IOError:
            print("We have gotten some error:")
            print(command[0])
            print(command[1])
            return np.array([0])

    if os.stat(fScore).st_size == 0:
        print("We have gotten some error:")
        print(command[0])
        print(command[1])
        return np.array([0])

    result = np.loadtxt(fScore)
    call(['rm', out, fScore])

    return result


def prepareDS_callL2R(l2r, train, test, fold, METRIC, parameters):
    outTrain = "outNewDatasetQRkFold.train" + str(fold)
    createNewDataset(outTrain, train)

    outTest = "outNewDatasetQRkFold.test" + str(fold)
    createNewDataset(outTest, test)

    del train.x
    del test.x
    result = executeExternalLib(l2r, outTrain, outTest, fold, METRIC, parameters)
    call(['rm', outTrain, outTest])
    return result
