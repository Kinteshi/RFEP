from __future__ import division

import sys
import os.path
from subprocess import check_output
from collections import defaultdict
#import rpy2.robjects as robjects
import numpy as np
import re
from subprocess import call

from scipy import stats

from External import *

class basicStructure:
    def __init__(self):
        self.marginal = None
        self.mat = None
        self.pvalue = None
        self.variance = None
        self.greaterIsBetter = None


class dataset:
    def __init__(self):
        self.q = None
        self.x = None
        self.y = None

def printParetoFrontier(vet, prediction, similarity,  obj1, obj2, fileName):

    vetOrdinal = np.array(range(0, len(vet)))
    mat = (np.vstack((np.reshape(vet, -1), np.reshape(vetOrdinal, -1)))).T
    mat = mat[np.argsort(mat[:, 0], kind="mergesort")]

    with open(fileName, 'w') as file:
        file.write("MaxMin"+obj1+ str(max(prediction)) + " " + str(min(prediction)) + "\n")
        file.write("MaxMin"+obj2+ str(max(similarity)) + " " + str(min(similarity)) + "\n")
        file.write("id " + obj1 + " "+ obj2+"\n")
        id = 0
        while mat[id][0] < 1:
            file.write( str(int(mat[id,1])) + " " + str(prediction[mat[id,1]]) + " " + str( similarity[mat[id,1]]) + "\n")
            id = id + 1


def obtainDominace(obj1, obj2, obj3, prediction, similarity, risk, variance, vetFeatures, nFeatures):

    dicX_dominate_YList = defaultdict(list)
    dicX_dominatedBy_YList = defaultdict(list)

    for fx in  vetFeatures:
        for fy in vetFeatures:
            # if x dominates j

            if obj3 == "null":
                if dominate2(fx, fy, obj1, obj2, prediction, similarity, risk, variance) == True:
                    dicX_dominate_YList[fx].append(fy)
                else:
                    # if x is dominated by y
                    if dominate2(fy, fx, obj1, obj2, prediction, similarity, risk, variance) == True:
                        dicX_dominatedBy_YList[fx].append(fy)

            else:
                if dominate3(fx, fy, obj1, obj2, obj3, prediction, similarity, risk, variance) == True:
                    dicX_dominate_YList[fx].append(fy)
                else:
                    # if x is dominated by y
                    if dominate3(fy, fx, obj1, obj2, obj3, prediction, similarity, risk, variance) == True:
                        dicX_dominatedBy_YList[fx].append(fy)

    # a fitness primeiramente eh, para cada x  resgata a lista "L" de individuos que domina x  e faz o somatorio de quanto esses L individuos dominam
    vetDominance = np.array([0] * nFeatures, dtype=float)
    for k in dicX_dominatedBy_YList:
        v = dicX_dominatedBy_YList[k]
        r = 0
        for iDominator in v:
            # soma quantos este ndividuo iDominator domina
            r += len(dicX_dominate_YList[iDominator])
        vetDominance[k] = r

    #for k, v in dicX_dominatedBy_YList.iteritems():
    #    print k, v

    #print vetDominance
    #for i in vetFeatures:
    #    if vetDominance[i] == 0:
    #        if i not in dicX_dominate_YList:
    #            #print "Feature ", i
    #            vetDominance[i]=1000

    return vetDominance



def gettingVarianceData(obj1, obj2, obj3, extraInfo, prediction, coll, fold, nFeatures, pvalueST):
    variance = basicStructure()
    if obj1 == "varinter" or obj2 == "varinter" or obj3 == "varinter":
        variance.marginal = np.array([0.0] * nFeatures)
        for f in range(0, nFeatures):
            variance.marginal[f] = np.var(prediction.mat[:, f])
        variance.mat = prediction.mat
        variance.pvalue = pvalueST
        variance.greaterIsBetter=True
        variance.variance = True

    if obj1 == "varintra" or obj2 == "varintra" or obj3 == "varintra":
        varTrain = dataset()
        trainFile = "../Colecoes/" + coll + "/Fold" + str(fold) + "/Norm.train.txt"
        mask = "1" * nFeatures
        varTrain.x, varTrain.y, varTrain.q = load_L2R_file(trainFile, mask)
        # varTrain.x = init
        # varTrain.q = [1,1,1,2,2,3,3,3]
        queriesList = getQueries(varTrain.q)

        variance.marginal = np.array([0.0] * nFeatures)
        variance.mat = np.zeros(shape=(len(queriesList), nFeatures))
        for f in range(0, nFeatures):
            idQ = 0
            for docs in queriesList:
                variance.mat[idQ, f] = np.var(varTrain.x[docs, f])
                idQ = idQ + 1
        if extraInfo == "var":
            for f in range(0, nFeatures):
                variance.marginal[f] = np.var(variance.mat[:, f])
        else:
            for f in range(0, nFeatures):
                variance.marginal[f] = np.mean(variance.mat[:, f])
        variance.pvalue = pvalueST
        variance.greaterIsBetter=True
        variance.variance = False

    return variance

def gettingRiskData (obj1, obj2, obj3, extraInfo, prediction, alpha, nFeatures, pvalueST):
    risk = basicStructure()
    if obj1 == "risk" or obj2 == "risk" or obj3 == "risk":
        if "max" == extraInfo:
            baseline = getMaxRiskBaseline(prediction.mat)
        else:
            baseline = getMeanRiskBaseline(prediction.mat)

        risk.mat = np.zeros(shape=(len(baseline), nFeatures))
        for i in range(nFeatures):
            risk.mat[:, i] = getRisk(prediction.mat[:, i], baseline)

        risk.marginal = np.array([0.0] * nFeatures)
        for i in range(nFeatures):
            risk.marginal[i] = np.mean(risk.mat[:, i])
        risk.pvalue = pvalueST
        risk.greaterIsBetter=False

    if obj1 == "trisk" or obj2 == "trisk" or obj3 == "trisk":
        if "mean" == extraInfo:
            baseline = getMeanRiskBaseline(prediction.mat)
        else:
            baseline = getMaxRiskBaseline(prediction.mat)

        risk.mat = np.zeros(shape=(len(baseline), nFeatures))
        for i in range(nFeatures):
            risk.mat[:, i] = getRisk(prediction.mat[:, i], baseline)

        risk.marginal = np.array([0.0] * nFeatures)
        for i in range(nFeatures):
            risk.marginal[i] = getTRisk(prediction.mat[:, i], baseline, alpha)
        risk.pvalue = pvalueST
        risk.greaterIsBetter=True

    if obj1 == "georisk" or obj2 == "georisk" or obj3 == "georisk":
        risk.marginal = getGeoRisk(prediction.mat, alpha)
        risk.mat = np.array([risk.marginal, risk.marginal]) ### This line allows get into the fuctino of tied.
        risk.pvalue = 0
        risk.greaterIsBetter=True

    risk.variance=False


    return risk


def getOutPuFile (algorithm, dataset, percFeat, pvalue, obj1, obj2, obj3, fold, predFile, corrFile , info):


    matchObj = re.search(r'_ml(.+)_', predFile, re.M | re.I)
    mlMetric = matchObj.group(1)

    if mlMetric == "6":
        mlMetric ="lm"
    if mlMetric == "1":
        mlMetric ="rf"
    if mlMetric == "4":
        mlMetric ="rl"

    #print "File", corrFile
    matchObj = re.search(r'_mat(.+)C_', corrFile, re.M | re.I)
    corrMetric = matchObj.group(1)
    #print "corrMetric", corrMetric
    if corrMetric == "K" or corrMetric == "k":
        corrMetric = "kendau"
    if corrMetric ==  "P" or corrMetric == "p":
        corrMetric = "pearson"

    if info != "":
        info = "."+info

    return algorithm+"/" + str(percFeat) + "/" + corrMetric + "/" + mlMetric + "/" + "result."+dataset+".pvalue" + str(pvalue) + ".obj1" + obj1 + ".obj2" + obj2 + ".obj3" + obj3 + info+".F" + str(fold)


def readingFile(file):

    vetPrec=[]
    with open(file, "r") as inFile:
        for line in inFile:
            if "mean" in line:
                m = re.search(r"mean=+>(.*)\s", line)
                vetPrec.append(m.group(1))
    return np.asarray(vetPrec)

def checkTied(x_vet, y_vet, statisticalTest, variance):

    tied=np.array_equal(x_vet, y_vet)
    if tied==True:
        return True

    if statisticalTest == 0:
        return False

    if variance:
        rd1 = (robjects.FloatVector(x_vet))
        rd2 = (robjects.FloatVector(y_vet))
        rvtest = robjects.r['var.test']
        pvalue = rvtest(rd1, rd2, paired=True)[2][0]


    else:
        #res, pvalue = wilcoxon(x_vet, y_vet, zero_method="pratt" )

        rd1 = (robjects.FloatVector(x_vet))
        rd2 = (robjects.FloatVector(y_vet))
        rvtest = robjects.r['wilcox.test']
        pvalue = rvtest(rd1, rd2, paired=True)[2][0]


#    print "StatisticalTest", str(statisticalTest)

    if statisticalTest <= pvalue:
        return True
    else:
        return False


def buildMatrixNaiveCorrelation (x_train, ql_train, metric="pearson"):

    #print "******************NAIVE", metric
    queriesList = getQueries(ql_train)
    nFeatures = x_train.shape[1]
    nQueries = len(queriesList)
    mat = np.zeros(shape=(nFeatures, nFeatures))


    for q in range(0, nQueries):
        qIdx = np.asarray(queriesList[q])

        auxMat = np.zeros(shape=(nFeatures, nFeatures))
        for fin in range (0, nFeatures):
            for fout in range(0, nFeatures):
                value = None
                if np.array_equal(x_train[qIdx[:, None], fout], x_train[qIdx[:, None], fin]) :
                    value = 0
                else:
                    if (metric == "kendalltau"):
                        value, p_value = stats.kendalltau(x_train[qIdx[:, None], fout],  x_train[qIdx[:, None], fin])
                    else:
                        value, p_value = stats.pearsonr(x_train[qIdx[:, None], fout],  x_train[qIdx[:, None], fin])

                if (np.isnan(value)):
                    value = 0
                    #print "Value", value
                    #print (x_train[qIdx[:, None], fout], "\n")
                    #print (x_train[qIdx[:, None], fin], "\n")
                    #print ("1st",metric, np.isfinite(x_train[qIdx[:, None], fout]).all(), "\n")
                    #print ("2st",metric, np.isfinite(x_train[qIdx[:, None], fin]).all(), "\n")
                auxMat[fout][fin] = value
        mat=mat+auxMat

    mat=mat/nQueries
    return mat

def buildMatrixPredCorrelation (matPredFeature,  metric="pearson"):
   # print "******************PREDICTION", metric
    nFeatures = matPredFeature.shape[1]
    mat = np.zeros(shape=(nFeatures, nFeatures))

    for fin in range(0, nFeatures):
        for fout in range(0, nFeatures):
            value = None
            if np.array_equal(matPredFeature[:, fout], matPredFeature[:, fin]):
                value = 0
            else:
                if (metric == "kendalltau"):
                    value, p_value = stats.kendalltau(matPredFeature[:, fout], matPredFeature[:, fin])
                else:
                    value, p_value = stats.pearsonr(matPredFeature[:, fout], matPredFeature[:, fin])

            if (np.isnan(value)):
                value = 0
                #print "Value", value
                #print (matPredFeature[:, fin], "\n")
                #print (matPredFeature[:, fout], "\n")
                #print ("1st", metric, np.isfinite(matPredFeature[:, fout]).all(), "\n")
                #print ("2st", metric, np.isfinite(matPredFeature[:, fin]).all(), "\n")

            mat[fout][fin] = value

    return mat

def getXvsY_OBJs_1_2_3(x, y, obj1, obj2, obj3 ):
    ##### OBJ1
    X_IsBetterY_obj1 = False
    if obj1.greaterIsBetter:
        if obj1.marginal[x] > obj1.marginal[y]:
            X_IsBetterY_obj1 = True
    else:
        if obj1.marginal[x] < obj1.marginal[y]:
            X_IsBetterY_obj1 = True

    X_IsTiedY_obj1 = checkTied(obj1.mat[:, x], obj1.mat[:, y], obj1.pvalue, obj1.variance)

    ##### OBJ2
    X_IsBetterY_obj2 = False
    if obj2.greaterIsBetter:
        if obj2.marginal[x] > obj2.marginal[y]:
            X_IsBetterY_obj2 = True
    else:
        if obj2.marginal[x] < obj2.marginal[y]:
            X_IsBetterY_obj2 = True

    X_IsTiedY_obj2 = checkTied(obj2.mat[:, x], obj2.mat[:, y], obj2.pvalue, obj2.variance)

    ##### OBJ3
    X_IsBetterY_obj3 = False
    if obj3.greaterIsBetter:
        if obj3.marginal[x] > obj3.marginal[y]:
            X_IsBetterY_obj3 = True
    else:
        if obj3.marginal[x] < obj3.marginal[y]:
            X_IsBetterY_obj3 = True

    X_IsTiedY_obj3 = checkTied(obj3.mat[:, x], obj3.mat[:, y], obj3.pvalue, obj3.variance)

    return X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2, X_IsBetterY_obj3, X_IsTiedY_obj3

def getXvsY_OBJs_1_2(x, y, obj1, obj2 ):
    ##### OBJ1 Prediction
    X_IsBetterY_obj1 = False

    #print "x", obj1.marginal[x] , "y", obj1.marginal[y]
    if obj1.greaterIsBetter:
        if obj1.marginal[x] > obj1.marginal[y]:
            X_IsBetterY_obj1 = True
    else:
        if obj1.marginal[x] < obj1.marginal[y]:
            X_IsBetterY_obj1 = True

    X_IsTiedY_obj1 = False
    # X_IsTiedY_obj1 = checkTied(obj1.mat[:, x], obj1.mat[:, y], obj1.pvalue, obj1.variance)

    ##### OBJ2 Risk or Variance or Similarity
    X_IsBetterY_obj2 = False
    #print "x", obj2.marginal[x], "y", obj2.marginal[y]
    if obj2.greaterIsBetter:
        if obj2.marginal[x] > obj2.marginal[y]:
            X_IsBetterY_obj2 = True
    else:
        if obj2.marginal[x] < obj2.marginal[y]:
            X_IsBetterY_obj2 = True

    X_IsTiedY_obj2 = False
    # X_IsTiedY_obj2 = checkTied(obj2.mat[:, x], obj2.mat[:, y], obj2.pvalue, obj2.variance)

    return X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2

def dominate3(x, y, obj1, obj2, obj3, prediction, similarity, risk, variance):

    if x == y:
        return False

    ########
    #Defining values of :
    # X_IsBetterY_obj1
    # X_IsBetterY_obj2
    # X_IsTiedY_obj1
    # X_IsTiedY_obj2


    ########## PREDICTION X SIMILARITY X RISK
    if (obj1 == "prediction" and obj2=="similarity" and (obj3 == "risk" or obj3 == "trisk" or obj3 == "georisk")):
        X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2, X_IsBetterY_obj3, X_IsTiedY_obj3 = getXvsY_OBJs_1_2_3(x, y, prediction, similarity, risk)  # obj1GreaterIsBetter, obj2GreaterIsBetter, variance

    ########## PREDICTION X SIMILARITY X VARIANCE
    elif (obj1 == "prediction" and obj2 == "similarity" and (obj3 =="varintra" or obj3=="varinter")):
        X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2, X_IsBetterY_obj3, X_IsTiedY_obj3 = getXvsY_OBJs_1_2_3(x, y, prediction, similarity, variance)  # obj1GreaterIsBetter, obj2GreaterIsBetter, variance

    ########## RISK X SIMILARITY X VARIANCE
    elif ((obj1 == "risk" or obj1 == "trisk" or obj1 == "georisk") and obj2=="similarity" and (obj3 == "varintra" or obj3 == "varinter")):
        X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2, X_IsBetterY_obj3, X_IsTiedY_obj3 = getXvsY_OBJs_1_2_3(x, y, risk, similarity, variance)  # obj1GreaterIsBetter, obj2GreaterIsBetter, variance

    ########## PREDICTION X RISK X VARIANCE
    elif (obj1 == "prediction" and (obj2 == "risk" or obj2 == "trisk" or obj2 == "georisk") and (obj3 == "varintra" or obj3 == "varinter")):
        X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2, X_IsBetterY_obj3, X_IsTiedY_obj3 = getXvsY_OBJs_1_2_3(x, y, prediction, risk, variance )  # obj1GreaterIsBetter, obj2GreaterIsBetter, variance

    #   Obj2                               Obj1
    # idv1 > idv2   and alpha<0.5           -                       => Does Not Dominate
    # idv1 == idv2  and alpha>0.5       idv1 > idv2 and alpha<0.5   => Dominates
    # idv1 < idv2   and alpha<5         idv1 == idv2  and alpha>0.5 => Dominates

    RESP = False
    if (X_IsBetterY_obj1 and not X_IsTiedY_obj1) and (X_IsBetterY_obj2 or X_IsTiedY_obj2) and (X_IsBetterY_obj3 or X_IsTiedY_obj3):
        RESP = True
    if (X_IsBetterY_obj2 and not X_IsTiedY_obj2) and (X_IsBetterY_obj1 or X_IsTiedY_obj1) and (X_IsBetterY_obj3 or X_IsTiedY_obj3):
        RESP = True
    if (X_IsBetterY_obj3 and not X_IsTiedY_obj3) and (X_IsBetterY_obj2 or X_IsTiedY_obj2) and (X_IsBetterY_obj1 or X_IsTiedY_obj1):
        RESP = True


    return RESP

def dominate2(x, y, obj1, obj2, prediction, similarity, risk, variance):

    if x == y:
        return False

    ########
    #Defining values of :
    # X_IsBetterY_obj1
    # X_IsBetterY_obj2
    # X_IsBetterY_obj3
    # X_IsTiedY_obj1
    # X_IsTiedY_obj2
    # X_IsTiedY_obj3

    if obj2 == "none":
        if obj1 == "prediction":
            if prediction.marginal[x] > prediction.marginal[y]:
                return True

        if obj1 == "similarity":
            if similarity.marginal[x] > similarity.marginal[y]:
                return True

        if obj1 == "risk":
            if risk.marginal[x] < risk.marginal[y]:
                return True


    ########## PREDICTION X RISK
    elif (obj1 == "prediction" and (obj2 == "risk" or obj2 == "trisk" or obj2 == "georisk")):
        X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2 = getXvsY_OBJs_1_2(x, y, prediction,risk)  # obj1GreaterIsBetter, obj2GreaterIsBetter, variance

    ########## PREDICTION X SIMILARITY
    elif (obj1 == "prediction" and obj2 == "similarity"):
        X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2 = getXvsY_OBJs_1_2(x, y, prediction,similarity)  # obj1GreaterIsBetter, obj2GreaterIsBetter, variance

    ########## PREDICTION X VARIANCE
    elif (obj1 == "prediction" and (obj2 == "varintra" or obj2 == "varinter")):
        X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2 = getXvsY_OBJs_1_2(x, y, prediction,variance)  # obj1GreaterIsBetter, obj2GreaterIsBetter, variance

    ########## RISK X SIMILARITY
    elif ((obj1 == "risk" or obj1 == "trisk" or obj1 == "georisk") and obj2 == "similarity"):
        X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2 = getXvsY_OBJs_1_2(x, y, risk,similarity)  # obj1GreaterIsBetter, obj2GreaterIsBetter, variance

    ########## RISK X ( VARINTRA or VARINTER)
    elif ((obj1 == "risk" or obj1 == "trisk" or obj1 == "georisk") and (obj2 == "varintra" or obj2 == "varinter")):
        X_IsBetterY_obj1, X_IsTiedY_obj1, X_IsBetterY_obj2, X_IsTiedY_obj2 = getXvsY_OBJs_1_2(x, y, risk,variance)  # obj1GreaterIsBetter, obj2GreaterIsBetter, variance


    #   Obj2                               Obj1
    # idv1 > idv2   and alpha<0.5           -                       => Does Not Dominate
    # idv1 == idv2  and alpha>0.5       idv1 > idv2 and alpha<0.5   => Dominates
    # idv1 < idv2   and alpha<5         idv1 == idv2  and alpha>0.5 => Dominates

    RESP=False
    if (X_IsBetterY_obj1 and (not X_IsTiedY_obj1 )) and (X_IsBetterY_obj2 or X_IsTiedY_obj2):
        RESP= True
    if (X_IsBetterY_obj2 and (not X_IsTiedY_obj2 )) and (X_IsBetterY_obj1 or X_IsTiedY_obj1):
        RESP= True



    if (X_IsTiedY_obj1 and X_IsTiedY_obj1 and X_IsBetterY_obj2 and (not X_IsTiedY_obj2)):
        if RESP == False:
            print( "ERROR IN DOMINACE DEFINITION - 1")


    if (X_IsTiedY_obj1 and (not X_IsTiedY_obj1) and X_IsBetterY_obj2 and X_IsTiedY_obj2):
        if RESP == False:
            print( "ERROR IN DOMINACE DEFINITION - 2")

    if (X_IsTiedY_obj1 and (not X_IsTiedY_obj1) and X_IsBetterY_obj2 and (not  X_IsTiedY_obj2)):
        if RESP == False:
            print( "ERROR IN DOMINACE DEFINITION - 3")

    if (X_IsTiedY_obj1 and (not X_IsTiedY_obj1) and (not X_IsBetterY_obj2) and X_IsTiedY_obj2):
        if RESP == False:
            print( "ERROR IN DOMINACE DEFINITION - 4")

    if ( (not X_IsTiedY_obj1) and  X_IsTiedY_obj1 and  X_IsBetterY_obj2 and (not X_IsTiedY_obj2)):
        if RESP == False:
            print( "ERROR IN DOMINACE DEFINITION - 5")

    return RESP


#                        if(vObj2Idv1 > vObj2Idv2 && alphaObj2 <= Config.better_when_p_less_than){ //INDV1 is Worst than IDV2 for Objective 1
#                                //System.out.println(":1****1 worst than 2");
#                                return false;
#                        }else if(vObj2Idv1 == vObj2Idv2 || alphaObj2 > Config.better_when_p_less_than ){//INDV1 is Tied with IDV2 for Objective 1
#                                if(ID1BetterID2_Obj1 && alphaObj1 <= Config.better_when_p_less_than){
#                                        //System.out.println(":2****1 better than 2");
#                                        return  true;
#                                }else{
#                                        //System.out.println(":3****1 worst than 2");
#                                        return false;
#                                }
#                        }else if(vObj2Idv1 < vObj2Idv2 && alphaObj2 <= Config.better_when_p_less_than ){ //INDV1 is Better than IDV2 for Objective 1
#                                if(ID1BetterID2_Obj1 || alphaObj1 > Config.better_when_p_less_than){
#                                        //System.out.println(":4****1 better than 2");
#                                        return true;
#                                }else{
#                                        //System.out.println(":5****1 worst than 2");
#                                        return  false;
#                                }
#                        }




def getEvaluationOverfitting(scoreList, featureFile, metric, append):

    searchObj = re.match(r".*@.*", metric, re.M | re.I)
    indexToGet = 1
    if searchObj:
        searchObj = re.search(r"(.*)@(.*)", metric, re.M | re.I)
        metric = searchObj.group(1)
        indexToGet = int(searchObj.group(2))
    scoreFile = "L2R.scoreMF"

    # print "Metric", metric, "index", indexToGet
    with open(scoreFile, "w") as outs:
        for s in scoreList:
            outs.write(str(float(s)) + "\n")
    call(["perl", "/home/daniel/programs/eval_score_LBD.pl", featureFile, scoreFile, "L2R.mapMF", "1"])

    #print "perl" + " /home/daniel/programs/eval_score_LBD.pl " + featureFile + " " + scoreFile + " L2R.mapMF 1"
    startFlag = 0
    iAP = 0
    MAP = 0
    with open("L2R.mapMF", "r") as ins:
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
                #print "teste: " +lineList[indexToGet].rstrip()
                MAP = MAP + float(lineList[indexToGet].rstrip())
                iAP = iAP + 1

    return (MAP / iAP)
def creatingNewFile(data, label, query):

    fileName ="training.txt"

    idx=0
    with open(fileName, "w") as outs:
        for s in data:

            line= str(int(label[idx]))+" qid:"+str(int(query[idx]))
            idx+=1
            idF=1
            for f in s:
                line = line +" " + str(idF)+":"+str(f)
                idF=idF+1

            line = line + " #docid\n"
            outs.write(line)

    return fileName



