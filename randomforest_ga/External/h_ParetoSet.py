# coding=utf-8
from __future__ import division
__author__ = 'daniel'

import sys, os
pathup = os.path.abspath(os.path.join(
    os.path.dirname(__file__+"/../"), os.path.pardir))
sys.path.insert(0, pathup)
#print(sys.path)
import numpy as np
import re

from operator import itemgetter

def getGAFile_and_CheckContentFile(dir, learner, coll, objective, fold, printing, checkName):
    fileGA = dir + "/logGA." + learner + "."+ coll + ".wilcoxon" + "." + objective.lower() +".Fold" + str(fold)
    if not (os.path.isfile(fileGA)):
        fileGA = dir + "/logGA." + learner + "."+ coll + ".Fold" + str(fold)
        if not (os.path.isfile(fileGA)):
            fileGA = dir + "/log." + learner + "." + objective+"."+coll + ".Fold" + str(fold)


    if printing == 0:
        return fileGA

    selection = ""
    if printing ==1:
        #print fileGA
        with open(fileGA, "r") as ins:
            for line in ins:
                if ">Criteria:" in line:
                    m = re.search('>Criteria:\s(.*)\n',line)
                    if m:
                        criteria = m.group(1)
                if ">DATASET:" in line:
                    m = re.search('>DATASET:\s(.*)\n', line)
                    if m:
                        dataset = m.group(1)

                if ">L2R_Method:" in line:
                    m = re.search('>L2R_Method:\s(.*)\n', line)
                    if m:
                        blackBox = m.group(1)

                if ">Selection:" in line:
                    m = re.search('>Selection:\s(.*)\n', line)
                    if m:
                        selection = m.group(1)
                if "inicializando..." in line:
                    break


        if dataset != coll:
            fileGA = "error"

        if blackBox == "4":
            blackBox = "Regressao"
        elif blackBox == "5":
            blackBox = "LongTree"
        elif blackBox == "10":
            blackBox = "ShortTree"
        elif blackBox == "1":
            blackBox = "RandomForest"
        print (objective, " <-> ", criteria , " | ", selection, "| ", blackBox)

        testName =  criteria + coll + blackBox + selection
        if fold != 0:
            if checkName != testName:
                print ("Error in some parameter inside of file. Before:", checkName, " now: ", testName)
                sys.exit(0)

        if fileGA == "error":
            print ("Error in some parameter")
            sys.exit(0)

    return fileGA, criteria, dataset, blackBox, selection

def getLastPop(logFile):

    start = 0
    lastPop= []

    lastText= ""
    with open(logFile, "r") as ins:
        for line in ins:
            if "Test Ranking[" in line:
                    lastText = line.strip()

    with open(logFile, "r") as ins:
        for line in ins:

            if lastText in line:
                start = 1
            if "TestRank:" in line and start == 1:

                if "TRisk" in line:
                    m = re.search('Fitness\s:\s(.*)\sParamC:\s(.*)\sResultValue\s\(avg\):\s(.*)\sTestResult\s:(.*)\sValiResult\s:(.*)\sTRisk\s:(.*)\sNumFeature:.*Features:\s(.*)', line)
                    if m:
                        fitness=m.group(1)
                        predTrain=m.group(3)
                        predTest=m.group(4)
                        predVali=m.group(5)
                        risk    =m.group(6)
                        genes   =m.group(7)
                elif "Risk" in line:
                    m = re.search('Fitness\s:\s(.*)\sParamC:\s(.*)\sResultValue\s\(avg\):\s(.*)\sTestResult\s:(.*)\sValiResult\s:(.*)\sRisk\s:(.*)\sNumFeature:.*Features:\s(.*)',line)
                    if m:
                        fitness = m.group(1)
                        predTrain = m.group(3)
                        predTest = m.group(4)
                        predVali = m.group(5)
                        risk = m.group(6)
                        genes = m.group(7)
                else:
                    m = re.search(
                        'Fitness\s:\s(.*)\sParamC:\s(.*)\sResultValue\s\(avg\):\s(.*)\sTestResult\s:(.*)\sValiResult\s:(.*)\sNumFeature:.*Features:\s(.*)',
                        line)
                    if m:
                        fitness = m.group(1)
                        predTrain = m.group(3)
                        predTest = m.group(4)
                        predVali = m.group(5)
                        genes = m.group(6)
                        risk = 0
                subList = []
                #0-fitness, 1-train, 2-test, 3-vali, 4-risk, 5-gene
                #print fitness, predTrain, predTest, predVali, genes.count("1")
                subList.append(float(fitness))
                subList.append(float(predTrain))
                subList.append(float(predTest))
                subList.append(float(predVali))
                subList.append(float(risk))
                subList.append(genes)
                lastPop.append(subList)
    return lastPop

def gettingFSBaselines(logFile, info):

    if "FSA" in info:
        if "web10k" in logFile:
            numFeatures = 136
        elif "yahoo" in logFile:
            numFeatures = 700
        elif "web30k" in logFile:
            numFeatures = 136

        vetor = np.array(["0"] * numFeatures)
        genes = ""
        with open(logFile, "r") as ins:
            for line in ins:
                if "SelectedFeatures" in line:
                    m = re.search('SelectedFeatures:\s*\[\[(.*)\]\]', line)
                    if m:
                        #print (m.group(1))
                        genes = m.group(1)
                        lGenes = genes.split(",")
                        #print (lGenes, numFeatures)
                        for f in lGenes:
                        #print ("G", f)
                            vetor[int(f)-1] = "1"
                        genes = ''.join(vetor)
        #print (genes)
    elif "BTFS" in info:

        with open(logFile, "r") as ins:
            for line in ins:
                if "Current Frame:" in line:
                    m = re.search('Current Frame: ([^\s]+) Size: ([^\s]+)\s',line)
                    if m:
                        genes = m.group(1)
                        features = m.group(2)

        genes=re.sub(';', '', genes)
    elif "DivFS"in info:

        if "web10k" in logFile:
            numFeatures = 60
        elif "yahoo" in logFile:
            numFeatures = 200
        elif "2003_td" in logFile:
            numFeatures = 27
        elif "2004_td" in logFile:
            numFeatures = 20

        with open(logFile, "r") as ins:
            for line in ins:
                if "[" in line:
                    scores = line.replace("[", "")
                    scores = scores.replace("]", "")
                    scores = np.fromstring(scores, dtype=float, sep=',')
                    sortFeatures = np.argsort(-scores)

                    genes = ["0"]* len(scores)

                    for i in range(numFeatures):
                        genes[sortFeatures[i]]="1"
                    genes = ''.join(genes)
                    break
    else:
        print ("Error. The FS baseline was not defined.")
        sys.exit(0)


    # 0-fitness, 1-train, 2-test, 3-vali, 4-risk, 5-gene
    temp = []
    temp.append(0)
    temp.append(0)
    temp.append(0)
    temp.append(0)
    temp.append(0)
    temp.append(genes)
    paretoSet = []
    paretoSet.append(temp)

    return paretoSet

def getWeakID (l2r):
    if l2r == "REGR" or l2r == "linearregression" or l2r == "linearRegression":
        return 4
    elif l2r == "longRegtree" or l2r == "REGTREE":
        return 5
    else:
        print ("Error to get the key", l2r)
        sys.exit(0)

def objectiveInitial(objective):

    if objective.lower() == "full":
        return "FULL"

    if objective.lower() == "corr":
        return "FSA"

    if objective.lower() == "georisk_features":
        return "G.F"

    if objective.lower() == "georisk_feature":
        return "G.F"
    if objective.lower() == "precision_feature":
        return "E.F"
    if objective.lower() == "prediction_feature_degradation":
        return "E.F.R"


    if objective.lower() == "precision_degradation":
        return "E.R"

    if objective.lower() == "precision_feature_georisk":
        return "E.F.G"
    if objective.lower() == "precision_feature_degradation":
        return "E.F.R"
    if objective.lower() == "prediction_feature_georisk":
        return "E.F.G"
    if objective.lower() == "preddegradation":
        return "E.R"
    if objective.lower() == "predgeorisk":
        return "E.G"
    if objective.lower() == "pred":
        return "E"
    if objective.lower() == "precision":
        return "E"
    if objective.lower() == "precision_georisk":
        return "E.G"
    if objective.lower() == "risk_feature":
        return "T.F"
    if objective.lower() == "risk":
        return "T"
    if objective.lower() == "georisk":
        return "G"

    if objective.lower() == "preddegradationtour":
        return "E.R"
    if objective.lower() == "predfeature":
        return "E.F"
    if objective.lower() == "predgeorisk":
        return "E.G"
    if objective.lower() == "predtour":
        return "E"
    if objective.lower() == "riskfeature":
        return "T.F"

    if "similarity" in objective.lower():
        return "DivFS"

    if "precfeature" in objective.lower():
        return "BTFS"

    print ("Problem to define the objective initials. Objective: ", objective)
    sys.exit(1)

def getParetoFrontierSet(logFile, spea2):

    lastPop = getLastPop(logFile)
    indexes = []
    # 0-fitness, 1-train, 2-test, 3-vali, 4-risk, 5-gene
    for i in range( len(lastPop)):
        #print lastPop[i]
        if lastPop[i][0]>=1 or lastPop[i][5].count('0') == 0: #To be in pareto frontier the indiviual must have fitness less than 1 and with some gene 0 on it.
            indexes.append(i)

    if spea2 == "spea2":
        if len(indexes) < len(lastPop):
            for index in sorted(indexes, reverse=True):
                del lastPop[index]
            lastPop = sorted(lastPop, key=itemgetter(1), reverse=True)
            return lastPop
        else:
            ind = -1
            while (lastPop[ind][5].count('0')==0):
                ind = ind -1 ## talking back
            temp = []
            temp.append(lastPop[ind])
            return temp

    elif spea2 != "notspea2":
        print ("Error to get maxValeu", spea2)
        sys.exit(0)

    lastPop = sorted(lastPop, key=itemgetter(1), reverse=False)


    return lastPop

def printPareto(paretoSet, fold):

    print ("PS F"+str(fold)+": size(" +str(len(paretoSet))+") FIT 1st [" + "{0:.4f}".format(paretoSet[0][0]) + "] 2sd [" + "{0:.4f}".format(paretoSet[-1][0]) + "] TRAIN 1st [" + "{0:.4f}".format(paretoSet[0][1]) + "] 2sd [" + "{0:.4f}".format(paretoSet[-1][1])+"]")


def parsingGA_ObjectivesDirectory(directory):
    m = re.search('GA\.(\w*)\.(\w*)\.(\w*)\.*(.*)$', directory)
    if m:
        learner = m.group(1)
        objective = m.group(2)
        coll = m.group(3)
        type = m.group(4)

    return learner, type, coll, objective

def parsingDirectory(directory):
    m = re.search('(.*)\.(.*)\.(.*)\.(.*)', directory)
    if m:
        learner = m.group(1)
        type = m.group(2)
        coll = m.group(3)
        spea2 = m.group(4)

    return learner, type, coll, spea2

def getBestNDCG(dicPrediction, paretoSet, decisionMaker, spea2):
    # 0-fitness, 1-train, 2-test, 3-vali, 4-risk, 5-gene
    if decisionMaker == "bestFitness":
        idDM = 0
    elif decisionMaker =="bestTrain":
        idDM = 1
    elif decisionMaker == "bestTest":
        idDM = 2
    elif decisionMaker == "mean":
        idDM = 1
    else:
        print ("There is not a decision maker set", decisionMaker)
        sys.error(0)

    #MAX is TRUE for all selection, except to spea2 and fitness
    maxValue = True
    if spea2 == "spea2" and idDM == 0:
        maxValue = False

    if decisionMaker == "bestTest": #### Filling the TestNDCG first
        for i in range(len(paretoSet)):
            test = dicPrediction[paretoSet[i][5]]
            paretoSet[i][2] = np.mean(test)
            if i > 10:
                break

    if decisionMaker == "mean":
        soma=0
        i=0
        for individual in paretoSet:
            if individual[5] not in dicPrediction:
                print ("Did not find gene", individual[5])
                sys.exit(0)
            else:
                soma=soma + dicPrediction[individual[5]]
            if i > 10:
                break
            i=i+1
        return soma/len(paretoSet), individual[5]

    value = paretoSet[0][idDM]
    gene = paretoSet[0][5]
    for individual in paretoSet:
        if maxValue == True:
            # 0fitness 1preTrain 2predTest 3predVali 4risk 5gene
            if value < individual[idDM]:
                value = individual[idDM]
                gene  = individual[5]
        else:
            if value > individual[idDM]:
                value = individual[idDM]
                gene = individual[5]

    if gene not in dicPrediction:
        print ("Did not find gene", gene)
        sys.exit(0)
    else:
        return dicPrediction[gene], gene


def getTimeDetails(logFile):

    #Temp indv: 181 Servidor: muck.lbd.dcc.ufmg.brMethodCompar:wilcoxonresultado met.Ap.: 0.381836866667 testR:-1.0 valiR:NaN numFeatures: 136Chomossome: 1000010011101000010100001000010001010111111001001111110111110011001000100100010111001000001011001001011010011001000101001111000000111100
    erro = 0
    totalTime = 0
    l2rProcTime = 0
    notExecuted = 0
    dicHosts= {}
    with open(logFile, "r") as ins:
        for line in ins:

            if "ERRO" in line or "Porblem" in line or "ERRO" in line or "Error" in line:
                erro = 1

            if "NoExecuted:" in line:
                notExecuted=notExecuted+1
            if "Tempo total da gera" in line:
                m = re.search('.*:\s(.*)\s', line)
                if m:
                    aux = int(m.group(1))
                    l2rProcTime = l2rProcTime + aux

            if "Total time" in line :
                m = re.search('Total time:\s(.*)', line)
                if m:
                    totalTime=int(m.group(1))


            if "Temp indv:" in line :
                m = re.search('Temp indv: (.*)\sServidor:\s(.*).lbd.dcc.ufmg.br', line)
                if m:
                    time = int(m.group(1))
                    host = m.group(2)

                    if host in dicHosts:
                        dicHosts[host][0]=dicHosts[host][0]+1
                        dicHosts[host][1]=dicHosts[host][1] + time
                    else:
                        dicHosts[host]=[1,time]



    return erro, totalTime/1000,l2rProcTime, notExecuted, dicHosts

def obtainRealMask(gmask, dataset):

    listMask = [y for y in str(gmask)]
    finalMask = ''

    if "web10k" in dataset:
        #0#1-5 covered query term number
        #1#6-10 covered query term ratio
        #2#11-15 stream length - doc
        #3#16-20 IDF
        #4#21-25 sum of term frequency
        #5#26-30 min of term frequency
        #6#31-35 max of term frequency
        #7#36-40 mean of term frequency
        #8#41-45 variance of term frequency
        #9#46-50 sum of stream length normalized term frequency
        #10#51-55 min of stream length normalized term frequency
        #11#56-60 max of stream length normalized term frequency
        #12#61-65 mean of stream length normalized term frequency
        #13#66-70 variance of stream length normalized term frequency
        #14#71-75 sum of TF*IDF
        #15#76-80 min of TF*IDF
        #16#81-85 max of TF*IDF
        #17#86-90 mean of TF*IDF
        #18#91-95 variance of TF*IDF
        #19#96-100 boolean model
        #20#101-105 vector space model
        #21#106-110 BM25
        #22#111-115 LMIR.ABS Language Model approach for information retrieval (IR) with absolute discouting smoothing
        #23#116-120 LIMIR.DIR Language Model approach for information retrieval (IR) with bayesian smoothing using Dirichlet priors
        #24#121-125 LIMIR.JM  Language Model approach for information retrieval (IR) with Jelinek-Mercer smoothing

        #25#126-127 Number of slash in URL and length of URL
        #26#128-131 Inlink, OutLink, pageRank, SiteRank
        #27#132-133 QualityScore and QualityScore2
        #28#134-136 Query-url click count, url click count , url dwell time

        for i in range(25):
            if listMask[i]=="1":
                finalMask=finalMask+"11111"
            else:
                finalMask = finalMask + "00000"

        if listMask[25] =="1":#Number of slash in URL and length of URL
            finalMask = finalMask + "11"
        else:
            finalMask = finalMask + "00"

        if listMask[26] == "1":  # Inlink, OutLink, pageRank, SiteRank
            finalMask = finalMask + "1111"
        else:
            finalMask = finalMask + "0000"

        if listMask[27] == "1":  # QualityScore and QualityScore2
            finalMask = finalMask + "11"
        else:
            finalMask = finalMask + "00"

        if listMask[28] == "1":  # Query-url click count, url click count , url dwell time
            finalMask = finalMask + "111"
        else:
            finalMask = finalMask + "000"

        return finalMask

    if "2003" in dataset or "2004" in dataset:
        #0#1-5 covered query term number
        #1#6-10 IDF
        #2#11-15 Count * IDF
        #3#16-20 stream length
        #4#21-25 BM25
        #5#26-30 LIMIR.ABS
        #6#31-35 LIMIR.DIR
        #7#36-40 LIMIR.JM

        #8#41-42 Sitemap
        #9#43-48 Hyperlink
        #10#49-50 HITS
        #11#51-52 PageRank HostRank
        #12#53-55 Topical
        #13#56-57 Inlink OutLink
        #14#58-60 URL
        #15#61-64  extracted title

        for i in range(8):
            if listMask[i] == "1":
                finalMask = finalMask + "11111"
            else:
                finalMask = finalMask + "00000"

        if listMask[8] == "1":  # Query-url click count, url click count , url dwell time
            finalMask = finalMask + "11"
        else:
            finalMask = finalMask + "00"

        if listMask[9] == "1":  # Query-url click count, url click count , url dwell time
            finalMask = finalMask + "111111"
        else:
            finalMask = finalMask + "000000"

        if listMask[10] == "1":  # Query-url click count, url click count , url dwell time
            finalMask = finalMask + "11"
        else:
            finalMask = finalMask + "00"

        if listMask[11] == "1":  # Query-url click count, url click count , url dwell time
            finalMask = finalMask + "11"
        else:
            finalMask = finalMask + "00"

        if listMask[12] == "1":  # Query-url click count, url click count , url dwell time
            finalMask = finalMask + "111"
        else:
            finalMask = finalMask + "000"

        if listMask[13] == "1":  # Query-url click count, url click count , url dwell time
            finalMask = finalMask + "11"
        else:
            finalMask = finalMask + "00"

        if listMask[14] == "1":  # Query-url click count, url click count , url dwell time
            finalMask = finalMask + "111"
        else:
            finalMask = finalMask + "000"

        if listMask[15] == "1":  # Query-url click count, url click count , url dwell time
            finalMask = finalMask + "1111"
        else:
            finalMask = finalMask + "0000"

        return finalMask
    if "yahoo" in dataset:
        print("ERRO, yahoo is not adaptable for this experiment.")
        sys.exit(0)

def getTrainOverGenerations(logFile):

    generation=-1
    listGenrations=[]
    with open(logFile, "r") as ins:
        for line in ins:

            if "===Test Ranking[" in line or "===Ranking[" in line:

                generation = generation +1
                listGenrations.append([])

            if "TestRank:" in line or "Rank:" in line:

                if "TRisk" in line:
                    m = re.search('Fitness\s:\s(.*)\sParamC:\s(.*)\sResultValue\s\(avg\):\s(.*)\sTestResult\s:(.*)\sValiResult\s:(.*)\sTRisk\s:(.*)\sNumFeature:.*Features:\s(.*)', line)
                    if m:
                        fitness=m.group(1)
                        predTrain=m.group(3)
                        predTest=m.group(4)
                        predVali=m.group(5)
                        risk    =m.group(6)
                        genes   =m.group(7)
                elif "Risk" in line:
                    m = re.search('Fitness\s:\s(.*)\sParamC:\s(.*)\sResultValue\s\(avg\):\s(.*)\sTestResult\s:(.*)\sValiResult\s:(.*)\sRisk\s:(.*)\sNumFeature:.*Features:\s(.*)',line)
                    if m:
                        fitness = m.group(1)
                        predTrain = m.group(3)
                        predTest = m.group(4)
                        predVali = m.group(5)
                        risk = m.group(6)
                        genes = m.group(7)
                else:
                    m = re.search(
                        'Fitness\s:\s(.*)\sParamC:\s(.*)\sResultValue\s\(avg\):\s(.*)\sTestResult\s:(.*)\sValiResult\s:(.*)\sNumFeature:.*Features:\s(.*)',
                        line)
                    if m:
                        fitness = m.group(1)
                        predTrain = m.group(3)
                        predTest = m.group(4)
                        predVali = m.group(5)
                        genes = m.group(6)
                        risk = 0
                if float(predTrain) != -1:
                    listGenrations[generation].append([float(predTrain), genes])

    return listGenrations
