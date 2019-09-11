import errno
import numpy
import os
import csv, sys

import pickle

import time

from External.ga import GeneticAlgorithm, Individual, ChangeName, Arquive, imprimir_individuo, nFeaturesColection, \
    geradorRelatorioValidacao, geradorRelatorioFinal
from External.h_functionsFilter import basicStructure, obtainDominace
from External.h_l2rMeasures import modelEvaluation, getTRisk
from External.h_l2rMiscellaneous import load_L2R_file
from ScikitLearnLocalModificado.forest import Forest

def setScoreGen(forest, population, metrica, fileCache, Vetores_Vali, numTrees, sizeGen, nFeatures):

    base_pred = forest.predict(Vetores_Vali.x, [1]*numTrees, fileCache)
    base_ndcg, _ = modelEvaluation(Vetores_Vali, base_pred, nFeatures)

    predictionListMean = []
    predictionListNoMean = []

    riskList = []
    riskListNoMean = []

    for ind in population:

        if ind.bool_score != 1:

            scores = forest.predict(Vetores_Vali.x, ind.mask, fileCache)
            valor_ndgc, _ = modelEvaluation(Vetores_Vali, scores, nFeatures)

            # if metrica == "ndcg":
            ind.setScore(valor_ndgc, "ndcg")
            # if metrica == "trisk":
            trisk, vectrisk = getTRisk(valor_ndgc, base_ndcg, 5)
            ind.setScore(trisk, "trisk", vectrisk)

            ind.bool_fit = 1

        if metrica == "spea2":
            ndcg, vetndcg = ind.getScore("ndcg")

            predictionListMean.append(ndcg)
            predictionListNoMean.append(vetndcg)

            trisk, vettrisk = ind.getScore("trisk")

            riskList.append(trisk)
            riskListNoMean.append(vettrisk)

    if metrica == "spea2":
        prediction = basicStructure()
        prediction.marginal = predictionListMean
        prediction.mat = predictionListNoMean
        prediction.greaterIsBetter = 1
        prediction.pvalue = 1
        prediction.variance = 1

        risk = basicStructure()
        risk.marginal = riskList
        risk.mat = riskListNoMean
        risk.greaterIsBetter = 1
        risk.pvalue = 1
        risk.variance = 1

        vetIndividuos = []
        for i in range(sizeGen):
            vetIndividuos.append(i)

        matrizDominance = obtainDominace("prediction", "trisk", "null", prediction, None, risk, None,
                                         vetIndividuos, sizeGen)

        cont = 0
        for i in population:
            if matrizDominance[cont] == 0:
                i.fitnessSpea2 = 2
            else:
                i.fitnessSpea2 = 1 / matrizDominance[cont]
            cont += 1

    return population


def main(colecao, fold, metrica, numTrees, original_geracoes_comparar, superMask=[]):

    # Manipular a DataBase
    name_train = colecao + "/Fold" + str(fold) + "/" + "Norm.train.txt"
    name_vali = colecao + "/Fold" + str(fold) + "/" + "Norm.vali.txt"
    name_test = colecao + "/Fold" + str(fold) + "/" + "Norm.test.txt"

    nFeatures = nFeaturesColection(colecao)
    MASK = [1] * nFeatures
    sizeGen=75
    '''
    Instacia as Classes
    Algoritmo Genetico e o RandomForest Modificado
    '''
    oGA = GeneticAlgorithm(sizeGen=sizeGen, numTrees=numTrees, typeScore=metrica, elitist=1)
    forest = Forest(n_estimators=numTrees, n_jobs=-1)

    numGeneration = original_geracoes_comparar

    fileCache = colecao.replace("../Colecoes",  "./TreesRF")  + "/AllTrees" + str(numTrees) + "Fold" + str(fold) + ".pickle"


    # Verifica se ja possui arvores criadas
    #  Se não as cria
    if not (os.path.isfile(fileCache)):

        X, y, z = load_L2R_file(name_train, MASK)
        # Vetores_train = ChangeName(X, y, z)
        forest.fit(X, y)

        trees = forest.estimators_

        try:
            os.mkdir(colecao.replace("../Colecoes",  "./TreesRF"))
        except OSError:
            print('')

        with open(fileCache, 'wb') as pFile:
            pickle.dump(trees, pFile)


    X2, y2, z2 = load_L2R_file(name_vali, MASK)
    Vetores_Vali = ChangeName(X2, y2, z2)

    #     Test Final
    X3, y3, z3 = load_L2R_file(name_test, MASK)
    Vetores_Test = ChangeName(X3, y3, z3)




    if original_geracoes_comparar > 1:

        #  Criando e dando o Score para a geracao inicial
        oGA.generatePopulation()
        oGA.population = setScoreGen(forest, oGA.population , metrica, fileCache, Vetores_Vali, numTrees, sizeGen, nFeatures)

        # Armazena no Arquive
        bag = Arquive(oGA.sizeGen * 2, metrica)
        bag.type = oGA.population[0].__class__
        bag.appendBag(oGA.population)

        # Iteracao principal do GA

        geracao_inicial = oGA.population

        ''' Iteracao + 1 para fitar a ultima geracao'''

        time_geracao = time.time()

        geracao_temp = []
        for numero_da_geracao in range(1, numGeneration + 1):

            if numero_da_geracao < numGeneration:
                ''' função GA, externa, recebe geracao com fitness, numero da geracao'''
                geracao_temp = oGA.runGA(geracao_inicial, numero_da_geracao + 1)
            if numero_da_geracao >= numGeneration:
                geracao_temp = geracao_inicial

            geracao_nova_com_fit = setScoreGen(forest, geracao_temp, metrica, fileCache, Vetores_Vali, numTrees, sizeGen, nFeatures)  ## Geração com Fitness
            geracao_temp = []

            ''' Agrupamento de Geracao com ou sem Elistia'''
            populacao = []
            if oGA.elitist == 1:
                populacao = oGA.elitistGroup(geracao_inicial, geracao_nova_com_fit)
            else:
                populacao = geracao_nova_com_fit

            '''
            imprimir a populacao por arquivo
            '''
            geracao_inicial = []
            geracao_inicial = populacao
            geracao_nova_com_fit = []

            time_fim_geracao = time.time()

            imprimir_individuo(colecao + '/' + metrica, populacao, numTrees, numero_da_geracao, fold, [oGA.boolSelectTourRoul, oGA.boolCrossUniPoint, oGA.elitist], time_fim_geracao - time_geracao)
            bag.appendBag(populacao)

        The_Best = bag.getBag(1)


        scoresBaseTest = forest.predict(Vetores_Test.x, [1] * numTrees, fileCache)
        base_ndgc, _ = modelEvaluation(Vetores_Test, scoresBaseTest, nFeatures)

        time_thebest = time.time()

        scoresTest = forest.predict(Vetores_Test.x, The_Best.mask, fileCache)
        scoreNDCG, _ = modelEvaluation(Vetores_Test, scoresTest, nFeatures)

        The_Best.setScore(scoreNDCG, "ndcg")
        trisk, vectrisk = getTRisk(scoreNDCG, base_ndgc, 5)
        The_Best.setScore(trisk, "trisk", vectrisk)

        time_final_thebest = time.time()

        imprimir_individuo(colecao + '/' + metrica, [The_Best], numTrees, numGeneration, fold,
                           [oGA.boolSelectTourRoul, oGA.boolCrossUniPoint, oGA.elitist],
                           time_final_thebest - time_thebest)

    elif original_geracoes_comparar == 1:

        time_original = time.time()

        origin = Individual([1]*numTrees, 1)

        scoreOriginal = forest.predict(X3, [1]*numTrees, fileCache)

        scoreNDCG, _ = modelEvaluation(Vetores_Test, scoreOriginal, nFeatures)

        origin.setScore(scoreNDCG, "ndcg")

        time_final_original = time.time()

        imprimir_individuo(colecao + '/' + metrica, [origin], numTrees, 1, fold,
                           [oGA.boolSelectTourRoul, oGA.boolCrossUniPoint, oGA.elitist],
                           time_final_original - time_original)

    else:

        time_original = time.time()

        origin = Individual(superMask, 1)

        scoreOriginal = forest.predict(X3, superMask, fileCache)

        scoreNDCG, _ = modelEvaluation(Vetores_Test, scoreOriginal, nFeatures)

        origin.setScore(scoreNDCG, "ndcg")

        time_final_original = time.time()

        imprimir_individuo(colecao + '/' + metrica, [origin], numTrees, 1, fold,
                           [oGA.boolSelectTourRoul, oGA.boolCrossUniPoint, oGA.elitist],
                           time_final_original - time_original)

    return 1

'''
colecao =   web10k
            2003_td_dataset
            2004_td_dataset
fold =      1 2 3 4 5

metrica =   ndcg, trisk, spea2
geradorRelatorioValidacao
numTrees = 300, 500, 750

'''

for metrica in ["ndcg","trisk", "spea2"]:##, "trisk","ndcg"]:
    for numTrees in [100,300, 500,750]:##,500,750,1000]:
        for fold in range(1, 6):
            for geracoes in [1,30]:
                main("../Colecoes/2003_td_dataset", fold, metrica, numTrees, geracoes)
                geradorRelatorioValidacao("2003_td_dataset/" + metrica, numTrees, fold)
        geradorRelatorioFinal("2003_td_dataset/" + metrica, numTrees)