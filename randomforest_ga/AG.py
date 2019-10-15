# coding=utf-8
import random  # GERACAO NUMEROS ALEATORIOS
# SELECAO     #MUTACAO   #CROSSOVER
from copy import copy
from math import ceil  # BIBLIOTECA DE ARREDONDAMENTO NUMERICO
from threading import Thread  # BIBLIOTECA DE GERACAO DE SUBPROCESSOS (THREADS)
from time import time
from warnings import warn

##from sympy import *

import numpy as np

from External import Individual, modelEvaluation, getTRisk, imprimir_individuo
from External.ga import sortGetIndice, chargeToPredict, Arquive
from External.h_functionsFilter import obtainDominace, basicStructure


class AG:
    def __init__(self, forest, mutacao, crossover, tipoSelecao, tipoCrossover, elitismo, tamPopulacao, tamIndividuo,
                 fitnes, geracoes, dataBase, fileCache, numthreads, fold,  AllTrees=[]):
        self.forest = forest
        self.mutacao = mutacao  # propriedade de mutacao
        self.crossover = crossover
        self.typeSection = tipoSelecao
        self.typeCross = tipoCrossover
        self.elitism = elitismo
        self.tamPopulacao = tamPopulacao  # numero populacao inicial
        self.tamIndividuo = tamIndividuo
        self.metrica = fitnes  # funcao de avaliacao NDCG, TRISK, SPEA2
        self.dataBase = dataBase  # Base q, x, y
        self.fileCache = fileCache
        self.populacao = []  # vetor contendo a populacao
        self.fitPopulacao = []  # vetor com resultado avaliacao de cada elem da populacao
        self.atualGeracao = 1
        self.fold = fold
        self.AllTrees = AllTrees

        ## cria o arquive
        self.arquive = Arquive(tamPopulacao*2, mode=self.metrica)

        self.NUM_THREADS = numthreads
        self.geracoes = geracoes
        start = time()
        self.populacaoInicial = []
        self.__initGeracao()

        self.__setFitnes()
        self.populacaoInicial = self.populacao

        self.arquive.appendBag(self.populacaoInicial)

        imprimir_individuo(self.fileCache + '/' + self.metrica, self.populacaoInicial,
                           self.tamIndividuo,
                           self.atualGeracao,
                           self.fold, [self.typeSection, self.typeCross, self.elitism], time() - start)

    def RUN(self):
        tempSize = self.tamPopulacao
        while self.atualGeracao < self.geracoes + 1:
            start = time()

            self.tamPopulacao = tempSize

            self.populacao = []         ## limpa a variavel
            self.__nextGeracao()        ## Gera a nova populacao ->  GA

            self.__setFitnes()          ## Avalia os individuos da geração nova

            self.arquive.appendBag(self.populacao)

            self.tamPopulacao = self.arquive.size
            # self.elitismFun(self.populacaoInicial, self.arquive.getBag())  ## aplica o elitismo, selecionando somente os 5 mais fortes
            self.elitismFun(self.populacaoInicial, self.populacao)  ## aplica o elitismo, selecionando somente os 5 mais fortes

            finish = time()

            self.atualGeracao += 1

            imprimir_individuo(self.fileCache + '/' + self.metrica, self.populacaoInicial, self.tamIndividuo,
                               self.atualGeracao,self.fold, [self.typeSection, self.typeCross, self.elitism], finish - start)


        return self.populacaoInicial

    # OK - GABRIEL
    def __initGeracao(self):
        try:
            threads = []
            for i in range(self.NUM_THREADS):
                threads.append(self.__startThreads('self.geraPopulacao',
                                                   {}))  # DISPARANDO THREADS PARA GERAR POPULACAO E AVALIACAO DOS ELEMENTOS
            self.__endThreads(threads)  # ESPERA THREADS ACABAREM
        except Exception as e:
            print(e)

    # OK - GABRIEL
    def __nextGeracao(self):
        try:
            threads = []
            tamMutacao = round(ceil((float(self.tamPopulacao) + 1) / 2))  # NUMERO DE INDIVIDUOS QUE SOFRERAO MUTACAO

            faixa = int(ceil(tamMutacao / self.NUM_THREADS))

            r = int((tamMutacao + 1 > self.NUM_THREADS and self.NUM_THREADS or tamMutacao))  # VERIFICA SE NUM THREADS E MAIOR QUE NUM ELEMENTOS QUE SOFRERAO MUTACAO

            if r != 1:
                for i in range(r):
                    threads.append(self.__startThreads('self.Mating', {
                        'tam': faixa}))  # DISPARANDO THREADS PARA EFETUAR MUTACAO NOS PIORES ELEMENTOS DA POPULACAO EM FAIXAS
                self.__endThreads(threads)  # ESPERA THREADS ACABAREM
            else:
                self.Mating((self.tamPopulacao+1) / 2)

        except Exception as e:
            print(e)

    # OK - GABRIEL
    def __setFitnes(self):  # IMENDA ROTAS MAIS BEM AVALIADAS EM ROTAS MAL AVALIADAS A PARTIR DE PONTO EM COMUM
        try:

            faixa = int(ceil(self.tamPopulacao / self.NUM_THREADS))

            r = int((faixa > self.NUM_THREADS and self.NUM_THREADS or faixa))  # VERIFICA SE NUM THREADS E MAIOR QUE NUM ELEMENTOS QUE SOFRERAO MUTACAO

            threads = []
            if r != 1:
                for i in range(0, r):
                    threads.append(self.__startThreads('self.geraFitnes', {'ini': i,'tam': faixa}))  # DISPARANDO THREADS PARA EFETUAR MUTACAO NOS PIORES ELEMENTOS DA POPULACAO EM FAIXAS
                self.__endThreads(threads)
            else:
                self.geraFitnes(0, self.tamPopulacao)

            if self.metrica == "spea2":

                predictionListMean = []
                predictionListNoMean = []

                riskList = []
                riskListNoMean = []

                vetIndividuos = []
                cont = 0
                for ind in self.populacao:

                    vetIndividuos.append(cont)
                    cont += 1

                    ndcg, vetndcg = ind.getScore("ndcg")

                    predictionListMean.append(ndcg)
                    predictionListNoMean.append(vetndcg)

                    trisk, vettrisk = ind.getScore("trisk")

                    riskList.append(trisk)
                    riskListNoMean.append(vettrisk)

                prediction = basicStructure()
                prediction.marginal = predictionListMean
                prediction.greaterIsBetter = 1
                # prediction.mat = predictionListNoMean
                # prediction.pvalue = 0.005
                # prediction.variance = 1

                risk = basicStructure()
                risk.marginal = riskList
                risk.greaterIsBetter = 1
                # risk.mat = riskListNoMean
                # risk.pvalue = 0.005
                # risk.variance = 1
                


                matrizDominance = obtainDominace("prediction", "trisk", "null", prediction, None, risk, None,
                                                 vetIndividuos, self.tamPopulacao)
                if matrizDominance.max() == 0.0:
                    warn("Error: obtainDominace")

                max = matrizDominance.max()

                ## falta calcular a densidade dos individius para desempate
                ## file:///home/gabrielbraga/Downloads/tese-final-text%20(50).pdf

                for cont in range(0, self.tamPopulacao):
                    self.populacao[cont].bool_max_min = 0
                    self.populacao[cont].setScore(matrizDominance[cont], "spea2")

        except Exception as e:
            print(e)

    # OK - GABRIEL
    def __startThreads(self, funcAlvo, k):
        try:
            t = Thread(target=eval(funcAlvo), kwargs=k)
            t.start()
            return t
        except Exception as e:
            print(e)

    # OK - GABRIEL
    def __endThreads(self, threads):
        for t in threads:
            t.join(10)
        return

    # OK - GABRIEL
    def geraPopulacao(self):
        if self.NUM_THREADS == 0:
            tam = self.tamPopulacao
        else:
            tam = int(ceil(float(self.tamPopulacao) / self.NUM_THREADS))

        try:
            for i in range(tam):

                p = np.zeros(self.tamIndividuo)
                for cont in range(len(p)):
                    p[cont] = random.randint(0, 1)  ## Criar opção de Porcentagem de Genes Dominantes

                self.populacao.append(Individual(p, 1))

            if self.populacao[0].mask.min != self.populacao[0].mask.max:
                self.populacao[0] = Individual(np.ones(self.tamIndividuo), 1)

            if len(self.populacao) == self.tamPopulacao:  # COMO AS THREADS PODEM SE DIVIDIR EM MAIS ELEMENTOS QUE O TAMANHO ORIGINAL PRAMETRIZADO
                return False  # DEVIDO AO ARREDONDAMENTO DE VEZES PARA CADA THREAD CRIAR ELEMENTOS
            # VERIFICO AQUI SE TAMANHO TOTAL DE ELEMENTOS JA FOI CRIADO ANTES SE INSERIR CADA ELEMENTO
            return True
        except Exception as e:
            print(e)

    def Mating(self, tam):
        try:
            ###################################################### 	Selection
            selection = []

            if self.typeSection == "torn":
                selection = self.selectionTorn()
            elif self.typeSection == "roul":
                selection = self.selectionRoul()

            for t in range(int(tam)):

                selToCross = []
                result = random.sample(range(0, self.tamPopulacao), 2)

                selToCross.append(selection[result[0]])
                selToCross.append(selection[result[1]])

                ###################################################### Cross
                if self.typeCross == "pontual":
                    self.crossPontual(selToCross)
                elif self.typeCross == "region":
                    self.crossRegion(selToCross)

            ###################################################### Mutation
            self.mutationPontual()

            return True
        except Exception as e:
            return False

    # OK - GABRIEL
    def selectionTorn(self):
        try:
            tamGeracao = self.tamPopulacao
            numSelTorn = 2

            theSelected = []

            status = True
            while status:
                contidos = []

                for n in range(numSelTorn):
                    contidos.append(random.randint(0, len(self.populacaoInicial) - 1))

                max = self.populacaoInicial[contidos[0]].getScore(self.metrica)[0]
                tempSelected = contidos[0]

                for c in contidos:
                    if (self.populacaoInicial[c].bool_max_min == 1 and max < self.populacaoInicial[c].getScore(self.metrica)[0]):
                        tempSelected = c
                        max = self.populacaoInicial[c].getScore(self.metrica)[0]

                    elif (self.populacaoInicial[c].bool_max_min == 0 and max > self.populacaoInicial[c].getScore(self.metrica)[0]):
                        tempSelected = c
                        max = self.populacaoInicial[c].getScore(self.metrica)[0]

                theSelected.append(tempSelected)
                if len(theSelected) == self.tamPopulacao:
                    status = False

            return theSelected

        except Exception as e:
            return False

    # OK - GABRIEL
    def selectionRoul(self):
        try:
            probability = []
            for p in self.populacao:
                probability.append(p)

            probabilityChance = sum(probability)

            posSelect = []
            status = True
            while status:
                chanceSelect = random.uniform(0, probabilityChance)

                sum_acc = 0
                pos = 0

                while (sum_acc < chanceSelect):
                    sum_acc += probability[pos]
                    pos += 1

                posSelect.append(pos-1)
                if len(posSelect) == self.tamPopulacao:
                    status = False

            return posSelect

        except Exception as e:
            return False

    # OK - GABRIEL
    def crossPontual(self, selects):
        try:
            pai1 = copy(self.populacaoInicial[selects[0]].mask)
            pai2 = copy(self.populacaoInicial[selects[1]].mask)

            ind1 = [0] * self.tamIndividuo
            ind2 = [0] * self.tamIndividuo

            for cross in range(self.tamIndividuo):
                valor_sorteado = random.randint(0, 101) / 100

                if valor_sorteado <= 0.5:
                    ind1[cross] = pai1[cross]
                else:
                    ind1[cross] = pai2[cross]

                valor_sorteado = random.randint(0, 101) / 100

                if valor_sorteado <= 0.5:
                    ind2[cross] = pai1[cross]
                else:
                    ind2[cross] = pai2[cross]

            self.populacao.append(Individual(ind1, self.atualGeracao + 1))
            self.populacao.append(Individual(ind2, self.atualGeracao + 1))

            return True
        except Exception as e:
            return False

    # OK - GABRIEL
    def crossRegion(self, selects):
        try:
            warn("nao implementado croosRegion")
            return True
        except Exception as e:
            return False

    # OK - GABRIEL
    def mutationPontual(self):
        try:

            for i in range(self.tamPopulacao):
                ## Chance de um individuo sofrer mutacao
                valor_sorteado1 = random.randint(0, 101) / 100

                if valor_sorteado1 <= self.mutacao: ## se houver mutacao

                    for m in range(self.tamIndividuo):
                        ## para cada gene a probabilidade de houver mutacao
                        valor_sorteado2 = random.randint(0, 101) / 100

                        if valor_sorteado2 <= self.mutacao:
                            if self.populacao[i].mask[m] == 1:
                                self.populacao[i].mask[m] = 0
                            else:
                                self.populacao[i].mask[m] = 1
            return True
        except Exception as e:
            return False

    # OK - GABRIEL
    def elitismFun(self, populationOLD, populationNEW):
        try:
            x = slice(0, self.tamPopulacao)
            fatorElitismo = (self.tamPopulacao / 4)

            if self.elitism == 0:

                self.populacaoInicial = populationNEW[x]
                self.populacao = []
            else:
                populationNEW = populationNEW[x]

                scores = []
                contElit = []
                for i in populationOLD:
                    scores.append(i.getScore(self.metrica)[0])
                    contElit.append(1)
                for i in populationNEW:
                    contElit.append(0)
                    scores.append(i.getScore(self.metrica)[0])

                if self.metrica == "spea2":
                    crescente = 0
                else:
                    crescente = 1

                indices = sortGetIndice(scores, crescente=crescente)

                indOrdenados = indices
                temp = []

                temp.extend(populationOLD)
                temp.extend(populationNEW)

                self.populacaoInicial = []

                sizeElit = 0
                for i in indOrdenados:
                    if contElit[i] == 1 and sizeElit < fatorElitismo:
                        sizeElit += 1
                        self.populacaoInicial.append(temp[i])
                    elif contElit[i] == 0:
                        self.populacaoInicial.append(temp[i])

                self.populacaoInicial = self.populacaoInicial[x]
                self.populacao = []
            return True
        except Exception as e:
            return False

    # OK - GABRIEL
    def geraFitnes(self, ini, tam):  # AVALIA CADA ROTA POSSIVEL DA POPULACAO
        try:
            # nFeatures = nFeaturesColection("2003_td_dataset")
            if self.fileCache.count("2003_td_dataset"):
                nFeatures = 64
            else:
                nFeatures = 136

            self.forest.estimators_ = chargeToPredict(self.fileCache, [1] * self.tamIndividuo, self.tamIndividuo, self.fold)
            self.forest.n_estimators = len(self.forest.estimators_)
            self.forest.n_outputs_ = 1

            base_pred = self.forest.predict(self.dataBase.x)
            base_ndcg, _ = modelEvaluation(self.dataBase, base_pred, nFeatures)

            x = slice(tam * ini, tam * ini + tam, 1)
            # print("utilizando matrix de oob")
            oob = False

            for ind in self.populacao[x]:

                if ind.bool_score != 1:
                    self.forest.estimators_ = chargeToPredict(self.fileCache, ind.mask,
                                                              self.tamIndividuo, self.fold)
                    self.forest.n_outputs_ = 1
                    self.forest.n_estimators = len(self.forest.estimators_)




                    if oob == True:
                        vetor4 = Matrix(len(self.dataBase.y), 1, self.dataBase.y)
                        self.forest._set_oob_score(self.dataBase.x, vetor4)

                        # ind.setScore(self.forest.oob_prediction_, "ndcg")
                        valor_ndgc, _ = modelEvaluation(self.dataBase, self.forest.oob_prediction_, nFeatures)


                        for pc in range(len(valor_ndgc)):
                            if valor_ndgc[pc] == 0.0:
                                  valor_ndgc[pc] = 1.0

                        ind.setScore(valor_ndgc, "ndcg")

                        # ind.NDCG = self.forest.oob_score_

                    else:
                        scores = self.forest.predict(self.dataBase.x)
                        valor_ndgc, _ = modelEvaluation(self.dataBase, scores, nFeatures)

                        ind.setScore(valor_ndgc, "ndcg")

                        trisk, vectrisk = getTRisk(valor_ndgc, base_ndcg, 5)
                        ind.setScore(trisk, "trisk", vectrisk)

                    ind.bool_score = 1

            return True
        except Exception as e:
            return False
