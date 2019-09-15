# coding=utf-8
import csv
import numpy
import random
import os

import errno
import numpy as np


class ChangeName:
    def __init__(self, x, y, z):
        '''
        :param x: Consultas
        :param y: Relevante ou não relevante
        :param z: todos os documentos
        '''
        self.q = z
        self.x = x
        self.y = y


class Individual:
    def __init__(self, mask, generation):
        '''
        
        :param mask: Mascara de genes do individuo 
        :param generation: geracao onde foi criado o individuo
        '''

        self.mask = mask
        self.generation = generation

        self.vetNDCG = []
        self.NDCG = 0

        self.vetTrisk = 0
        self.Trisk = 0

        self.Spea2 = 0
        self.vetTStastitico = 0

        self.bool_max_min = 1
        self.bool_score = 0

    def setScore(self, score, type, vetorTrisk=0):
        if type == "ndcg":
            self.vetNDCG = score
            self.NDCG = np.average(score)
        elif type == "trisk":
            self.vetTrisk = vetorTrisk
            self.Trisk = score
        elif type == "spea2":
            self.vetTStastitico = score
            self.Spea2 = np.average(score)

    def getScore(self, type):
        if type == "ndcg":
            return self.NDCG, self.vetNDCG
        elif type == "trisk":
            return self.Trisk, self.vetTrisk
        elif type == "spea2":
            return self.Spea2, self.vetTStastitico

    def size(self):
        return len(self.mask)

    def getGen(self):
        return self.generation


class GeneticAlgorithm:

    def __init__(self, sizeGen, numTrees, typeScore, elitist=0, Selection=1, Cross=1, probabilityCross=0.5, probalityMutation=0.3):
        '''
        
        :param sizeGen: Quantidade de individuos na geração
        :param numTrees: Numero de genes em um individuo
        :param typeScore: tipos de score, NDCG, TRISK, GeoRISK
        :param elitist: 0,1 (não,sim) para elitismo
        :param Selection: 0,1 Selecao por Torneio ou Roleta
        :param Cross: 0,1 Crossover uniforme ou por Ponto a Ponto
        :param probabilityCross: Probabilidade de haver crossover
        :param probalityMutation: Probabilidade de haver mutacao
        '''
        self.population = []
        self.numberGen = 1

        self.typeScore = typeScore
        self.newGen = []

        self.sizeGen = sizeGen
        self.numTrees = numTrees

        self.elitist = elitist
        self.boolSelectTourRoul = Selection  ## Torneio 1 Roleta 0
        self.boolCrossUniPoint = Cross  ## Uniforme 1 Pontual 0

        self.probability = probabilityCross  ## de crossover
        self.probabilityMutation = probalityMutation  ## de Mutation


    def runGA(self, geracao_com_fitness, numero_da_geracao):
        '''
        
        :param geracao_com_fitness: geracao que havera o processo de selecao
        :param numero_da_geracao: numero da geracao
        :return: nova geracao com o resultado da operacoes geneticas na geracao_com_fitness
        '''

        self.population = geracao_com_fitness
        self.numberGen    = numero_da_geracao

        sizeGen = len(self.population)
        probabilidades = np.zeros(sizeGen)

        for i in range(sizeGen):
            probabilidades[i] = self.myFitness(self.population[i])

        self.nova_geracao = []
        sizeNewGeration = 0

        while sizeNewGeration < sizeGen:

            ''' Tipo de Selecao de Pais'''
            [ind1, ind2] = self.selection(tipo=self.boolSelectTourRoul)

            ''' Tipo de Selecao de Crossover'''
            cruzamentos = self.crossover(self.population[int(ind1)], self.population[int(ind2)],
                                         tipo=self.boolSelectTourRoul)

            ''' Mutação nos dois filhos'''
            valor_sorteado = random.randint(0, 101) / 100
            if valor_sorteado <= self.probabilityMutation:
                cruzamentos[0] = self.mutation(cruzamentos[0], tipo=1)

            valor_sorteado = random.randint(0, 101) / 100
            if valor_sorteado <= self.probabilityMutation:
                cruzamentos[1] = self.mutation(cruzamentos[1], tipo=1)

            ''' Salvando novos individuos'''

            self.nova_geracao.append(Individual(cruzamentos[0], self.numberGen))
            self.nova_geracao.append(Individual(cruzamentos[1], self.numberGen))

            sizeNewGeration += 2

        return self.nova_geracao

    def generatePopulation(self):

        populationStart = []

        for j in range(self.sizeGen):
            p = np.zeros(self.numTrees)
            for i in range(self.numTrees):
                p[i] = random.randint(0, 1)  ## Criar opção de Porcentagem de Genes Dominantes
            populationStart.append(p)

        ''' Converte mascara para individuo'''
        individuos_g1 = []

        for g in populationStart:
            individuos_g1.append(Individual(g, 1))  ### Geracao Inicial Parametro = 1
        individuos_g1[0] = Individual(np.ones(self.numTrees), 1)
        self.population = individuos_g1
        return individuos_g1

    ''' Tipo: 1 para torneio 0 para roleta'''

    def selection(self, total_num_sorteados=2, tipo=1):

        if tipo == 0:  ### 0 para Roleta
            num_individuos = len(self.population)
            probabilidades = np.zeros(num_individuos)
            for i in range(num_individuos):
                probabilidades[i] = self.myFitness(self.population[i])
            total = sum(probabilidades)
            tempIndSelected = np.zeros(total_num_sorteados)
            cont1 = 0
            while (cont1 < total_num_sorteados):
                valor = random.uniform(0, total)
                soma_acumulada = 0
                posicao = 0
                while (soma_acumulada < valor):
                    soma_acumulada += probabilidades[posicao]
                    posicao += 1
                contido = 0
                cont2 = 0
                while (cont2 < cont1):
                    if (tempIndSelected[cont2] == posicao - 1):
                        contido = 1
                        break
                    cont2 += 1
                if not (contido):
                    tempIndSelected[cont1] = posicao - 1
                    cont1 += 1

        elif tipo == 1:  ## 1 Selecao por Torneio
            tempIndSelected = [-1] * total_num_sorteados
            total_sorteados = 0
            while (total_sorteados < total_num_sorteados):
                tempIndSelected[total_sorteados] = self.torn(total_num_sorteados)
                total_sorteados += 1

        return tempIndSelected

    def torn(self, quantidade_competidores=2):
        sorteados = np.zeros(quantidade_competidores)
        total_sorteados = 0
        total_individuos = len(self.population)
        while total_sorteados < quantidade_competidores:
            valor_aleatorio = random.randint(0, total_individuos - 1)
            contido = 0
            cont1 = 0

            while cont1 < total_sorteados:
                if sorteados[cont1] == valor_aleatorio:
                    contido = 1
                    break
                cont1 += 1
            if not contido:
                sorteados[total_sorteados] = valor_aleatorio
                total_sorteados += 1
        posi_inicial = 0
        for i in range(quantidade_competidores):
            if self.myFitness(self.population[int(sorteados[i])]) > self.myFitness(self.population[int(sorteados[posi_inicial])]):
                posi_inicial = i
        return sorteados[posi_inicial]

    ''' Tipo: 1 para uniforme 0 para pontual'''

    def crossover(self, individuoA, individuoB, tipo=1):
        individuoAx = individuoA.mask
        individuoBx = individuoB.mask

        if tipo == 0:  ## Cruzamento em um ponto
            tamanho = individuoA.len()
            valor_sorteado = random.randint(0, 101) / 100
            if (valor_sorteado <= self.probability):
                posicao = random.randint(0, tamanho - 1)
                temp = individuoAx[posicao]
                individuoAx[posicao] = individuoBx[posicao]
                individuoBx[posicao] = temp

        elif tipo == 1:  ## Cruzamento uniforme
            for i in range(len(individuoAx)):
                valor_sorteado = random.randint(0, 101) / 100
                if valor_sorteado <= self.probability:
                    temp = individuoAx[i]
                    individuoAx[i] = individuoBx[i]
                    individuoBx[i] = temp

        return [individuoAx, individuoBx]

    ''' Tipo= 1 para uniforme 0 para pontual'''

    def mutation(self, mask, tipo=1):

        if tipo == 1:  ## Mutation uniforme
            valor_sorteado = random.randint(0, 101) / 100
            individuox = mask
            for posicao in range(len(mask)):
                if valor_sorteado <= self.probabilityMutation:
                    if individuox[posicao] == 1:
                        individuox[posicao] = 0
                    else:
                        individuox[posicao] = 1
        if tipo == 0:  ## Mutation em um ponto
            valor_sorteado = random.randint(0, 101) / 100
            individuox = mask
            posicao = random.randint(0, len(mask))

            if valor_sorteado <= self.probabilityMutation:
                if individuox[posicao] == 1:
                    individuox[posicao] = 0
                else:
                    individuox[posicao] = 1
        return individuox

    def elitistGroup(self, populationA, populationB):
        sizeGen = len(populationB) + len(populationA)

        populationNova = []

        populationNova.extend(populationA + populationB)

        for i in range(sizeGen):
            j = i + 1
            while j < sizeGen:
                if self.myFitness(populationNova[i]) < self.myFitness(populationNova[j]):
                    temp = populationNova[i]
                    populationNova[i] = populationNova[j]
                    populationNova[j] = temp
                j += 1

        x = slice(0, len(populationA), 1)
        return populationNova[x]

    def myFitness(self, individuo):
        fitness, vect = individuo.getScore(self.typeScore)
        return fitness


class Arquive:
    def __init__(self, tamanho=100, mode="ndcg"):
        self.mode = mode
        self.size = tamanho
        self.arq = []
        self.type = 0

    def fit(self, individuo):
        fitness, vect = individuo.getScore(self.mode)
        return fitness

    def getBag(self, params='default'):

        if params == 'default':
            return self.arq
        elif params == 1:
            return self.arq[0]
        elif (params > 1) and (params < len(self.arq)):
            x = slice(0, params, 1)
            return self.arq[x]

    def appendBag(self, newItens):
        ArquiveNew = []

        sizeA = len(newItens) + len(self.arq)
        sizeM = self.size

        ArquiveNew.extend(self.arq + newItens)

        for i in range(sizeA):
            j = i + 1
            while j < sizeA:
                n1 = self.fit(ArquiveNew[i])
                n2 = self.fit(ArquiveNew[j])
                if n1 < n2:
                    temp = ArquiveNew[i]
                    ArquiveNew[i] = ArquiveNew[j]
                    ArquiveNew[j] = temp
                j += 1

        if sizeA > sizeM:
            x = slice(0, sizeM, 1)
            self.arq = []
            self.arq = ArquiveNew
        else:
            x = slice(0, sizeA, 1)
            self.arq = []
            self.arq = ArquiveNew

def nFeaturesColection(dataset):
    nFeatures = 0

    dataset = dataset.replace("/mnt/c/Users/jefma/Documents/GitHub/PIBIC/Gabriel/TCC_1/",  "")
    if dataset=="web10k":
        nFeatures = 136
    elif dataset=="2003_td_dataset":
        nFeatures = 64
    elif dataset=="2004_td_dataset":
        nFeatures = 64
    # elif dataset=="web10k":
    #     nFeatures = 700
    else:
        print("There is no evalution to this dataset: ", dataset)
        exit(0)

    return nFeatures


def files_path04(path):
    list_arquives = []
    for p, _, files in os.walk(os.path.dirname(path)):
        for file in files:
            list_arquives.append(os.path.join(p, file))

    return  list_arquives


def somar(valores):
    soma = 0.0
    for v in valores:
        if v != '':
            soma += float(v)
    return soma

def media(valores):
    soma = somar(valores)
    qtd_elementos = len(valores)
    media = soma / float(qtd_elementos)
    return media

def variancia(valores):
    _media = media(valores)
    soma = 0
    _variancia = 0

    for valor in valores:
        if valor != '':
            soma += numpy.math.pow((float(valor) - _media), 2)
    _variancia = soma / float(len(valores))
    return _variancia

def geradorRelatorioValidacao(path, ntrees, fold):

    list = files_path04('./FinalResults/' + path + '/' + str(ntrees) + '/Fold' + str(fold) + '/')

    vetor_numero_geracao = []
    vetor_numero_geracao_individuos = []
    vetor_ndcg = []
    vetor_trisk = []
    vetor_tempo = []

    for l in list:

        temp_geracao = []
        temp_ndcg = []
        temp_trisk = []

        temp_tempo = 0

        if l.count('Geracao') == 1:

            with open(l, 'r') as ficheiro:
                reader = csv.reader(ficheiro)
                vetor_numero_geracao.append(l.replace('./FinalResults/' + path + '/' + str(ntrees) + '/Fold' + str(fold) + '/Geracao_', '').replace('.', 'w').split('w')[0])

                for linha in reader:
                    if linha[0] != 'Geracao':
                        temp_geracao.append(linha[0])

                        temp_ndcg.append(linha[1])
                        temp_trisk.append(linha[2])

                        if temp_tempo == 0:
                            temp_tempo = linha[len(linha) - 1]

                vetor_numero_geracao_individuos.append(temp_geracao)

                vetor_ndcg.append(temp_ndcg)
                vetor_trisk.append(temp_trisk)
                vetor_tempo.append(temp_tempo)

    name_arq = path + '' + str(ntrees) + 'relatorio_validacao_fold_' + str(fold) + '.csv'

    arquivo = open(name_arq, 'w')
    arquivo.write("Geracao,Maior NDCG, Menor NDCG, Media NDCG, Variancia NDCG, Maior TRISK, Menor TRISK, Media TRISK, Variancia TRISK, Individuo Novo, "
                  "Individuo Velho, Tempo\n")

    cont = 0
    for g in range(len(vetor_numero_geracao)):
        arquivo.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(str(vetor_numero_geracao[cont]),
                                                                 str(max(vetor_ndcg[cont])),
                                                                 str(min(vetor_ndcg[cont])),
                                                                 str(media(vetor_ndcg[cont])),
                                                                 str(variancia(vetor_ndcg[cont])),
                                                                 str(max(vetor_trisk[cont])),
                                                                 str(min(vetor_trisk[cont])),
                                                                 str(media(vetor_trisk[cont])),
                                                                 str(variancia(vetor_trisk[cont])),
                                                                 str(max(vetor_numero_geracao_individuos[cont])),
                                                                 str(min(vetor_numero_geracao_individuos[cont])),
                                                                 str(vetor_tempo[cont])))
        cont += 1
    arquivo.close()

def geradorRelatorioFinal(path, ntrees):

    list = files_path04('./FinalResults/' + path + '/' + str(ntrees) + '/')

    vetor_folds = []
    vetor_ndcg = []

    vetor_tempo = []

    for l in list:

        if l.count('TheBests') == 1:
            with open(l, 'r') as ficheiro:
                reader = csv.reader(ficheiro)

                vetor_folds.append(l.replace('./FinalResults/' + path + '/' + str(ntrees) + '/', '', 1))

                for linha in reader:
                    if linha[0] != 'Geracao' and linha[0] != '':
                        vetor_ndcg.append(linha[1])
                        vetor_tempo.append(linha[len(linha) - 1])

        if l.count('Original') == 1:
            with open(l, 'r') as ficheiro:
                reader = csv.reader(ficheiro)

                vetor_folds.append(l.replace('./FinalResults/' + path + '/' + str(ntrees) + '/', ''))

                for linha in reader:
                    if linha[0] != 'Geracao' and linha[0] != '':
                        vetor_ndcg.append(linha[1])
                        vetor_tempo.append(linha[len(linha) - 1])



    name_arq = 'FinalResults' + str(ntrees) + 'relatorio_final' + '.csv'

    arquivo = open(name_arq, 'w')
    arquivo.write("Fold,NDCG Test, Tempo do Otimizado\n")


    cont = 0
    for g in range(len(vetor_folds)):
        arquivo.write("{0},{1},{2}\n".format(str(vetor_folds[cont]),
                                                 str(vetor_ndcg[cont]),
                                                 str(vetor_tempo[cont])))

        cont += 1
    arquivo.close()

def imprimir_individuo(colecao, vetor_individuo, num_arvores, total_geracao, fold, ga, time=0):
    '''
    Final Por Fold			Pasta			Nome Arquivo
			                FinalResultados/			/TheBests
    coleção		Fitness		GA
    fold	n arvores	ndcg	map	    geração	g. Ind	qt. ind	selecao	crossover	mutacao	elitismo	mascara

    1	    1000	    0.358	0.24	50  	48	    30	    1	    1   	1	    1
    '''

    colecao = colecao.replace("../Colecoes/", "")
    path = "./FinalResults/" + str(colecao) + '/'+ str(num_arvores) + "/Fold" + str(fold)
    # print(path)

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # try:
    # os.mkdir(path)
    # except OSError:
    #     print('.')


    if len(vetor_individuo) == 1 and np.array(vetor_individuo[0].mask).tolist().count(1) == num_arvores:
        arquivo_name = './FinalResults/' + colecao + '/'+ str(num_arvores) + '/Original_' + str(fold) + '.csv'
    elif len(vetor_individuo) == 1:
        arquivo_name = './FinalResults/' + colecao + '/'+ str(num_arvores) + '/TheBests_' + str(fold)+ '.csv'
    elif len(vetor_individuo) == 1 and vetor_individuo[0].mask.cont(1) != num_arvores:
        arquivo_name = './FinalResults/' + colecao + '/'+ str(num_arvores)+ '/Compare_' + str(fold) + '.csv'
    else:
        arquivo_name = './FinalResults/' + colecao + '/'+ str(num_arvores)+ '/Fold' + str(fold) + '/Geracao_' + str(total_geracao) + '.csv'


    arquivo = open(arquivo_name, 'w')

    arquivo.write("Geracao,NDCG,TRisk,Spea2,")

    if type(vetor_individuo[0].vetNDCG) != 'int':
        arquivo.write("Vetor NDCG,")
        for size in range(len(vetor_individuo[0].vetNDCG)):
            arquivo.write(",")
    if str(type(vetor_individuo[0].vetTrisk)) != '<class \'int\'>':
        arquivo.write("Vetor TRisk,")
        for size in range(len(vetor_individuo[0].vetTrisk)):
            arquivo.write(",")
    if str(type(vetor_individuo[0].vetTStastitico)) != '<class \'int\'>':
        arquivo.write("Vetor Estatistico,")
        for size in range(len(vetor_individuo[0].vetTStastitico)):
            arquivo.write(",")

    arquivo.write("Selecao,Crossover,Elitismo\n")


    for individuo in vetor_individuo:

        arquivo.write(str(individuo.generation) + ',') ## Geracao

        arquivo.write(str(individuo.NDCG)  + ',') ## NDCG
        arquivo.write(str(individuo.Trisk) + ',') ## TRISK
        arquivo.write(str(individuo.Spea2) + ',') ## SPEA2

        try:
            for c in individuo.vetNDCG:
                arquivo.write(str(c) + ',')
            arquivo.write(',')
        except:
            _ = 0
        try:
            for c in individuo.vetTrisk:
                arquivo.write(str(c) + ',')
            arquivo.write(',')
        except:
            _ = 0
        try:
            for c in individuo.vetTStastitico:
                arquivo.write(str(c) + ',')
            arquivo.write(',')
        except:
            _ = 0
        arquivo.write('{0},{1},{2},,'.format(str(ga[0]), str(ga[1]),str(ga[2])))

        for i in individuo.mask:
            arquivo.write(str(i) + ',')

        arquivo.write(',' + str(time))

        arquivo.write("\n")

    arquivo.close()