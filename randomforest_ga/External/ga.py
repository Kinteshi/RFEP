# coding=utf-8
import csv
import errno
import os
import pickle

import matplotlib.pyplot as plt
import numpy
import numpy as np

# from External import modelEvaluation, getTRisk
# from External.h_functionsFilter import basicStructure, obtainDominace

textFinal = './../finalResults/'

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
            self.Trisk = score
            self.vetTrisk = vetorTrisk
        elif type == "spea2":
            # self.vetTStastitico = score
            self.Spea2 = score

    def getScore(self, type):
        if type == "ndcg":
            return self.NDCG, self.vetNDCG
        elif type == "trisk":
            return self.Trisk, self.vetTrisk
        elif type == "spea2":
            return self.Spea2, [0]

    def size(self):
        return len(self.mask)

    def getGen(self):
        return self.generation

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
                if self.mode != "spea2" and n1 < n2:
                    temp = ArquiveNew[i]
                    ArquiveNew[i] = ArquiveNew[j]
                    ArquiveNew[j] = temp
                if self.mode == "spea2" and n1 > n2:
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

    dataset = dataset.replace("C:/Users/jefma/Documents/GitHub/PIBIC/Colecoes/",  "")
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

    list = files_path04(textFinal + path + '/' + str(ntrees) + '/Fold' + str(fold) + '/')

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
                vetor_numero_geracao.append(int(l.replace(textFinal + path + '/' + str(ntrees) + '/Fold' + str(fold) + '/Geracao_', '').replace('.', 'w').split('w')[0]))

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

    name_arq = textFinal + path + '/' + str(ntrees) + '/relatorio_validacao_fold_' + str(fold) + '.csv'

    arquivo = open(name_arq, 'w')
    arquivo.write("Geracao,Maior NDCG, Menor NDCG, Media NDCG, Variancia NDCG, Maior TRISK, Menor TRISK, Media TRISK, Variancia TRISK, Individuo Novo, "
                  "Individuo Velho, Tempo\n")

    indices = sortGetIndice(vetor_numero_geracao, crescente=1)

    vetOrdenado = []

    Nvet = []
    NvetVMax = []
    NvetVMin = []
    NvetVMean = []

    Tvet = []
    TvetVMax = []
    TvetVMin = []
    TvetVMean = []

    for cont in indices:
        vetOrdenado.append(int(vetor_numero_geracao[cont]))

        Nvet.append(vetor_ndcg[cont])
        NvetVMax.append(float(max(vetor_ndcg[cont])))
        NvetVMin.append(float(min(vetor_ndcg[cont])))
        NvetVMean.append(float(media(vetor_ndcg[cont])))

        Tvet.append(vetor_trisk[cont])
        TvetVMax.append(float(max(vetor_trisk[cont])))
        TvetVMin.append(float(min(vetor_trisk[cont])))
        TvetVMean.append(float(media(vetor_trisk[cont])))

    dn = {'x': vetOrdenado,
          'y1': NvetVMax,
          'y2': NvetVMean,
          'y3': NvetVMin}

    dt = {'x': vetOrdenado,
          'y1': TvetVMax,
          'y2': TvetVMean,
          'y3': TvetVMin}

    ds = {'x': Nvet,
          'y': Tvet}

    if path.count("ndcg"):
        # style
        plt.style.use('seaborn-darkgrid')

        plt.plot('x', 'y1', data=dn, marker='', color="red", linewidth=2,label="Max")
        plt.plot('x', 'y2', data=dn, marker='', color="green", linewidth=2, label="Mean")
        plt.plot('x', 'y3', data=dn, marker='', color="blue", linewidth=2, label="Min")
        plt.xlim(min(vetOrdenado)*0.90, max(vetOrdenado)*1.10)
        plt.ylim(0.0, 0.80)

        # Add titles
        plt.title(path + ' NDCG - Progresso por Geracao N:' + str(ntrees) + '  Fold:' + str(fold))
        plt.xlabel("Geracao")
        plt.ylabel("NDCG")

        plt.show()

    if path.count("trisk"):

        plt.style.use('seaborn-darkgrid')

        plt.plot('x', 'y1', data=dt, marker='', color="red", linewidth=2,label="Max")
        plt.plot('x', 'y2', data=dt, marker='', color="green", linewidth=2, label="Mean")
        plt.plot('x', 'y3', data=dt, marker='', color="blue", linewidth=2, label="Min")
        plt.xlim(min(vetOrdenado)*0.90, max(vetOrdenado)*1.10)

        if min(TvetVMin) < min(TvetVMean):
            tt = min(TvetVMin)
        else:
            tt = min(TvetVMean)
        plt.ylim(tt*1.10, max(TvetVMax)*1.10)

        # Add titles
        plt.title(path + 'TRISK - Progresso por Geracao N:' + str(ntrees) + '  Fold:' + str(fold))
        plt.xlabel("Geracao")
        plt.ylabel("TRISK")

        plt.show()

    if path.count("spea2"):
        # style
        plt.style.use('seaborn-darkgrid')

        plt.plot('x', 'y', data=ds, marker='', color="red", linewidth=2,label="")
        plt.xlim(min(Nvet)*0.90, max(Nvet)*1.10)
        plt.ylim(min(Tvet)*0.90, max(Tvet)*1.10)

        # Add titles
        plt.title(path + ' SPEA2 - Progresso por Individuo:' + str(ntrees) + '  Fold:' + str(fold))
        plt.xlabel("NDCG")
        plt.ylabel("TRISK")

        plt.show()

    for cont in indices:
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

    arquivo.close()

def geradorRelatorioFinal(path, ntrees):

    list = files_path04(textFinal + path + '/' + str(ntrees) + '/')

    vetor_folds = []
    vetor_ndcg = []

    vetor_tempo = []

    for l in list:

        if l.count('TheBests') == 1:
            with open(l, 'r') as ficheiro:
                reader = csv.reader(ficheiro)

                vetor_folds.append(l.replace(textFinal + path + '/' + str(ntrees) + '/', '', 1))

                for linha in reader:
                    if linha[0] != 'Geracao' and linha[0] != '':
                        vetor_ndcg.append(linha[1])
                        vetor_tempo.append(linha[len(linha) - 1])

        if l.count('Original') == 1:
            with open(l, 'r') as ficheiro:
                reader = csv.reader(ficheiro)

                vetor_folds.append(l.replace(textFinal + path + '/' + str(ntrees) + '/', ''))

                for linha in reader:
                    if linha[0] != 'Geracao' and linha[0] != '':
                        vetor_ndcg.append(linha[1])
                        vetor_tempo.append(linha[len(linha) - 1])



    name_arq = textFinal + path + '/' + str(ntrees) + '/relatorio_final' + '.csv'

    arquivo = open(name_arq, 'w')
    arquivo.write("Fold,NDCG Test, Tempo do Otimizado\n")

    indices = sortGetIndice(vetor_folds, crescente=1)

    for g in indices:
        arquivo.write("{0},{1},{2}\n".format(str(vetor_folds[g]),
                                                 str(vetor_ndcg[g]),
                                                 str(vetor_tempo[g])))
    arquivo.close()

def imprimir_individuo(colecao, vetor_individuo, num_arvores, total_geracao, fold, ga, time=0):

    # colecao = colecao.replace("../Colecoes/", "")
    if colecao.count("2003_td_dataset"):
        temp = "2003_td_dataset"
    elif colecao.count("2004_td_dataset"):
        temp = "2004_td_dataset"
    elif colecao.count("web10k"):
        temp = "web10k"

    if colecao.count("ndcg"):
        temp2 = "ndcg"
    elif colecao.count("trisk"):
        temp2 = "trisk"
    elif colecao.count("spea2"):
        temp2 = "spea2"

    nColecao = temp + "/" + temp2

    path = textFinal + str(nColecao) + '/'+ str(num_arvores) + "/Fold" + str(fold)
    # print(path)

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    if len(vetor_individuo) == 1 and np.array(vetor_individuo[0].mask).tolist().count(1) == num_arvores:
        arquivo_name = textFinal + nColecao + '/'+ str(num_arvores) + '/Original_' + str(fold) + '.csv'
    elif len(vetor_individuo) == 1:
        arquivo_name = textFinal + nColecao + '/'+ str(num_arvores) + '/TheBests_' + str(fold)+ '.csv'
    elif len(vetor_individuo) == 1 and vetor_individuo[0].mask.cont(1) != num_arvores:
        arquivo_name = textFinal + nColecao + '/'+ str(num_arvores)+ '/Compare_' + str(fold) + '.csv'
    else:
        arquivo_name = textFinal + nColecao + '/'+ str(num_arvores)+ '/Fold' + str(fold) + '/Geracao_' + str(total_geracao) + '.csv'


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

    classificacao = []
    for x in vetor_individuo:
        classificacao.append(x.getScore("ndcg")[0])

    indices = sortGetIndice(classificacao, crescente=1)

    for i in indices:
        individuo = vetor_individuo[i]
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

def gerarRelatorioApresentacao(colecao, arvores):

    ll = 0

    for ntrees in arvores:
        vetorOriginal = [0,0,0,0,0]
        vetorMelhores = []

        for teste in ["ndcg", "trisk", "spea2"]:
            list = files_path04(textFinal + colecao + '/' + teste + '/'+ str(ntrees) + '/')

            vetorMelhoresCada = [0,0,0,0,0]

            for l in list:
                if l.count('relatorio_final') == 1:
                    with open(l, 'r') as ficheiro:
                        reader = csv.reader(ficheiro)

                        for linha in reader:
                            if str(linha[0]).count('Original'):
                                temp1 = str(linha[0]).replace('Original_', '')[:-4]
                                if vetorOriginal[int(temp1) - 1] == 0:
                                    vetorOriginal[int(temp1) - 1] = float(linha[1])

                            if str(linha[0]).count('TheBests'):
                                temp1 = str(linha[0]).replace('TheBests_', '')[:-4]
                                vetorMelhoresCada[int(temp1) - 1] = float(linha[1])

            vetorMelhores.append(vetorMelhoresCada)

        if len(arvores) == 1:
            name_arq = textFinal + colecao + '/ResultadosFinais_' + str(ntrees) + '.csv'
            arquivo = open(name_arq, 'w')

        else:
            name_arq = textFinal + colecao + '/ResultadosFinais_.csv'
            arquivo = open(name_arq, 'a')


        arquivo.write(colecao+",," + str(ntrees) + "\n\n")
        arquivo.write("Folds,,SCORE ORIGINAL\n\n")

        cont = 0
        for cont in range(1,6):
            arquivo.write(str(cont) + ",," + str(vetorOriginal[cont - 1]) + "\n")

        arquivo.write("\nFolds,,NDCG,,,TRISK,,,SPEA2\n\n")

        bb = 13 + ll
        oo = 5  + ll

        cont = 0
        for cont in range(1,6):
            ifNDCG =  "=IF(((C" + str(bb + cont-1) + "*100)/C" + str(oo + cont-1) + ")>100;1;0)"
            ifTRISK = "=IF(((F" + str(bb + cont-1) + "*100)/C" + str(oo + cont-1) + ")>100;1;0)"
            ifSPEA2 = "=IF(((I" + str(bb + cont-1) + "*100)/C" + str(oo + cont-1) + ")>100;1;0)"

            arquivo.write("{0},,{1},{2},,{3},{4},,{5},{6}\n".format(str(cont),
                                                 str(vetorMelhores[0][cont-1]),
                                                 ifNDCG,
                                                 str(vetorMelhores[1][cont-1]),
                                                 ifTRISK,
                                                 str(vetorMelhores[2][cont-1]),
                                                 ifSPEA2))

        arquivo.write("\n\n")
        ll += 19
    arquivo.close()

def sortGetIndice(listaOrdernar, crescente=1):
    indices = []

    for i in range(len(listaOrdernar)):
        indices.append(i)

    for c1 in range(len(listaOrdernar)):
        for c2 in range(len(listaOrdernar)):

            if crescente and listaOrdernar[indices[c1]] > listaOrdernar[indices[c2]]:
                temp = indices[c2]
                indices[c2] = indices[c1]
                indices[c1] = temp
            elif not(crescente) and listaOrdernar[indices[c1]] < listaOrdernar[indices[c2]]:
                temp = indices[c2]
                indices[c2] = indices[c1]
                indices[c1] = temp

    return indices

def storeTrees(colecao, AllTrees, size, fold):

    try:
        os.makedirs(colecao.replace("../Colecoes", "./TreesRF"))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    fileCache = colecao.replace("../Colecoes", "./TreesRF") + "/AllTrees" + str(size) + "Fold" + str(fold)

    divisor = size
    if colecao.count("web10k"):
        divisor = 50
    elif colecao.count("web10k"):
        divisor = 10
    elif colecao.count("2003"):
        divisor = size


    count = int(size / divisor)

    for ptr in range(count):
        x = slice(ptr*divisor, divisor + (divisor*ptr))

        trees = AllTrees[x]

        if divisor != 1:
            addPTR = "ptr" + str(ptr)
        else:
            addPTR = ""

        with open(fileCache + addPTR + ".pickle", 'wb') as pFile:
            pickle.dump(trees, pFile)

def chargeToPredict(colecao, mask, size, fold, boolStatus=False):

    fileCache = colecao.replace("../Colecoes", "./TreesRF") + "/AllTrees" + str(size) + "Fold" + str(fold)

    divisor = 0
    if colecao.count("web10k"):
        divisor = 50
    elif colecao.count("web10k"):
        divisor = 10
    elif colecao.count("2003"):
        divisor = size

    count = int(size / divisor)

    treesAll = []

    for ptr in range(count):
        x = slice(ptr * divisor, divisor + (divisor * ptr))

        if divisor != 1:
            addPTR = "ptr" + str(ptr)
        else:
            addPTR = ""

        if not os.path.isfile(fileCache + addPTR + ".pickle"):
            return False

        if os.path.isfile(fileCache + addPTR + ".pickle"):
            if boolStatus:
                return True
            with open(fileCache + addPTR + ".pickle", 'rb') as handle:
                treesAll.extend(pickle.load(handle))

    trees = []

    cont = 0
    for gene in mask:
        if gene == 1:
            trees.append(treesAll[cont])
        cont += 1

    return trees
