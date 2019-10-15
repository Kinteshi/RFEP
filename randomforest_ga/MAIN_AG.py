# coding=utf-8
import time

from sklearn.ensemble import RandomForestRegressor

from AG import AG
from External import load_L2R_file, ChangeName, nFeaturesColection, \
    modelEvaluation, getTRisk, imprimir_individuo, Individual

from External.ga import sortGetIndice, chargeToPredict, storeTrees, geradorRelatorioFinal, geradorRelatorioValidacao


def Executar(colecao, fold, metrica, numTrees, original_geracoes_comparar, superMask=[]):
    # Manipular a DataBase
    name_train = colecao + "/Fold" + str(fold) + "/" + "Norm.train.txt"
    name_vali = colecao + "/Fold" + str(fold) + "/" + "Norm.vali.txt"
    name_test = colecao + "/Fold" + str(fold) + "/" + "Norm.test.txt"

    nFeatures = nFeaturesColection(colecao)
    MASK = [1] * nFeatures
    sizeGen = 75
    '''
    Instacia as Classes
    Algoritmo Genetico e o RandomForest Modificado
    '''
    forest = RandomForestRegressor(
            n_estimators=numTrees, n_jobs=-1, oob_score=True)
    #numGeneration = original_geracoes_comparar
    tempGERACAO = original_geracoes_comparar

    fileCache = colecao.replace("../Colecoes", "./TreesRF") + "/AllTrees" + str(numTrees) + "Fold" + str(
        fold) + ".pickle"

    trees = []
    # Verifica se ja possui arvores criadas
    #  Se não as cria
    if not(chargeToPredict(colecao, [1]*numTrees, numTrees, fold, boolStatus=True)):

        X, y, z = load_L2R_file(name_train, MASK)
        
        forest.fit(X, y)

        trees = forest.estimators_
        storeTrees(colecao, trees, numTrees, fold)

    for numGeneration in tempGERACAO:
        original_geracoes_comparar = numGeneration

        if original_geracoes_comparar > 1:
            # print("Metrica de OOB")
            X2, y2, z2 = load_L2R_file(name_vali, MASK)
            Vetores_Vali = ChangeName(X2, y2, z2)

            ag = AG(forest=forest, mutacao=0.3, crossover=0.7, tipoSelecao="torn", tipoCrossover="pontual", elitismo=1,
                    tamPopulacao=sizeGen, tamIndividuo=numTrees, fitnes=metrica, geracoes=numGeneration,
                    dataBase=Vetores_Vali, fileCache=colecao, numthreads=1, fold=fold)

            populacaoFinal = ag.RUN()

        #     Test Final
        X3, y3, z3 = load_L2R_file(name_test, MASK)
        Vetores_Test = ChangeName(X3, y3, z3)

        if original_geracoes_comparar > 1:

            scoresTEMP = []
            for p in populacaoFinal:
                scoresTEMP.append(p.getScore(metrica)[0])

            if metrica == "spea2":
                crescente = 0
            else:
                crescente = 1

            indices = sortGetIndice(scoresTEMP, crescente=crescente)

            The_Best = populacaoFinal[indices[-1]]

            forest.estimators_ = chargeToPredict(colecao, [1] * numTrees,
                                                 numTrees, fold)
            forest.n_estimators = len(forest.estimators_)
            forest.n_outputs_ = 1

            scoresBaseTest = forest.predict(Vetores_Test.x)
            base_ndgc, _ = modelEvaluation(
                Vetores_Test, scoresBaseTest, nFeatures)

            time_thebest = time.time()

            forest.estimators_ = chargeToPredict(colecao, The_Best.mask,
                                                 numTrees, fold)
            forest.n_outputs_ = 1

            scoresTest = forest.predict(Vetores_Test.x)
            scoreNDCG, _ = modelEvaluation(Vetores_Test, scoresTest, nFeatures)

            The_Best.setScore(scoreNDCG, "ndcg")
            trisk, vectrisk = getTRisk(scoreNDCG, base_ndgc, 5)
            The_Best.setScore(trisk, "trisk", vectrisk)

            time_final_thebest = time.time()

            imprimir_individuo(colecao + '/' + metrica, [The_Best], numTrees, numGeneration, fold,
                               ["torn", "pontual", 1],
                               time_final_thebest - time_thebest)

        elif original_geracoes_comparar == 1:

            time_original = time.time()

            origin = Individual([1] * numTrees, 1)

            forest.estimators_ = chargeToPredict(colecao, [1] * numTrees,
                                                 numTrees, fold)
            forest.n_outputs_ = 1

            scoreOriginal = forest.predict(X3)

            scoreNDCG, _ = modelEvaluation(
                Vetores_Test, scoreOriginal, nFeatures)

            origin.setScore(scoreNDCG, "ndcg")

            time_final_original = time.time()

            imprimir_individuo(colecao + '/' + metrica, [origin], numTrees, 1, fold,
                               ["torn", "pontual", 1],
                               time_final_original - time_original)

        else:

            time_original = time.time()

            origin = Individual(superMask, 1)

            forest.estimators_ = chargeToPredict(colecao, superMask,
                                                 numTrees, fold)
            forest.n_outputs_ = 1

            scoreOriginal = forest.predict(X3)

            scoreNDCG, _ = modelEvaluation(
                Vetores_Test, scoreOriginal, nFeatures)

            origin.setScore(scoreNDCG, "ndcg")

            time_final_original = time.time()

            imprimir_individuo(colecao + '/' + metrica, [origin], numTrees, 1, fold,
                               ["torn", "pontual", 1],
                               time_final_original - time_original)

    return 1


if __name__ == "__main__":
    # for metrica in ["ndcg", "trisk"]:##,"trisk", "spea2"]:##, "trisk","ndcg"]:
    #     for numTrees in [100,300,500,1000]:##,500,750,1000]:
    #         for fold in range(1, 6):
    #             Executar("../Colecoes/2003_td_dataset", fold, metrica, numTrees, [1, 30])
            #     geradorRelatorioValidacao("2003_td_dataset/" + metrica, numTrees, fold)
            # geradorRelatorioFinal("2003_td_dataset/" + metrica, numTrees)

        # for metrica in ["ndcg"]:##,"trisk", "spea2"]:##, "trisk","ndcg"]:
        # for numTrees in [100, 500, 1000]:##,500,750,1000]:
        #     for fold in range(1, 6):
        #         # for geracoes in [1]:
        #         Executar("../Colecoes/2003_td_dataset", fold, metrica, numTrees, [1])
        #         geradorRelatorioValidacao("2003_td_dataset/" + metrica, numTrees, fold)
        #     geradorRelatorioFinal("2003_td_dataset/" + metrica, numTrees)

    # for metrica in ["trisk"]:  ##,"trisk", "spea2"]:##, "trisk","ndcg"]:
    #     for numTrees in [500, 750]:  ##,500,750,1000]:
    #         for fold in range(1, 6):
    #             # for geracoes in [1, 30]:
    #Executar("../Colecoes/2003_td_dataset", 1, "ndcg", 100, [0], [1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,1,1,1,1,1,1])
    #             geradorRelatorioValidacao("2003_td_dataset/" + metrica, numTrees, fold)
    #         geradorRelatorioFinal("2003_td_dataset/" + metrica, numTrees)
    #

    # Executar("../Colecoes/web10k", 1, "trisk", 250, [30])

    dt = '2003_td_dataset'

    # ,"trisk", "spea2"]:##, "trisk","ndcg"]:
    for metrica in ["spea2"]:
        for numTrees in [10]:  # ,500,750,1000]:
            for fold in range(1, 2):
                # for geracoes in [1,30]:
                Executar("C:/Users/jefma/Documents/GitHub/PIBIC/Colecoes/" + dt,
                         fold, metrica, numTrees, [1, 5])
                '''
                geradorRelatorioValidacao(
                    dt + '/' + metrica, numTrees, fold)
                
                geradorRelatorioFinal(dt + '/' + metrica, numTrees)
                '''
            