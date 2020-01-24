import l2rCodesSerial
import numpy as np
from sklearn import model_selection


def getEval(individuo, model, NUM_GENES, X, y, query_id_train, ENSEMBLE, NTREES, SEED,
            DATASET, METRIC, NUM_FOLD, ALGORITHM, oob_predict):
    evaluation = []
    ndcg, queries = getPrecisionAndQueries(individuo, model, NUM_GENES, X, y, query_id_train,
                                           ENSEMBLE, NTREES, SEED, DATASET,
                                           METRIC, oob_predict)

    evaluation.append(queries)
    # evaluation.append(ndcg)
    evaluation.append(np.round(getRisk(queries, DATASET, NUM_FOLD, ALGORITHM), 5))
    evaluation.append(getTotalFeature(individuo))
    #evaluation.append(getTRisk(queries, DATASET, NUM_FOLD, ALGORITHM))

    return evaluation


def getWeights(params):
    weights = []
    if 'NDCG' in params:
        weights.append(1)
    if 'GeoRisk' in params:
        weights.append(1)
    if 'feature' in params:
        weights.append(-1)
    if 'TRisk' in params:
        weights.append(-1)

    return weights


def getPrecision(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED,
                 DATASET,
                 METRIC):
    ndcg, queries = getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train,
                                           ENSEMBLE, NTREES, SEED, DATASET,
                                           METRIC)
    return ndcg


def getTotalFeature(individuo):
    return sum([int(i) for i in individuo])


# PRECISA SER CORRIGIDA SE HOUVER MAIS DE UM BASEINE
def getRisk(queries, DATASET, NUM_FOLD, ALGORITHM):
    base = []

    arq = open(r'./baselines/' + DATASET + '/Fold' +
               NUM_FOLD + '/' + ALGORITHM + '.txt')
    for line in arq:
        base.append([float(line.split()[0])])
    basey = base.copy()

    for k in range(len(basey)):
        basey[k].append(queries[k])

    r = (l2rCodesSerial.getGeoRisk(np.array(basey), 5))[1]
    return r


def getTRisk(queries, DATASET, NUM_FOLD, ALGORITHM):
    base = []

    arq = open(r'./baselines/' + DATASET + '/Fold' +
               NUM_FOLD + '/' + ALGORITHM + '.txt')
    for line in arq:
        base.append(float(line.split()[0]))

    r, vetorRisk = (l2rCodesSerial.getTRisk(queries, base, 5))
    return vetorRisk


def getPrecisionAndQueries(individuo, model, NUM_GENES, X, y, query_id, ENSEMBLE, NTREES,
                           SEED, DATASET,
                           METRIC, oob_predict):

    list_mask = list(individuo)

    #queriesList = l2rCodesSerial.getQueries(query_id_train)
    if oob_predict:
        resScore = model.oob_predict(X, y, list_mask)
    else:
        resScore = model.predict(X, list_mask)

    ndcg, queries = l2rCodesSerial.getEvaluation(
        resScore, query_id, y, DATASET, METRIC, "test")
    return ndcg, queries
