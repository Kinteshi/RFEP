import l2rCodes
import numpy as np
from sklearn import model_selection
import cudf

def getEval(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED,
            DATASET, METRIC, NUM_FOLD, ALGORITHM):
    evaluation = []
    ndcg, queries = getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train,
                                           ENSEMBLE, NTREES, SEED, DATASET,
                                           METRIC)

    # evaluation.append(ndcg)
    evaluation.append(queries)
    evaluation.append(getRisk(queries, DATASET, NUM_FOLD, ALGORITHM))
    evaluation.append(getTotalFeature(individuo))

    return evaluation


def getWeights(params):
    weights = []
    if 'precision' in params:
        weights.append(1)
    if 'risk' in params:
        weights.append(1)
    if 'feature' in params:
        weights.append(-1)

    return weights


def getPrecision(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED, DATASET,
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

    arq = open(r'./baselines/' + DATASET + '/Fold' + NUM_FOLD + '/' + ALGORITHM + '.txt')
    for line in arq:
        base.append([float(line.split()[0])])
    basey = base.copy()

    for k in range(len(basey)):
        basey[k].append(queries[k])

    r = (l2rCodes.getGeoRisk(np.array(basey), 0.1))[1]
    return r


def getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES,
                           SEED, DATASET,
                           METRIC):
    list_mask = list(individuo)
    features = []
    for i in range(NUM_GENES):
        if list_mask[i] == '1':
            features.append(i)
    X_train_ind = X_train[:, features]
    # X_test_ind = X_test[:, features]

    queriesList = l2rCodes.getQueries(query_id_train)
    scoreTrain = [0] * len(y_train)

    kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=SEED)
    XF_train_index = []
    XF_test_index = []

    for qtrain_index, qtest_index in kf.split(queriesList):

        del XF_train_index[:]
        del XF_test_index[:]

        for qtr in qtrain_index:
            XF_train_index = XF_train_index + queriesList[qtr]
        for qts in qtest_index:
            XF_test_index = XF_test_index + queriesList[qts]

        XF_train, XF_test = X_train_ind[XF_train_index], X_train_ind[XF_test_index]####
        yf_train, yf_test = y_train[XF_train_index], y_train[XF_test_index]
        # qf_train, qf_test = query_id_train[XF_train_index], query_id_train[XF_test_index]

        # resScore = None
        model = l2rCodes.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, DATASET)
        ##
        XF_train = cudf.DataFrame.from_records(XF_train)
        XF_test = cudf.DataFrame.from_records(XF_test)
        yf_train = cudf.Series(yf_train)
        ##
        model.fit(XF_train, yf_train)
        resScore = model.predict(XF_test)

        # sortRespRel=[rel for (score, rel) in sorted(zip(resScore,yf_test), key=lambda pair: pair[0], reverse=True)]
        c = 0
        for i in XF_test_index:
            scoreTrain[i] = resScore[c]
            c = c + 1
    ndcg, queries = l2rCodes.getEvaluation(scoreTrain, query_id_train, y_train, DATASET, METRIC, "test")
    return ndcg, queries



    # X_train_ind = X_train.iloc[:, features]
    # X_test_ind = X_test.iloc[:, features]


