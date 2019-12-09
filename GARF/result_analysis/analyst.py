# %%
from ScikitLearnModificado import Forest
from l2rCodesSerial import load_L2R_file
import numpy as np
import json
from evaluateIndividuoSerial import getEval
import os
import pickle



# %%
def generate_report(path, identifier, fold):
    with open(path + f'{identifier}/Fold{fold}/config.json') as config_file:
        config = json.load(config_file)
        config_file.close()

    n_genes = n_trees = config['randomForestOptions']['numberOfTrees']
    dataset = config['randomForestOptions']['datasetName']
    sparse = False
    seed = config['generalOptions']['seed']

    if config['generalOptions']['persistForest']:
        if not os.path.exists(path + 'forests/' + f'Fold{fold}.pkl') or not config['generalOptions']['persistForest']:
            X_train, y_train, query_id_train = load_L2R_file(
                './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'train' + '.txt', '1' * n_genes, sparse)
            model = Forest(n_estimators=n_trees, max_features=0.3, max_leaf_nodes=100, min_samples_leaf=1,
                           random_state=seed, n_jobs=-1)
            model.fit(X_train, y_train)

            if config['generalOptions']['persistForest']:
                with open(path + 'forests/' + f'Fold{fold}.pkl', 'wb') as forest:
                    pickle.dump(model, forest)
                    forest.close()
        else:
            with open(path + 'forests/' + f'Fold{fold}.pkl', 'rb') as forest:
                model = pickle.load(forest)
                forest.close()
    else:
        X_train, y_train, query_id_train = load_L2R_file(
            './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'train' + '.txt', '1' * n_genes, sparse)
        model = Forest(n_estimators=n_genes, max_features=0.3, max_leaf_nodes=100, min_samples_leaf=1,
                       random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)

    model.estimators_ = np.array(model.estimators_)

    X_test, y_test, query_id_test = load_L2R_file(
        './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'test' + '.txt', '1' * n_genes, sparse)
    X_vali, y_vali, query_id_vali = load_L2R_file(
        './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'vali' + '.txt', '1' * n_genes, sparse)

    with open(path + f'{identifier}/Fold{fold}/chromosomeCollectionBest.json') as file:
        best = json.load(file)
        file.close()

    best_ind = ''

    for g in best['precision']['ind']:
        best_ind += str(g)

    original_ind = '1' * n_genes

    results = {}

    results['validationSet']['original'] = getEval(original_ind, model, n_genes, X_vali, y_vali,
                                  query_id_vali, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg')
    results['testSet']['original'] = getEval(original_ind, model, n_genes, X_test, y_test,
                            query_id_test, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg')

    results['testSet']['bestModel'] = getEval(best_ind, model, n_genes, X_test, y_test,
                      query_id_test, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg')
    results['validationSet']['bestModel'] = getEval(best_ind, model, n_genes, X_vali, y_vali,
                            query_id_vali, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg')

    with open(path + f'{identifier}/Fold{fold}/resultReport.json') as result_file:
        json.dump(results, result_file)
        result_file.close()
