# %%
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ScikitLearnModificado.forest import Forest
from evaluateIndividuoSerial import getEval
from l2rCodesSerial import load_L2R_file


def generate_report(path, identifier, fold):
    fold = str(fold)

    with open(path + f'{identifier}/Fold{fold}/config.json') as config_file:
        config = json.load(config_file)
        config_file.close()

    n_genes = n_trees = config['randomForestOptions']['numberOfTrees']
    oob_predict = False
    dataset = config['datasetOptions']['datasetName']
    sparse = False
    seed = config['generalOptions']['seed']


    if config['generalOptions']['persistForest']:
        if not os.path.exists(path + 'forests/' + f'Fold{fold}.pkl') or not config['generalOptions']['persistForest']:
            X_train, y_train, query_id_train = load_L2R_file(
                f'./dataset/{dataset}/Fold{fold}/Norm.train.txt', sparse)
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
            f'./dataset/{dataset}/Fold{fold}/Norm.train.txt', sparse)
        model = Forest(n_estimators=n_genes, max_features=0.3, max_leaf_nodes=100, min_samples_leaf=1,
                       random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)

    model.estimators_ = np.array(model.estimators_)

    X_test, y_test, query_id_test = load_L2R_file(
        f'./dataset/{dataset}/Fold{fold}/Norm.test.txt', sparse)
    X_vali, y_vali, query_id_vali = load_L2R_file(
        f'./dataset/{dataset}/Fold{fold}/Norm.vali.txt', sparse)

    with open(path + f'{identifier}/Fold{fold}/chromosomeCollectionBest.json') as file:
        best = json.load(file)
        file.close()

    best_ind = ''

    for g in best['NDCG']['ind']:
        best_ind += str(g)

    original_ind = '1' * n_genes

    results = {}
    results['original'] = {}
    results['best'] = {}

    original_validation = getEval(original_ind, model, n_genes, X_vali, y_vali,
                                  query_id_vali, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg', oob_predict)
    original_test = getEval(original_ind, model, n_genes, X_test, y_test,
                            query_id_test, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg', oob_predict)
    best_test = getEval(best_ind, model, n_genes, X_test, y_test,
                        query_id_test, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg', oob_predict)
    best_validation = getEval(best_ind, model, n_genes, X_vali, y_vali,
                              query_id_vali, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg', oob_predict)

    results['original']['validation'] = {}
    results['original']['test'] = {}
    results['best']['validation'] = {}
    results['best']['test'] = {}

    results['original']['validationNDCG'] = np.mean(original_validation[0])
    results['original']['testNDCG'] = np.mean(original_test[0])
    results['best']['validationNDCG'] = np.mean(best_validation[0])
    results['best']['testNDCG'] = np.mean(best_test[0])

    results['original']['numberOfTrees'] = n_genes
    results['best']['numberOfTrees'] = 0

    for gene in best_ind:
        if gene == '1':
            results['best']['numberOfTrees'] += 1

    results['overallStats'] = {}

    results['overallStats']['validationNDCGGain'] = ((results['best']['validationNDCG'] -
                                                      results['original']['validationNDCG']) / \
                                                     results['original']['validationNDCG']) * 100

    results['overallStats']['testNDCGGain'] = ((results['best']['testNDCG'] -
                                                results['original']['testNDCG']) / \
                                               results['original']['testNDCG']) * 100

    results['overallStats']['treesVariation'] = ((results['best']['numberOfTrees'] - results['original'][
        'numberOfTrees']) / results['original']['numberOfTrees']) * 100

    with open(path + f'{identifier}/Fold{fold}/resultReport.json', 'w') as result_file:
        json.dump(results, result_file)
        result_file.close()


def make_graphics(path, identifier, fold):
    with open(f'{path}{identifier}/Fold{fold}/chromosomeCollectionArchive.json', 'r') as file:
        archive = json.load(file)
        file.close()

    with open(f'{path}{identifier}/Fold{fold}/chromosomeCollection.json', 'r') as base:
        inds = json.load(base)
        base.close()

    with open(path + f'{identifier}/Fold{fold}/config.json') as config_file:
        config = json.load(config_file)
        config_file.close()

    # generations data
    gen_size = config['geneticAlgorithmOptions']['generationNumber']
    gen = []
    for i in range(0, gen_size):
        gen.append({})

    for ind, specs in inds.items():
        gen[specs['geracao_s'] - 1][ind] = specs

    stats_generations = {}
    stats_generations['max'] = np.zeros(gen_size)
    stats_generations['min'] = np.zeros(gen_size)
    stats_generations['mean'] = np.zeros(gen_size)
    stats_generations['var'] = np.zeros(gen_size)
    stats_generations['std'] = np.zeros(gen_size)

    for i in range(0, gen_size):
        g = gen[i]
        precisions = [np.mean(spec['NDCG']) for ind, spec in g.items()]
        stats_generations['max'][i] = np.max(precisions)
        stats_generations['min'][i] = np.min(precisions)
        stats_generations['mean'][i] = np.mean(precisions)
        stats_generations['var'][i] = np.var(precisions)
        stats_generations['std'][i] = np.std(precisions)

    # Archive data

    gen = []
    for i in range(0, gen_size):
        gen.append({})

    for i in range(1, gen_size + 1):
        gen_archive = archive[str(i)]
        for ind in gen_archive:
            key = [str(gene) for gene in ind]
            key = ''.join(key)
            gen[i - 1][key] = inds[key]

    stats_archives = {}
    stats_archives['max'] = np.zeros(gen_size)
    stats_archives['min'] = np.zeros(gen_size)
    stats_archives['mean'] = np.zeros(gen_size)
    stats_archives['var'] = np.zeros(gen_size)
    stats_archives['std'] = np.zeros(gen_size)

    for i in range(0, gen_size):
        g = gen[i]
        precisions = [np.mean(spec['NDCG']) for ind, spec in g.items()]
        stats_archives['max'][i] = np.max(precisions)
        stats_archives['min'][i] = np.min(precisions)
        stats_archives['mean'][i] = np.mean(precisions)
        stats_archives['var'][i] = np.var(precisions)
        stats_archives['std'][i] = np.std(precisions)

    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('dark')
    sns.set_context('poster')


    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot()

    legend = []
    for stat in ['max', 'mean', 'min']:
        ax.plot(stats_generations[stat])
        legend.append(stat)

    ax.set_ylabel('NDCG')
    ax.set_xlabel('Generations')
    ax.legend(legend)
    ax.set_title('NDCG by generation')
    plt.savefig(f'{path}{identifier}/Fold{fold}/basicGen.png')

    ax.cla()
    legend = []
    for stat in ['var', 'std']:
        ax.plot(stats_generations[stat])
        legend.append(stat)

    ax.set_ylabel('NDCG')
    ax.set_xlabel('Generations')
    ax.legend(legend)
    ax.set_title('NDCG by generation')
    plt.savefig(f'{path}{identifier}/Fold{fold}/miscGen.png')

    ax.cla()
    legend = []
    for stat in ['max', 'mean', 'min']:
        ax.plot(stats_archives[stat])
        legend.append(stat)

    ax.set_ylabel('NDCG')
    ax.set_xlabel('Generations')
    ax.legend(legend)
    ax.set_title('NDCG by generation\'s archive')
    plt.savefig(f'{path}{identifier}/Fold{fold}/basicArchive.png')

    ax.cla()
    legend = []
    for stat in ['var', 'std']:
        ax.plot(stats_archives[stat])
        legend.append(stat)

    ax.set_ylabel('NDCG')
    ax.set_xlabel('Generations')
    ax.legend(legend)
    ax.set_title('NDCG by generation\'s archive')
    plt.savefig(f'{path}{identifier}/Fold{fold}/miscArchive.png')
