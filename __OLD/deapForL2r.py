# %%
import random
from copy import deepcopy
import os
from deap import creator, base, tools, algorithms
from evaluateIndividuoSerial import getEval, getWeights
from l2rCodesSerial import load_L2R_file
import json
import time
import numpy as np
import datetime as dt
from ScikitLearnModificado.forest import Forest
import controlTime as ct
import readSintetic
import pickle
from GAUtils import oob_synthetic

# %%
SINTETIC = False
sparse = False
seed = 1313
ensemble = 1  # for regression forest
baseline_algorithm = 'reg'  # for baseline
METHOD = 'spea2'

# %%
random.seed(seed)

# %%


def main(input_options):

    # Load dataset files
    seed = input_options['generalOptions']['seed']

    random.seed(seed)

    dataset = input_options['datasetOptions']['datasetName']
    fold = str(input_options['datasetOptions']['fold'])

    n_trees = input_options['randomForestOptions']['numberOfTrees']
    oob_predict = input_options['randomForestOptions']['oobPredict']

    fitness_metrics = input_options['geneticAlgorithmOptions']['fitnessMetrics']
    population_size = input_options['geneticAlgorithmOptions']['populationNumber']
    number_of_generations = input_options['geneticAlgorithmOptions']['generationNumber']
    crossover_prob = input_options['geneticAlgorithmOptions']['crossoverProbability']
    chromosome_mutation_prob = input_options['geneticAlgorithmOptions']['chromosomeMutationProbability']
    gene_mutation_prob = input_options['geneticAlgorithmOptions']['geneMutationProbability']
    tournament_size = input_options['geneticAlgorithmOptions']['tournamentSize']

    output_folder = input_options['outputOptions']['shortExperimentIdentifier']

    X_train, y_train, query_id_train = load_L2R_file(
        './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'train' + '.txt', sparse)
    # X_test, y_test, query_id_test = load_L2R_file(
    #    './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'test' + '.txt', sparse)
    X_vali, y_vali, query_id_vali = load_L2R_file(
        './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'vali' + '.txt', sparse)

    forest_path = os.getcwd() + '/output/forests/'
    if not os.path.exists(forest_path + f'{dataset}{seed}{n_trees}/'):
        os.mkdir(forest_path + f'{dataset}{seed}{n_trees}/')

    forest_path += f'{dataset}{seed}{n_trees}/'

    if not os.path.exists(forest_path + f'Fold{fold}.pkl') or not input_options['generalOptions']['persistForest']:
        model = Forest(n_estimators=n_trees, max_features=0.3, max_leaf_nodes=100, min_samples_leaf=1,
                       random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)

        if input_options['generalOptions']['persistForest']:
            with open(forest_path + f'Fold{fold}.pkl', 'wb') as forest:
                pickle.dump(model, forest)
                forest.close()
    else:
        with open(forest_path + f'Fold{fold}.pkl', 'rb') as forest:
            model = pickle.load(forest)
            forest.close()

    model.estimators_ = np.array(model.estimators_)

    # if oob_predict:
    #    model.oob_predict_buffer(X_train, y_train)

    base_collection_name = f'./output/{output_folder}/Fold{fold}/chromosomeCollection'
    base_collection = {}

    try:
        with open(base_collection_name + '.json', 'r') as fp:
            base_collection = json.load(fp)

        for item in base_collection:
            for att in base_collection[item]:
                try:
                    if len(base_collection[item][att]) > 1:
                        base_collection[item][att] = np.array(
                            base_collection[item][att])
                except:
                    pass

        print('A base tem ' + str(len(base_collection)) + ' indivíduos!\n')

    except:

        print('Primeira vez executando ...')

    current_generation_s = 1
    archive_record = {}

    def evalIndividuo(individual):

        evaluation = []
        individuo_ga = ''
        for i in range(n_trees):
            individuo_ga += str(individual[i])
        if '1' not in individuo_ga:
            if 'NDCG' in fitness_metrics:
                evaluation.append(0)
            if 'GeoRisk' in fitness_metrics:
                evaluation.append(0)
            if 'TRisk' in fitness_metrics:
                evaluation.append(0)
        elif individuo_ga in base_collection:
            if 'NDCG' in fitness_metrics:
                evaluation.append(base_collection[individuo_ga]['NDCG'])
            if 'GeoRisk' in fitness_metrics:
                evaluation.append(base_collection[individuo_ga]['GeoRisk'])
            if 'TRisk' in fitness_metrics:
                evaluation.append(base_collection[individuo_ga]['TRisk'])
        else:
            if oob_predict:
                result = getEval(individuo_ga, model, n_trees, X_train, y_train,
                                 query_id_train,
                                 ensemble, n_trees, seed, dataset, fitness_metrics, fold, baseline_algorithm, oob_predict)
            else:
                result = getEval(individuo_ga, model, n_trees, X_vali, y_vali,
                                 query_id_vali,
                                 ensemble, n_trees, seed, dataset, fitness_metrics, fold, baseline_algorithm, oob_predict)
            base_collection[individuo_ga] = {}
            base_collection[individuo_ga]['NDCG'] = result[0]
            base_collection[individuo_ga]['GeoRisk'] = result[1]
            #base_collection[individuo_ga]['TRisk'] = result[2]
            base_collection[individuo_ga]['geracao_s'] = current_generation_s

            if 'NDCG' in fitness_metrics:
                evaluation.append(result[0])
            if 'GeoRisk' in fitness_metrics:
                evaluation.append(result[1])
            if 'TRisk' in fitness_metrics:
                evaluation.append(result[2])

        return evaluation
    random.seed(seed)
    creator.create("MyFitness", base.Fitness,
                   weights=getWeights(fitness_metrics))
    creator.create("Individual", list, fitness=creator.MyFitness)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, n=n_trees)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalIndividuo)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=gene_mutation_prob)
    toolbox.register("selectTournament", tools.selTournament,
                     tournsize=tournament_size)
    paretoFront = tools.ParetoFront()

    toolbox.register("select", tools.selSPEA2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("mean", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("var", np.var, axis=0)

    logbook = tools.Logbook()
    # logbook.header = "gen", "evals", "std", "min", "avg", "max"
    logbook.header = "gen", "min", "max", "mean", "std"

    population = toolbox.population(n=population_size)
    if SINTETIC:
        list_individuos = readSintetic.get(dataset, fold, population_size)
        for indice_individuo in range(population_size):
            temp_ind = list_individuos[indice_individuo]
            for indice_gene in range(n_trees):
                population[indice_individuo][indice_gene] = temp_ind[indice_gene]

    archive = []

    for gen in range(number_of_generations):

        fits = toolbox.map(toolbox.evaluate, population)
        fitsA = toolbox.map(toolbox.evaluate, archive)

        for fit, ind in zip(fits, population):
            ind.fitness.values = fit
        for fit, ind in zip(fitsA, archive):
            ind.fitness.values = fit

        archive = toolbox.select(population + archive, k=population_size)

        mating_pool = toolbox.selectTournament(archive, k=population_size)
        offspring_pool = map(toolbox.clone, mating_pool)

        offspring_pool = algorithms.varAnd(
            offspring_pool, toolbox, cxpb=crossover_prob, mutpb=chromosome_mutation_prob)

        if len(fitness_metrics) > 1:
            paretoFront.update(population)

        population = offspring_pool

        print(len(paretoFront))

        if gen % 5 == 0:
            TEMP_COLECAO_BASE = deepcopy(base_collection)
            for item in TEMP_COLECAO_BASE:
                for att in TEMP_COLECAO_BASE[item]:
                    try:
                        if len(TEMP_COLECAO_BASE[item][att]) > 1:
                            TEMP_COLECAO_BASE[item][att] = TEMP_COLECAO_BASE[item][att].tolist(
                            )
                    except:
                        pass
            with open(base_collection_name + '.json', 'w+') as fp:
                json.dump(TEMP_COLECAO_BASE, fp, indent=4)

        record = stats.compile(archive)
        archive_record[f'{current_generation_s}'] = [
            ''.join(map(str, chromosome)) for chromosome in archive]
        if len(fitness_metrics) > 1:
            paretoFront.update(archive)
        current_generation_s += 1
        logbook.record(gen=gen, **record)

        print(logbook.stream)

    log_json = {}
    for i in range(len(logbook)):
        log_json[i] = {}
        log_json[i]['gen'] = logbook[i]['gen']
        log_json[i]['min'] = logbook[i]['min'].tolist()
        log_json[i]['max'] = logbook[i]['max'].tolist()
        log_json[i]['mean'] = logbook[i]['mean'].tolist()
        log_json[i]['std'] = logbook[i]['std'].tolist()
        log_json[i]['var'] = logbook[i]['var'].tolist()

    with open(base_collection_name + '-log.json', 'w') as fp:
        json.dump(log_json, fp, indent=4)

    TEMP_COLECAO_BASE = base_collection.copy()
    for item in TEMP_COLECAO_BASE:
        for att in TEMP_COLECAO_BASE[item]:
            try:
                if len(TEMP_COLECAO_BASE[item][att]) > 1:
                    TEMP_COLECAO_BASE[item][att] = TEMP_COLECAO_BASE[item][att].tolist(
                    )
            except:
                pass

    with open(base_collection_name + '.json', 'w') as fp:
        json.dump(TEMP_COLECAO_BASE, fp, indent=4)

    top = {}
    for j in range(0, len(fitness_metrics)):
        for i in range(0, len(archive)):
            if i == 0:
                bigger = archive[i].fitness.values[j]
                bigger_index = i
            else:
                if bigger < archive[i].fitness.values[j]:
                    bigger = archive[i].fitness.values[j]
                    bigger_index = i
                else:
                    pass
        top[fitness_metrics[j]] = {
            'ind': archive[bigger_index], 'score': bigger}

    with open(base_collection_name + 'Best.json', 'w') as fp:
        json.dump(top, fp, indent=4)

    with open(base_collection_name + 'Archive.json', 'w') as fp:
        json.dump(archive_record, fp, indent=4)

    if len(fitness_metrics) > 1:
        pareto_front_ = []
        for exceptional in list(paretoFront):
            exc = ''
            for gene in exceptional:
                exc += str(gene)
            pareto_front_.append(exc)
        pareto_front_ = pareto_front_

        with open(base_collection_name + 'ParetoFront.json', 'w') as fp:
            json.dump(pareto_front_, fp, indent=4)
    '''
    if METHOD == 'nsga2':
        return population
    elif METHOD == 'spea2':
        return archive
    '''
