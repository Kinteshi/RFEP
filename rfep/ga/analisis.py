from pathlib import Path
from .misc import DictPersist, ModelPersist
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Analyst():

    def __init__(self, mp, dp, ev, baselines_path, n_trees, seed):

        self.__dict_persist = dp
        self.__model_persist = mp
        self.__model = None
        self.__evaluator = ev
        self.baselines_path = baselines_path
        self.__n_trees = n_trees
        self.__seed = seed

    def __load_model(self, fold):

        self.__model = self.__model_persist.load(
            f'Fold{fold}/{self.__n_trees}{self.__seed}')

    def __process_fold(self, fold):

        self.__load_model(fold)

        self.__population_bank = self.__dict_persist.load(
            f'Fold{fold}/population_bank')

        if len(self.__evaluator.metrics) == 2:
            pareto_front = self.__dict_persist.load(f'Fold{fold}/pareto_front')
            best = 0
            for i, ind in enumerate(pareto_front):
                if i == 0:
                    best = i
                else:
                    if np.mean(self.__population_bank[pareto_front[best]]['ndcg']) < np.mean(self.__population_bank[ind]['ndcg']):
                        best = i
            self.__best = pareto_front[best]
        else:
            self.__best = self.__dict_persist.load(f'Fold{fold}/best_ind')

        evaluations = self.__fold_comparison(fold)

        report = {}

        for ind, fitness in zip(['initial', 'final'], evaluations):
            report[ind] = {}
            for fit, value in zip(['ndcg', 'georisk'], fitness):
                report[ind][fit] = value.tolist()

        report['initial']['n_trees'] = len(self.__best)
        report['final']['n_trees'] = self.__best.count('1')
        report['initial']['ndcg_mean'] = np.mean(report['initial']['ndcg'])
        report['final']['ndcg_mean'] = np.mean(report['final']['ndcg'])

        self.__dict_persist.save(report, f'Fold{1}/fold_comparison')

        return evaluations

    def __fold_comparison(self, fold):

        matrix = []

        path = Path(self.baselines_path) / f'Fold{fold}'

        for file_name in path.glob('*.txt'):
            with open(file_name, 'r') as file:
                matrix.append([float(line.rstrip('\n')) for line in file])
                file.close()

        new_matrix = np.zeros((len(matrix) + 2, len(matrix[0])))

        for i in range(len(matrix)):
            new_matrix[i, :] = np.array(matrix[i])

        evaluations = self.__evaluator.evaluate_compare(
            ['1'*len(self.__best), self.__best], self.__model, new_matrix)

        return evaluations

    def __plot_evolution(self, fold):

        archive_bank = self.__dict_persist.load(f'Fold{fold}/archive_bank')
        population_bank = self.__dict_persist.load(
            f'Fold{fold}/population_bank')

        for metric in self.__evaluator.metrics:
            maximum, mean, minimum, std, var = [], [], [], [], []

            for n_gen, generation in archive_bank.items():
                if len(generation) == 0:
                    continue
                maximum.append(
                    np.max([np.mean(population_bank[ind][metric]) for ind in generation]))
                mean.append(np.mean([np.mean(population_bank[ind][metric])
                                     for ind in generation]))
                minimum.append(
                    np.min([np.mean(population_bank[ind][metric]) for ind in generation]))
                std.append(np.std([np.std(population_bank[ind][metric])
                                   for ind in generation]))
                var.append(np.var([np.mean(population_bank[ind][metric])
                                   for ind in generation]))

            gens = range(0, len(maximum))

            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot()

            ax.plot(gens, maximum, label='Max', color='blue')
            ax.plot(gens, mean, label='Mean', color='red')
            ax.plot(gens, minimum, label='Minimum', color='green')

            ax.grid(True)
            ax.legend(loc='upper left')
            ax.set_xlabel('Generations')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} by generation')

            plt.savefig(self.__dict_persist.path /
                        f'Fold{fold}/{metric}basic.png')

            ax.cla()

            ax.plot(gens, std, label='std')

            ax.grid(True)
            ax.legend(loc='upper left')
            ax.set_xlabel('Generations')
            ax.set_ylabel(f'{metric.upper()}_std')
            ax.set_title(f'{metric.upper()} std by generation')

            plt.savefig(self.__dict_persist.path /
                        f'Fold{fold}/{metric}std.png')

            ax.cla()

            ax.plot(gens, var, label='var')

            ax.grid(True)
            ax.legend(loc='upper left')
            ax.set_xlabel(f'Generations')
            ax.set_ylabel(f'{metric.upper()}_var')
            ax.set_title(f'{metric.upper()} var by generation')

            plt.savefig(self.__dict_persist.path /
                        f'Fold{fold}/{metric}var.png')

    def report(self, folds):

        comparisons = []

        for fold in folds:
            comparisons.append(self.__process_fold(fold))
            self.__plot_evolution(fold)
