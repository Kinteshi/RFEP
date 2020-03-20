from ..l2r.l2rCodes import getEvaluation, getGeoRisk, getQueries
import numpy as np
import multiprocessing as mp
from .misc import _chromosome_to_key
import time

class Evaluator:

    def __init__(self, metrics, weights, dataset_name, __X_dataset, __y_dataset, __queries_dataset, _oob=False, parallel=True):

        self.metrics = metrics
        self.weights = weights
        self.dataset_name = dataset_name
        self.__X_dataset = __X_dataset
        self.__y_dataset = __y_dataset
        self.__queries_dataset = __queries_dataset
        self._oob = _oob
        self.parallel = parallel
        self.__predict_method = None

        try:
            self.__validate_metrics()
        except Exception as e:
            print(e)

        self.__n_queries = getQueries(self.__queries_dataset)

    def __validate_metrics(self):

        if f'__evaluate_{self.metrics}' in dir(self):
            if isinstance(self.metrics, list) and len(self.metrics) == len(self.weights):
                for w in self.weights:
                    if isinstance(w, int) or isinstance(w, float):
                        return True
                    else:
                        raise ValueError(
                            'Weights must be a list of int or float')
            else:
                ValueError(
                    'Weights must be a list of int or float with same lenght as metrics')
        else:
            ValueError('The specified metrics cannot be evaluated')

    def set_predict_method(self, method):

        self.__predict_method = method

    def __get_scores(self, ind):

        return self.__predict_method(ind)

    def __evaluate_ndcg(self, scores):

        return getEvaluation(
            scores, self.__queries_dataset, self.__y_dataset, self.dataset_name, 'ndcg')[1]

    def __evaluate_georisk(self, matrix, alpha=5):

        return getGeoRisk(matrix, alpha)

    def evaluate(self, population, bank=None):

        evaluations = []

        if 'ndcg' in self.metrics:

            ndcg = np.zeros((len(population), len(self.__n_queries)))

            for i in range(0, len(population)):
                if bank and _chromosome_to_key(population[i]) in bank:
                    ndcg[i, :] = np.array(
                        bank[_chromosome_to_key(population[i])]['ndcg'])
                else:
                    scores = self.__get_scores(list(population[i]))
                    ndcg[i, :] = self.__evaluate_ndcg(scores)

            evaluations.append(ndcg)

        if 'georisk' in self.metrics and 'ndcg' in self.metrics:
            georisk = self.__evaluate_georisk(np.transpose(ndcg))
            evaluations.append(georisk)

            for i, ind in enumerate(population):
                if bank and _chromosome_to_key(ind) in bank:
                    bank[_chromosome_to_key(ind)]['georisk'] = georisk[i]

        return zip(*evaluations)

    def evaluate_compare(self, inds, matrix):

        evaluations = []

        ndcgs = []

        for i, ind in enumerate(inds):
            ndcg = self.__evaluate_ndcg(self.__get_scores(ind))
            ndcgs.append(ndcg)
            matrix[i-2, :] = ndcg[:]

        georisk = self.__evaluate_georisk(matrix.transpose())
        evaluations.append(georisk[-2:])

        return zip(ndcgs, *evaluations)


def _eval_ind(e_function, ind, bank, model):

    if bank and _chromosome_to_key(ind) in bank:
        return np.array(
            bank[_chromosome_to_key(ind)]['ndcg'])
    else:
        return e_function(ind, model.oob_buffered_predict)