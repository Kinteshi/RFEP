# coding=utf-8
import pickle
import threading
from warnings import warn
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.base import _partition_estimators
from sklearn.ensemble.forest import MAX_INT, _parallel_build_trees, _accumulate_prediction
from sklearn.exceptions import DataConversionWarning
from sklearn.externals.joblib import Parallel, delayed
from sklearn.tree._tree import issparse, DOUBLE, DTYPE
from sklearn.utils import check_array, check_random_state
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted


class Forest(RandomForestRegressor):

    def predict(self, X, mask):
        """Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        mask : array que armazena a informaçao de haver ou não haver a arvore na floresta.

        fileCache : Rota onde se encontra as arvores que seram trabalhadas, lembrando que cada colecao e cada fold
            possui um conjunto unico de arvores
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """

        #temp = np.copy(self.estimators_)
        mask = [i == '1' for i in mask]

        #self.estimators_ = self.estimators_[mask]

        self.n_outputs_ = 1

        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        def pred(e, gene):
            if gene:
                e.predict()



        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction_mod)(e.predict, X, gene, [y_hat], lock)
            for e, gene in zip(self.estimators_, mask))

        n_trees = 0
        for g in mask:
            if g:
                n_trees += 1

        y_hat /= n_trees

        #self.estimators_ = temp

        return y_hat


def _accumulate_prediction_mod(predict, X, gene, out, lock):
    """This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    if gene:
        prediction = predict(X, check_input=False)
        with lock:
            if len(out) == 1:
                out[0] += prediction
            else:
                for i in range(len(out)):
                    out[i] += prediction[i]
    else:
        pass
