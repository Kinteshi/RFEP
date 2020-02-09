# coding=utf-8
import numbers
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

        mask = [i == '1' for i in mask]

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

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction_mod)(
                e.predict, X, gene, [y_hat], lock)
            for e, gene in zip(self.estimators_, mask))

        n_trees = 0
        for g in mask:
            if g:
                n_trees += 1

        y_hat /= n_trees

        return y_hat

    def oob_predict(self, X, y, genes):
        """
        Compute out-of-bag prediction.
        """
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, None
        )

        genes = [i == '1' for i in genes]

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_oob_accumulate_prediction)(
                e.predict, X, gene, [predictions, n_predictions], lock, n_samples, n_samples_bootstrap, self.n_outputs_, e.random_state)
            for e, gene in zip(self.estimators_, genes))

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        return predictions

    def oob_predict_buffer(self, X, y):

        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = X.shape[0]

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples, None
        )

        prediction_buffer = np.zeros(
            (n_samples_bootstrap, len(self.estimators_)))

        prediction_buffer[:, :] = np.nan

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_oob_bufferize_prediction)(
                e.predict, X, estimator, prediction_buffer, lock, n_samples, n_samples_bootstrap, self.n_outputs_, e.random_state)
            for e, estimator in zip(self.estimators_, range(0, len(self.estimators_))))

        self.prediction_buffer_ = prediction_buffer

    def oob_buffered_predict(self, genes):

        genes = np.array(genes)
        estimators = np.where(genes == '1')[0]

        y_hat = np.zeros((self.prediction_buffer_.shape[0]))

        for sample in range(0, y_hat.shape[0]):
            y_hat[sample] += np.nansum(
                self.prediction_buffer_[sample, estimators])
            y_hat[sample] /= (self.n_estimators -
                              len(np.where(np.isnan(self.prediction_buffer_[sample, estimators]))[0]))

        return y_hat


def _oob_accumulate_prediction(predict, X, gene, out, lock, n_samples, n_samples_bootstrap, n_outputs_, random_state):

    if gene:
        unsampled_indices = _generate_unsampled_indices(
            random_state, n_samples, n_samples_bootstrap)
        p_estimator = predict(
            X[unsampled_indices, :], check_input=False)

        if n_outputs_ == 1:
            p_estimator = p_estimator[:, np.newaxis]

        with lock:
            out[0][unsampled_indices, :] += p_estimator
            out[1][unsampled_indices, :] += 1
    else:
        pass


def _oob_bufferize_prediction(predict, X, estimator, out, lock, n_samples, n_samples_bootstrap, n_outputs_, random_state):

    unsampled_indices = _generate_unsampled_indices(
        random_state, n_samples, n_samples_bootstrap)
    p_estimator = predict(
        X[unsampled_indices, :], check_input=False)

    with lock:
        out[unsampled_indices, estimator] = p_estimator


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


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(round(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples,
                                              n_samples_bootstrap)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices
