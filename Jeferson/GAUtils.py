from ScikitLearnModificado.forest import Forest
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.tree._tree import DTYPE
from sklearn.metrics.regression import r2_score
from warnings import warn


def oob_synthetic(X_train, y_train, forest):
    X_train = check_array(X_train, dtype=DTYPE, accept_sparse='csr')

    n_samples = y_train.shape[0]

    predictions = np.zeros((n_samples, forest.n_outputs_))
    n_predictions = np.zeros((n_samples, forest.n_outputs_))

    for estimator in forest.estimators_:
        unsampled_indices = _generate_unsampled_indices(
            estimator.random_state, n_samples)
        p_estimator = estimator.predict(
            X_train[unsampled_indices, :], check_input=False)

        if forest.n_outputs_ == 1:
            p_estimator = p_estimator[:, np.newaxis]

        predictions[unsampled_indices, :] += p_estimator
        n_predictions[unsampled_indices, :] += 1

    if (n_predictions == 0).any():
        warn("Some inputs do not have OOB scores. "
             "This probably means too few trees were used "
             "to compute any reliable oob estimates.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions
    forest.oob_prediction_ = predictions

    if forest.n_outputs_ == 1:
        forest.oob_prediction_ = \
            forest.oob_prediction_.reshape((n_samples,))
    '''
    forest.oob_score_ = 0.0

    for k in range(forest.n_outputs_):
        forest.oob_score_ += r2_score(y_train[:, k], predictions[:, k])

    forest.oob_score_ /= forest.n_outputs_
    '''