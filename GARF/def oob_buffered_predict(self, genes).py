def oob_buffered_predict(self, genes):

    genes = np.array(genes)
    estimators = np.where(genes == '1')[0]

    y_hat = np.zeros((self.prediction_buffer_.shape[0]))

    n_jobs, _, _ = _partition_estimators(
        self.prediction_buffer_.shape[0], self.n_jobs)

    lock = threading.Lock()

    Parallel(n_jobs=n_jobs, verbose=self.verbose,
             **_joblib_parallel_args(require="sharedmem"))(
        delayed(_oob_accumulate_buffered_prediction)(
            self.prediction_buffer_, sample, estimators, self.n_estimators, y_hat, lock)
        for sample in range(0, y_hat.shape[0]))
    return y_hat


def _oob_accumulate_buffered_prediction(buffer, sample, estimators, n_estimators, out, lock):

    out[sample] += np.nansum(buffer[sample, estimators])
    out[sample] /= (n_estimators -
                    len(np.where(np.isnan(buffer[sample, estimators]))[0]))
