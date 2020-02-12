from ScikitLearnModificado.forest import Forest
import timeit
import numpy as np
from l2rCodesSerial import load_L2R_file
import pickle
import matplotlib.pyplot as plt
dataset = 'web10k'


X_train, y_train, query_id_train = load_L2R_file(
    './dataset/' + dataset + '/Fold' + '1' + '/Norm.' + 'train' + '.txt', False)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


no_buffer = []
buffer = []
trees = []

for trees in np.arange(100, 1000, 100):
    model = Forest(n_estimators=trees, max_features=0.3, max_leaf_nodes=100, min_samples_leaf=1,
                   random_state=2567, n_jobs=-1)
    model.fit(X_train, y_train)

    evaluating1 = wrapper(model.oob_predict, X_train, y_train,
                          ('1' * trees) + ('0' * (5000-trees)), False)
    time1 = timeit.timeit(evaluating1, number=1)

    model.oob_predict_buffer(X_train, y_train)

    evaluating2 = wrapper(model.oob_buffered_predict, list(
        ('1' * trees) + ('0' * (5000-trees))))
    time2 = timeit.timeit(evaluating2, number=1)

    no_buffer.append(time1)
    buffer.append(time2)
    trees.append(trees)

fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot()

ax.plot(no_buffer, trees, color='green')
ax.plot(buffer, trees, color='blue')

ax.set_xlim(np.max(trees), np.min(trees))
ax.set_ylim(np.max(no_buffer + buffer), np.min(no_buffer + buffer))
ax.set_xlabel('Trees')
ax.set_ylabel('Time')

ax.legend(['No buffer', 'Buffer'], loc='upper left')

plt.savefig('TimeTest.png')

json_holder = {}

json_holder['nobuffer'] = no_buffer
json_holder['buffer'] = buffer
json_holder['trees'] = trees
