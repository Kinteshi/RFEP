# %%
from ScikitLearnModificado.forest import Forest
import timeit
import numpy as np
from l2rCodesSerial import load_L2R_file
import pickle
import matplotlib.pyplot as plt
import psutil
import os
import json


# %%
dataset = 'web10k'
seed = 2567


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


trees = []

no_buffer_serial = []
no_buffer_parallel = []

buffer_creation = []
buffer_serial = []
buffer_parallel = []

no_buffer_mem = []
buffer_mem = []

pid = os.getpid()
py = psutil.Process(pid)

# %%
for forest in np.arange(100, 5100, 100):
    trees.append(forest)

    forest_path = os.getcwd() + '/output/forests/'
    if not os.path.exists(forest_path + f'{dataset}{seed}{forest}/'):
        os.mkdir(forest_path + f'{dataset}{seed}{forest}/')

    forest_path += f'{dataset}{seed}{forest}/'

    if not os.path.exists(forest_path + f'Fold{1}.pkl'):
        X_train, y_train, query_id_train = load_L2R_file(
            './dataset/' + dataset + '/Fold' + '1' + '/Norm.' + 'train' + '.txt', False)
        model = Forest(n_estimators=forest, max_features=0.3, max_leaf_nodes=100, min_samples_leaf=1,
                       random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)

        with open(forest_path + f'Fold{1}.pkl', 'wb') as forest:
            pickle.dump(model, forest)
            forest.close()
    else:
        with open(forest_path + f'Fold{1}.pkl', 'rb') as forest:
            model = pickle.load(forest)
            forest.close()

    memoryUse = py.memory_info()[0] / 2.**30
    no_buffer_mem.append(memoryUse)

    evaluating = wrapper(model.oob_predict, X_train, y_train,
                         '1'*forest, False)
    time = timeit.timeit(evaluating, number=1)
    no_buffer_serial.append(time)

    evaluating = wrapper(model.oob_predict, X_train, y_train,
                         '1'*forest, True)
    time = timeit.timeit(evaluating, number=1)
    no_buffer_parallel.append(time)

    evaluating = wrapper(model.oob_predict_buffer, X_train, y_train)
    time = timeit.timeit(evaluating, number=1)
    buffer_creation.append(time)
    memoryUse = py.memory_info()[0] / 2.**30
    buffer_mem.append(memoryUse)

    evaluating = wrapper(model.oob_buffered_predict, list(
        '1'*forest))
    time = timeit.timeit(evaluating, number=1)
    buffer_serial = time

    evaluating = wrapper(model.oob_buffered_predict, list(
        '1'*forest), True)
    time = timeit.timeit(evaluating, number=1)
    buffer_parallel = time

    json_holder = {}

    json_holder['trees'] = [int(t) for t in trees]

    json_holder['noBufferSerial'] = no_buffer_serial
    json_holder['noBufferParallel'] = no_buffer_parallel

    json_holder['bufferCreation'] = buffer_creation

    json_holder['bufferSerial'] = buffer_serial
    json_holder['bufferParallel'] = buffer_parallel

    json_holder['bufferMem'] = buffer_mem
    json_holder['noBufferMem'] = no_buffer_mem

    with open('res.json', 'w') as file:
        json.dump(json_holder, file, indent=4)
        file.close()

    del model

# %%


# %%
'''
fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot()

ax.plot(no_buffer, trees, color='green')
ax.plot(buffer, trees, color='blue')

ax.set_xlim(np.max(trees), np.min(trees))
ax.set_ylim(np.max(no_buffer + buffer), np.min(no_buffer + buffer))
ax.set_xlabel('Trees')
ax.set_ylabel('Time')

ax.legend(['No buffer', 'Buffer'], loc='upper left')
ax.set_title('Time')

plt.savefig('TimeTest.png')

# %%
ax.cla()

ax.plot(no_buffer_mem, trees, color='green')
ax.plot(buffer_mem, trees, color='blue')

ax.set_xlim(np.max(trees), np.min(trees))
ax.set_ylim(np.max(no_buffer_mem + buffer_mem),
            np.min(no_buffer_mem + buffer_mem))
ax.set_xlabel('Trees')
ax.set_ylabel('Memory')

ax.legend(['No buffer', 'Buffer'], loc='upper left')
ax.set_title('Memory')

plt.savefig('MemTest.png')
'''
# %%
