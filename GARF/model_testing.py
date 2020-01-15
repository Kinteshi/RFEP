# %%
import random
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
X_train, y_train, query_id_train = load_L2R_file(
    './dataset/' + '2003_td_dataset' + '/Fold' + '1' + '/Norm.' + 'train' + '.txt', False)
model = Forest(n_estimators=50, max_features=0.3, max_leaf_nodes=100, min_samples_leaf=1,
               random_state=2525, n_jobs=-1)
model.fit(X_train, y_train)

# %%
print(model.oob_predict(X_train, y_train, '1'*50))
print(model.predict(X_train, '1'*50))

# %%
