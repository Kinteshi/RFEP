#%%
from ScikitLearnModificado import Forest
from l2rCodesSerial import load_L2R_file
import numpy as np
import json
from evaluateIndividuoSerial import getEval

#%%
dataset, fold, gen_n, identifier_string, params, seed, sparse = 'web10k', '1', 25, 'Mod50Mut90Cx', 'precision', 1313, False


#%%
#def main(dataset, fold, gen_n, identifier_string, params, seed, sparse=False):

with open(f'new_resultados/modified_params/{dataset}-Fold{fold}-base-testingspea2{params}{identifier_string}topind.json') as file:
    top = json.load(file)
    file.close()

best_ind = ''

for g in top['precision']['ind']:
    best_ind += str(g)

n_genes = len(best_ind)

original_ind = '1' * n_genes
#%%
X_train, y_train, query_id_train = load_L2R_file(
    './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'train' + '.txt', '1' * n_genes, sparse)
X_test, y_test, query_id_test = load_L2R_file(
    './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'test' + '.txt', '1' * n_genes, sparse)
X_vali, y_vali, query_id_vali = load_L2R_file(
        './dataset/' + dataset + '/Fold' + fold + '/Norm.' + 'vali' + '.txt', '1' * n_genes, sparse)

model = Forest(n_estimators=n_genes, max_features=0.3, max_leaf_nodes=100, min_samples_leaf=1,
               random_state=seed, n_jobs=-1)

model.fit(X_train, y_train)

model.estimators_ = np.array(model.estimators_)

#%%
original_metrics = getEval(original_ind, model, n_genes, X_vali, y_vali,
                           query_id_test, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg')

ga_metrics = getEval(best_ind, model, n_genes, X_vali, y_vali,
                     query_id_vali, 1, n_genes, seed, dataset, 'NDCG', fold, 'reg')

#%%
original_metrics = np.mean(original_metrics[0])
ga_metrics = np.mean(ga_metrics[0])

#%%
print(f'Original: {original_metrics}')
print(f'GA: {ga_metrics}')