import l2rCodes
import cudf
import numpy as np

best = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
        1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0]
sbest = ''
for i in best:
    sbest += str(i)
DATASET = 'web10k'
ENSEMBLE = 1  # random forest
NTREES = 300
SEED = 1887
NUM_FOLD = '1'
METRIC = "NDCG"
sparse = False
ALGORITHM = 'rf'

X_train, y_train, query_id_train = l2rCodes.load_L2R_file(
    './dataset/' + DATASET + '/Fold' + NUM_FOLD + '/Norm.' + 'train' + '.txt', sbest, sparse)
X_test, y_test, query_id_test = l2rCodes.load_L2R_file(
    './dataset/' + DATASET + '/Fold' + NUM_FOLD + '/Norm.' + 'test' + '.txt', sbest, sparse)

model = l2rCodes.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, DATASET)

XF_train = cudf.DataFrame.from_records(X_train)
XF_test = cudf.DataFrame.from_records(X_test)
yf_train = cudf.Series(y_train)

model.fit(X_train, y_train)
resScore = model.predict(X_test)

scoreTest = [0] * len(y_test)
c = 0
for i in resScore:
    scoreTest[c] = i
    c = c + 1

ndcg, queries = l2rCodes.getEvaluation(scoreTest, query_id_test, y_test, DATASET, METRIC, "test")
base = []

arq = open(r'./baselines/' + DATASET + '/Fold' + NUM_FOLD + '/' + ALGORITHM + '.txt')
for line in arq:
    base.append([float(line.split()[0])])
basey = base.copy()

for k in range(len(basey)):
    basey[k].append(queries[k])

risk = (l2rCodes.getGeoRisk(np.array(basey), 0.1))[1]

print('NDCG:')
print(ndcg)
print('Queries:')
print(queries)
print('Georisk:')
print(risk)
