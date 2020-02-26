# %%
from rfep.ga.evaluation import Evaluator
from rfep.ga.misc import DatasetHandler, ModelPersist, DictPersist
from rfep.ga.pruning import GeneticAlgorithmRandomForest

# %%
val = DatasetHandler('data/dataset/2003_td_dataset/Fold1/Norm.vali.txt')
val.load()
ev = Evaluator(['ndcg'], [1], '2003_td_dataset', val.X, val.y, val.query_id)

mp = ModelPersist('output/forests/2003_td_dataset/Fold1')
dp = DictPersist('output/TESTENOVOLINDO/Fold1')

# %%
ga = GeneticAlgorithmRandomForest(
    20, ev, model_persist=mp, dict_persist=dp, pop_size=10)
train = DatasetHandler('dataset/2003_td_dataset/Fold1/Norm.train.txt')
train.load()
ga.fit_model(train.X, train.y, True)


# %%
ga.evolve_model(5)


# %%
