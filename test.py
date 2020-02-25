# %%
from GARF.rfep.evaluation import Evaluator
from GARF.rfep.misc import DatasetHandler, ModelPersist, DictPersist
from GARF.rfep.pruning import GeneticAlgorithmRandomForest

# %%
val = DatasetHandler('GARF/dataset/2003_td_dataset/Fold1/Norm.vali.txt')
val.load()
ev = Evaluator(['ndcg'], [1], '2003_td_dataset', val.X, val.y, val.query_id)

mp = ModelPersist('GARF/output/forests/2003_td_dataset/Fold1')
dp = DictPersist('GARF/output/TESTENOVOLINDO/Fold1')

# %%
ga = GeneticAlgorithmRandomForest(
    20, ev, model_persist=mp, dict_persist=dp, pop_size=10)
train = DatasetHandler('GARF/dataset/2003_td_dataset/Fold1/Norm.train.txt')
train.load()
ga.fit_model(train.X, train.y, True)


# %%
ga.evolve_model(5)


# %%
