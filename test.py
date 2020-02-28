# %%
from rfep.ga.evaluation import Evaluator
from rfep.ga.misc import DatasetHandler, ModelPersist, DictPersist
from rfep.ga.pruning import GeneticAlgorithmRandomForest
from rfep.ga.analisis import Analyst

# %%
'''
val = DatasetHandler('data/dataset/2003_td_dataset/Fold1/Norm.vali.txt')
val.load()
ev = Evaluator(['ndcg', 'georisk'], [1, 1],
               '2003_td_dataset', val.X, val.y, val.query_id)

mp = ModelPersist('output/forests/2003_td_dataset/Fold1')
dp = DictPersist('output/TESTENOVOLINDO/Fold1')

# %%
ga = GeneticAlgorithmRandomForest(
    75, ev, model_persist=mp, dict_persist=dp, pop_size=50)
train = DatasetHandler('data/dataset/2003_td_dataset/Fold1/Norm.train.txt')
train.load()
ga.fit_model(train.X, train.y, load_from_storage=True)


# %%
ga.evolve_model(20)
'''

# %%
test = DatasetHandler('data/dataset/2003_td_dataset/Fold1/Norm.test.txt')
test.load()

# %%
test_ev = Evaluator(['ndcg', 'georisk'], [1, 1],
                    '2003_td_dataset', test.X, test.y, test.query_id)
test_mp = ModelPersist('output/forests/2003_td_dataset/')
test_dp = DictPersist('output/TESTENOVOLINDO')


# %%
an = Analyst(test_mp, test_dp, test_ev,
             'data/baselines/2003_td_dataset', 75, 2567)

# %%
an.report([1])
