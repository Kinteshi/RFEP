# %%
from rfep.ga.evaluation import Evaluator
from rfep.ga.misc import DatasetHandler, ModelPersist, DictPersist
from rfep.ga.pruning import GeneticAlgorithmRandomForest
from rfep.ga.analisis import Analyst


# %%

dataset_name = '2003_td_dataset'
folds = [1]
evolution_set = 'vali'
oob_evolution = False
n_gen = 5
n_trees = 20
pop_size = 10
seed = 2567

objectives = ['ndcg']
weights = [1]
run_name = 'test_run'

load_from_storage = True
save_to_storage = False

# %%


for fold in folds:
    eset = DatasetHandler(
        f'data/dataset/{dataset_name}/Fold{fold}/Norm.{evolution_set}.txt')
    eset.load()
    ev = Evaluator(objectives, weights,
                   dataset_name, eset.X, eset.y, eset.query_id, _oob=oob_evolution)

    mp = ModelPersist(f'output/forests/{dataset_name}/Fold{fold}')
    dp = DictPersist(f'output/{run_name}/Fold{fold}')
    ga = GeneticAlgorithmRandomForest(
        n_trees, ev, model_persist=mp, dict_persist=dp, pop_size=pop_size, seed=seed)
    train = DatasetHandler(
        f'data/dataset/{dataset_name}/Fold{fold}/Norm.train.txt')
    train.load()
    ga.fit_model(train.X, train.y, load_from_storage=load_from_storage,
                 save_to_storage=save_to_storage)
    ga.evolve_model(n_gen)

# %%
test = DatasetHandler(
    f'data/dataset/{dataset_name}/Fold{fold}/Norm.test.txt')
test.load()
test_ev = Evaluator(objectives, weights,
                    dataset_name, test.X, test.y, test.query_id)
test_mp = ModelPersist(f'output/forests/{dataset_name}')
test_dp = DictPersist(f'output/{run_name}')
an = Analyst(test_mp, test_dp, test_ev,
             f'data/baselines/{dataset_name}', n_trees, seed)
an.report(folds)
an.final_report()
