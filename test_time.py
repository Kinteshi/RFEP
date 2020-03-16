# %%
from rfep.ga.evaluation import Evaluator
from rfep.ga.misc import DatasetHandler, ModelPersist, DictPersist
from rfep.ga.pruning import GeneticAlgorithmRandomForest
from rfep.ScikitLearnModificado.forest import Forest
from functools import partial
from rfep.ga.analisis import Analyst
import shutil

# %%
dataset_name = 'web10k'
folds = [1]
evolution_set = 'train'
oob_evolution = True
buffered = True
n_gen = 3
n_trees = 5000
pop_size = 10
seed = 2567

objectives = ['ndcg']
weights = [1]
run_name = 'test_run'

# %%


for fold in folds:
    train = DatasetHandler(
        f'data/dataset/{dataset_name}/Fold{fold}/Norm.train.txt')
    train.load()

    mp = ModelPersist(f'output/forests/{dataset_name}/Fold{fold}')

    try:
        model = mp.load(f'{n_trees}{seed}')
    except:
        model = Forest(
                n_estimators=n_trees,
                max_features=0.3,
                max_leaf_nodes=100,
                n_jobs=-1,
                random_state=seed,
                min_samples_leaf=1
            )
        model.fit(train.X, train.y)
        mp.save(model, f'{n_trees}{seed}')

    if evolution_set == 'train':
        eset = train
    else:
        eset = DatasetHandler(
            f'data/dataset/{dataset_name}/Fold{fold}/Norm.{evolution_set}.txt')
        eset.load()

    if oob_evolution:
        if buffered:
            model.oob_predict_buffer(train.X, train.y)
            p_method = partial(model.oob_buffered_predict)
        else:
            p_method = partial(model.oob_predict, train.X, train.y)
    else:
        p_method = partial(model.oob_predict, eset.X, eset.y)

    ev = Evaluator(objectives, weights,
                   dataset_name, eset.X, eset.y, eset.query_id)

    ev.set_predict_method(p_method)

    dp = DictPersist(f'output/{run_name}/Fold{fold}')
    ga = GeneticAlgorithmRandomForest(
        n_trees, ev, dict_persist=dp, pop_size=pop_size, seed=seed)

    ga.evolve_model(n_gen)

# %%

test_mp = ModelPersist(f'output/forests/{dataset_name}')
test_dp = DictPersist(f'output/{run_name}')
an = Analyst(objectives, weights, test_mp, test_dp, f'data/baselines/{dataset_name}', dataset_name, n_trees, seed)
an.report(folds)
an.final_report()

# %%

shutil.make_archive(f'output/{run_name}', 'zip', f'output/{run_name}')
