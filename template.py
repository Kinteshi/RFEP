# %%
from rfep.ga.evaluation import Evaluator
from rfep.ga.misc import DatasetHandler, ModelPersist, DictPersist
from rfep.ga.pruning import GeneticAlgorithmRandomForest
from rfep.ScikitLearnModificado.forest import Forest
from functools import partial
from rfep.ga.analisis import Analyst
import shutil

# %%
dataset_name = 'example'  # str :: Name of the dataset and also the name on the folder
# list[int] :: List containing the folds that'll be used on the training
folds = [1, 2, 3, 4, 5]
evolution_set = 'vali'  # str :: Either 'train' or 'vali'. 'train' needs to be used with 'oob_evolution' = True and 'vali' needs it to be False
oob_evolution = evolution_set == 'train'  # bool :: True if train false if vali
buffered = True  # bool :: Sets use of buffer in the evolution
n_gen = 5  # int :: Number of generations to compute
n_trees = 20  # int :: Size of the ensemble forest
pop_size = 10  # int :: Size of the population in each generation
seed = 2567  # int :: Random seed to reproduce similar results, mostly for the forest to be the same. Irrelevant to the evolution

# list[str] :: List of metrics to use. As of now it is obligatory to contain 'ndcg', but can also contain 'georisk' along with it
objectives = ['ndcg', 'georisk']
# list[int] :: Needs to be the same size of 'objectives' as represents the weigth of each one of the metrics
weights = [1, 1]
run_name = 'test_run'  # str :: Name of the output folder and zipped file

# %%


for fold in folds:

    # DatasetHandler take as attr a relative or absolute path to find the dataset
    train = DatasetHandler(
        f'data/dataset/{dataset_name}/Fold{fold}/Norm.train.txt')

    # This next line loads the dataset in memory
    train.load()

    # ModelPersist take as attr the path in which it'll persist the forest for reuse
    mp = ModelPersist(f'output/forests/{dataset_name}/Fold{fold}')

    try:
        # The load method tries to read a forest
        model = mp.load(f'{n_trees}{seed}')
    except:
        # If it fails a new forest is trained and then saved in storage
        model = Forest(
            n_estimators=n_trees,
            max_features=0.3,
            max_leaf_nodes=100,
            n_jobs=-1,
            random_state=seed,
            min_samples_leaf=1
        )

        # Note that the train::DatasetPersist object is being used to pass the data to fit the forest
        model.fit(train.X, train.y)
        # By default the forest is saved with n_trees+seed as name
        mp.save(model, f'{n_trees}{seed}')

    # Checks which set of the data will be used for evolution to avoid extra loading overhead
    if evolution_set == 'train':
        eset = train
    else:
        eset = DatasetHandler(
            f'data/dataset/{dataset_name}/Fold{fold}/Norm.{evolution_set}.txt')
        eset.load()

    # Checks which prediction strategy will be used and partially apply the method so that the last parameter is the chromosome
    if oob_evolution:
        if buffered:
            # This next line creates the buffer
            model.oob_predict_buffer(train.X, train.y)

            p_method = partial(model.oob_buffered_predict)
        else:
            p_method = partial(model.oob_predict, train.X, train.y)
    else:
        p_method = partial(model.oob_predict, eset.X, eset.y)

    # The evaluator class takes care of the evaluation
    ev = Evaluator(objectives, weights,
                   dataset_name, eset.X, eset.y, eset.query_id)

    # Passing the method onto the evaluator object
    ev.set_predict_method(p_method)

    # DictPersist is necessary for outputting the data generated in evolution
    dp = DictPersist(f'output/{run_name}/Fold{fold}')

    # Instantiating the evolution class
    ga = GeneticAlgorithmRandomForest(
        n_trees, ev, dict_persist=dp, pop_size=pop_size, seed=seed)

    # Evolve for n_gen!
    ga.evolve_model(n_gen)

# %%

# This bit if for pre-processing the results and is not really necessary if you want to venture in the data all by yourself
test_mp = ModelPersist(f'output/forests/{dataset_name}')
test_dp = DictPersist(f'output/{run_name}')
an = Analyst(objectives, weights, test_mp, test_dp,
             f'data/baselines/{dataset_name}', dataset_name, n_trees, seed)
an.report(folds)
an.final_report()

# %%

# This last line is for zipping the files altogether
shutil.make_archive(f'output/{run_name}', 'zip', f'output/{run_name}')
