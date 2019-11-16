from deapForL2r import main
from pstats import SortKey


def run():
    for fold in range(1, 2):
        for params in [['precision']]:
            main(DATASET='2003_td_dataset', NUM_FOLD=str(fold), METHOD='spea2', PARAMS=params,
                 NUM_GENES=100, NUM_INDIVIDUOS=25, NUM_GENERATIONS=50)


run()
