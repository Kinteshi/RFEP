from deapForL2r import main

for fold in range(1, 6):
    for params in [['precision'], ['risk'], ['precision, risk']]:
        main(DATASET='web10k', FOLD=str(fold), METHOD='spea2', PARAMS=params,
             NUM_GENES=10000, NUM_INDIVIDUOS=100, NUM_GENERATIONS=50)
