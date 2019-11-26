import gerarBaseline

# DATASET = '2003_td_dataset'
DATASET = 'web10k'

ENSEMBLE = 4
# ENSEMBLE = 1

ALGORITHM = 'reg'
# ALGORITHM = 'rf'

for i in range(5):
    # for i in range(1):
    NUM_FOLD = str(i + 1)
    print('gerando baseline ' + ALGORITHM + ' Fold' + NUM_FOLD + '... ')
    gerarBaseline.save(DATASET, NUM_FOLD, ENSEMBLE, ALGORITHM)
    print('Ok!\n')
