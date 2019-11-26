import random


def get(DATASET, FOLD, NUM_INDIVIDUOS):
    sintetic = []
    file_sintetic = r'./sintetic/' + DATASET + '.SinteticPopulation.Fold' + str(int(FOLD) - 1)
    fp = open(file_sintetic)
    cont = 0
    for line in fp:
        values = line.split('= ')[1]
        values = values.replace('\n', '')
        values = values.split(';')
        integer_values = []
        for value in values:
            integer_values.append(int(value))
        sintetic.append(integer_values)
        cont += 1
        if cont == NUM_INDIVIDUOS:
            return sintetic
    tamanho_individuo = len(sintetic[0])
    while cont != NUM_INDIVIDUOS:
        new_sintetic = []
        for i in range(tamanho_individuo):
            new_sintetic.append(random.randint(0, 1))
        sintetic.append(new_sintetic)
        cont += 1
    return sintetic
