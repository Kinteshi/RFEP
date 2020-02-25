import time


class Timer:
    def __init__(self, nome=''):
        self.nome = nome
        self.max = None
        self.min = None
        self.count = 0
        self.sum = 0
        self.mean = 0
        self.time1 = 0
        self.time2 = 0

    def start(self):
        self.time1 = time.time()

    def stop(self):
        self.time2 = time.time()

        dif = self.time2 - self.time1

        if self.max is None or dif > self.max:
            self.max = dif

        if self.min is None or dif < self.min:
            self.min = dif

        self.count += 1
        self.sum += dif
        self.mean = self.sum / self.count

    def getInfo(self):
        print('---- ' + self.nome + ' -----')
        print('Total Executions: ' + str(self.count))
        print('Média Tempo de execução: ' + str(self.mean))
        print('Tempo Total de execução: ' + str(self.sum))

        print('Mínimo tempo de execução: ' + str(self.min))
        print('Máximo tempo de execução: ' + str(self.max))
        print('\n')
        info = {'count': self.count, 'mean': self.mean, 'sum': self.sum, 'min': self.min, 'max': self.max}
        return info


def testTimer():
    t = Timer()
    t.start()
    t.stop()
    print(t.mean)
    print(t.sum)
    print(t.count)
    print(t.min)
    print(t.max)

    t.start()
    time.sleep(5)
    t.stop()
    print(t.mean)
    print(t.sum)
    print(t.count)
    print(t.min)
    print(t.max)
