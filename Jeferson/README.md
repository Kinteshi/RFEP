# Feature Selection for Learning To Rank with Multi Objective Genetic Algorithms

Códigos para feature selection, utilizando algoritmos genéticos em contexto de Aprendizagem de Ranqueamento


## Datasets

* [WEB10K](https://www.microsoft.com/en-us/research/project/mslr/) - Microsoft Learning to Rank Datasets 
* [LETOR](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/) - Learning to Rank for Information Retrieval 
* [Yahoo](http://proceedings.mlr.press/v14/chapelle11a/chapelle11a.pdf) - Yahoo! Learning to Rank Challenge Overview

## Built With

* [DEAP](https://github.com/deap) - Distributed Evolutionary Algorithms in Python
* [cuML](https://github.com/rapidsai/cuml) - RAPIDS Machine Learning Library 
* [cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame Library
* [scikit-learn](https://github.com/scikit-learn/scikit-learn) - Machine Learning in Python


## Configuration

A biblioteca DEAP não disponibiliza comparação de objetivos multivalorados com TTest. Devido esse problema foi desenvolvido um fork da biblioteca, alterando as funções de dominância, selSPEA2 e selNSGA2. Agora essas aplicam teste estatístico para comparação dos indivíduos. 

Para instalar desinstale versões anteriores do DEAP:
```
pip uninstall deap
```
E instale com o código proveniente do fork:
```
pip install git+https://github.com/Haiga/deap#egg=deap
```

Nota: Esse fork utiliza a biblioteca [rpy2](https://rpy2.bitbucket.io/) - R in Python, e deve ser instalada previamente.
