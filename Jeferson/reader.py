#%%
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%%

def draw_experiment_graph(dataset, fold, params, identifier_string, middle_path):
    with open(f'new_resultados/{middle_path}{dataset}-Fold{fold}-base-testingspea2{params}{identifier_string}archives.json', 'r') as file:
        archive = json.load(file)
        file.close()
    

    with open(f'new_resultados/{middle_path}{dataset}-Fold{fold}-base-testingspea2{params}{identifier_string}.json', 'r') as base:
        inds = json.load(base)
        base.close()
        
    
    # generations data
    gen_size = 50
    gen = []
    for i in range(0, gen_size):
        gen.append({})
        
    for ind, specs in inds.items():
        gen[specs['geracao_s'] - 1][ind] = specs
        
    stats_generations = {}
    stats_generations['max'] = np.zeros(gen_size)
    stats_generations['min'] = np.zeros(gen_size)
    stats_generations['mean'] = np.zeros(gen_size)
    stats_generations['var'] = np.zeros(gen_size)
    stats_generations['std'] = np.zeros(gen_size)
    
    for i in range(0, gen_size):
        g = gen[i]
        precisions = [np.mean(spec['precision']) for ind, spec in g.items()]
        stats_generations['max'][i] = np.max(precisions)
        stats_generations['min'][i] = np.min(precisions)
        stats_generations['mean'][i] = np.mean(precisions)
        stats_generations['var'][i] = np.var(precisions)
        stats_generations['std'][i] = np.std(precisions)
    
    
    # Archive data
    
    gen = []
    for i in range(0, gen_size):
        gen.append({})
    
    for i in range(1, gen_size + 1):
        gen_archive = archive[str(i)]
        for ind in gen_archive:
            key = [str(gene) for gene in ind]
            key = ''.join(key)
            gen[i - 1][key] = inds[key]
    
    
    stats_archives = {}
    stats_archives['max'] = np.zeros(gen_size)
    stats_archives['min'] = np.zeros(gen_size)
    stats_archives['mean'] = np.zeros(gen_size)
    stats_archives['var'] = np.zeros(gen_size)
    stats_archives['std'] = np.zeros(gen_size)
    
    for i in range(0, gen_size):
        g = gen[i]
        precisions = [np.mean(spec['precision']) for ind, spec in g.items()]
        stats_archives['max'][i] = np.max(precisions)
        stats_archives['min'][i] = np.min(precisions)
        stats_archives['mean'][i] = np.mean(precisions)
        stats_archives['var'][i] = np.var(precisions)
        stats_archives['std'][i] = np.std(precisions)
    
    
     
    sns.set()    
    sns.set_style('whitegrid')
    sns.set_palette('dark')
    sns.set_context('poster')
    
    
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot()
    figname = dataset + fold + params + identifier_string
    
    legend = []
    for stat in ['max','mean','min']:
        ax.plot(stats_generations[stat])
        legend.append(stat)
    
    
    ax.set_ylabel('NDCG')
    ax.set_xlabel('Generations')
    ax.legend(legend)
    ax.set_title('NDCG by generation')
    plt.savefig(figname+'mmmby_gen')
    
    ax.cla()
    legend = []
    for stat in ['var', 'std']:
        ax.plot(stats_generations[stat])
        legend.append(stat)
    
    ax.set_ylabel('NDCG')
    ax.set_xlabel('Generations')
    ax.legend(legend)
    ax.set_title('NDCG by generation')
    plt.savefig(figname+'stdvby_gen')
    
    ax.cla()
    legend = []
    for stat in ['max', 'mean', 'min']:
        ax.plot(stats_archives[stat])
        legend.append(stat)

    ax.set_ylabel('NDCG')
    ax.set_xlabel('Generations')    
    ax.legend(legend)
    ax.set_title('NDCG by generation\'s archive')
    plt.savefig(figname+'mmmby_archive')
    
    ax.cla()
    legend = []
    for stat in ['var', 'std']:
        ax.plot(stats_archives[stat])
        legend.append(stat)
 
    ax.set_ylabel('NDCG')
    ax.set_xlabel('Generations')   
    ax.legend(legend)
    ax.set_title('NDCG by generation\'s archive')
    plt.savefig(figname+'stdvby_archive')
    
    
#%%


draw_experiment_graph('web10k', '1', 'precision', 'Mod50Mut90Cx', 'modified_params/')
draw_experiment_graph('web10k', '1', 'precision', 'Original20Mut90Cx', 'original_params/')