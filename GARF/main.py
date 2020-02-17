from deapForL2r import main
from resultAnalisis import generate_report, make_graphics, plot_pareto_front, final_report
import shutil
import sys
import os
import json
import time


input_file_path = os.getcwd() + '/' + sys.argv[1]
assert os.path.exists(input_file_path), 'This file doesn\'t exist'
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    options_dict = json.load(input_file)
    input_file.close()

for experiment_options in options_dict:
    output_path = os.getcwd() + '/output/'
    results_path = output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(output_path + 'forests/')
    output_path += experiment_options['outputOptions']['shortExperimentIdentifier'] + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for fold in experiment_options['datasetOptions']['folds']:
        if not os.path.exists(output_path + f'Fold{fold}/'):
            os.mkdir(output_path + f'Fold{fold}/')

        experiment_options['datasetOptions']['fold'] = fold
        ident = experiment_options['outputOptions']['shortExperimentIdentifier']
        if experiment_options['outputOptions']['runExperiment']:
            with open(f'./output/{ident}/Fold{fold}/config.json', 'w') as file:
                json.dump(experiment_options, file, indent=4)
            start_time = time.time()
            main(experiment_options)
            end_time = time.time()
            elapsed_time = end_time - start_time
        else:
            with open(f'./output/{ident}/Fold{fold}/resultReport.json', 'r') as res_file:
                elapsed_time = json.load(
                    res_file)['overallStats']['elapsedTime']
                res_file.close()
        if experiment_options['outputOptions']['generateReport']:
            generate_report(results_path, ident, fold, elapsed_time)
        if experiment_options['outputOptions']['generatePlots']:
            make_graphics(results_path, ident, fold)
            if len(experiment_options['geneticAlgorithmOptions']['fitnessMetrics']) > 1:
                plot_pareto_front(results_path, ident, fold)
    final_report(results_path, ident)
    if experiment_options['outputOptions']['zipFiles']:
        shutil.make_archive(results_path + experiment_options['outputOptions']['shortExperimentIdentifier'], 'zip',
                            results_path + experiment_options['outputOptions']['shortExperimentIdentifier'])
