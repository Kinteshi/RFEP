from deapForL2r import main

import sys
import os
import json

input_file_path = os.getcwd() + '/' + sys.argv[1]
assert os.path.exists(input_file_path), 'This file doesn\'t exist'
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    options_dict = json.load(input_file)
    input_file.close()

for input_options in options_dict:
    output_path = os.getcwd() + '\\output\\'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(output_path + 'forests\\')
    output_path += input_options['outputOptions']['shortExperimentIdentifier'] + '\\'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    fold = input_options['datasetOptions']['fold']

    if not os.path.exists(output_path + f'Fold{fold}\\'):
        os.mkdir(output_path + f'Fold{fold}\\')

    ident = input_options['outputOptions']['shortExperimentIdentifier']
    fold = input_options['datasetOptions']['fold']
    with open(f'./output/{ident}/Fold{fold}/config.json', 'w') as file:
        json.dump(input_options, file)

    main(input_options)
