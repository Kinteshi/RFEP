from pathlib import Path, WindowsPath
import re
import numpy as np
import datetime as date
import time


def reformat_baselines():
    path = Path('data/baselines/web10k/Fold1')

    for file_name in path.rglob(r'*'):
        new_file = ''
        with open(file_name, 'r') as file:
            for line in file:
                if re.search(r'mean=>\s*\d[.]*[0-9]*', line):
                    new_file += re.search(r'\d[.]*[0-9]*\n', line).group()
                    # print(new_file)
            file.close()

        with open(f'{file_name}.txt', 'w') as file:
            file.write(new_file)
            file.close()


def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, WindowsPath):
        return obj.__str__()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    try:
        return obj.__dict__
    except:
        return str(obj)
