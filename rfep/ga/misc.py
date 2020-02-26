from pathlib import Path
from abc import ABC
import pickle
import json
from ..l2r.l2rCodes import load_L2R_file


class Persist(ABC):

    def __init__(self, path, ext):

        self.path = Path(path)
        self.ext = ext

    def save(self, obj, filename, create_if_not_exists=True):
        pass

    def load(self, filename):
        pass

    def _check_path(self, create_if_not_exists=True):

        if not self.path.exists():
            if create_if_not_exists:
                self.path.mkdir(parents=True, exist_ok=True)
                return True
            else:
                return False
        else:
            return True

    def set_path(path):

        self.path = Path(path)


class ModelPersist(Persist):

    def __init__(self, path, ext='.pkl'):
        super().__init__(path, ext=ext)

    def save(self, obj, filename, create_if_not_exists=True):

        path = self.path / (filename + self.ext)

        if self._check_path(create_if_not_exists):
            try:
                with open(path, 'wb') as handler:
                    pickle.dump(obj, handler)
                    handler.close()
            except IOError:
                raise IOError('Unable to save')

    def load(self, filename):

        path = self.path / (filename + self.ext)

        if path.exists():
            with open(path, 'rb') as handler:
                model = pickle.load(handler)
                handler.close()
            return model
        else:
            raise ValueError(f'Unable to load from {path}')


class DictPersist(Persist):

    def __init__(self, path, ext='.json'):
        super().__init__(path, ext=ext)

    def save(self, obj, filename, create_if_not_exists=True):

        path = self.path / (filename + self.ext)

        if self._check_path(create_if_not_exists):
            try:
                with open(path, 'w') as handler:
                    json.dump(obj, handler, indent=4)
                    handler.close()
            except IOError:
                raise IOError('Unable to save')

    def load(self, filename):

        path = self.path / (filename + self.ext)

        if path.exists():
            with open(path, 'r') as handler:
                obj = json.load(handler)
                handler.close()
            return obj
        else:
            raise ValueError(f'Could not load this file: {path}')


class DatasetHandler:

    def __init__(self, path):

        self.__path = path
        self.X = None
        self.y = None
        self.query_id = None

    def load(self):
        self.X, self.y, self.query_id = load_L2R_file(self.__path, False)


def _chromosome_to_key(chromosome):

    return ''.join(map(str, chromosome))
