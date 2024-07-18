import json
import os.path
from mode import Mode

class MetaData:

    def __init__(self, file_name: str, target: str, output_folder: str,
                 mode: Mode, low: float, high: float,
                 levels: int = 3, amplification: int = 20
                 ):
        self.__data = {
            'file': file_name,
            'output': output_folder,
            'target': target,
            'low': low,
            'high': high,
            'levels': levels,
            'amplification': amplification,
            'mode': mode.name,
            'date': None,
        }

    def __getitem__(self, item):
        return self.__data[item]
