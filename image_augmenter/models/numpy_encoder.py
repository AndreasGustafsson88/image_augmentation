import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom json encoder for handling numpy arrays and convert them to lists"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)