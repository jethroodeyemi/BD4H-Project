import os
from io import open
import torch
import numpy

class TimeseriesTorch(object):
    def __init__(self, path, num_features):
        self.train = self.load_series(os.path.join(path, 'train.dat'), num_features)
        self.valid = self.load_series(os.path.join(path, 'dev.dat'), num_features)
        self.test = self.load_series(os.path.join(path, 'test.dat'), num_features)

    def load_series(self, path, num_features):
        with open(path, 'r', encoding="utf8") as f:
            timesteps = sum(1 for _ in f)

        with open(path, 'r', encoding="utf8") as f:
            steps = torch.Tensor(timesteps, num_features)
            targets = torch.Tensor(timesteps)
            for pos, line in enumerate(f):
                healthdata = line.split()
                values = [float(value) for value in healthdata[4:4+num_features]]
                steps[pos] = torch.from_numpy(numpy.array(values))
                targets[pos] = float(healthdata[2])
        return [steps, targets]

class TimeseriesNumPy(object):
    def __init__(self, path, num_features):
        self.train = self.load_series(os.path.join(path, 'train.dat'), num_features)
        self.valid = self.load_series(os.path.join(path, 'dev.dat'), num_features)
        self.test = self.load_series(os.path.join(path, 'test.dat'), num_features)

    def load_series(self, path, num_features):
        stepsl = []
        targetsl = []
        
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                if not line.strip(): continue
                healthdata = line.split()
                try: float(healthdata[0]) 
                except ValueError: continue 
                values = [float(value) for value in healthdata[4:4+num_features]]
                stepsl.append(values)
                targetsl.append(float(healthdata[2]))
        steps = numpy.array(stepsl, dtype=float)
        targets = numpy.array(targetsl, dtype=float)
        return [steps, targets]