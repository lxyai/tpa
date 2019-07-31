from random import shuffle
import pandas as pd
import numpy as np


class Field:
    def __init__(self):
        self.length = 0
        self.st = 0
        self.ed = 0
        self.values = None


class TimeSeriesDataset:
    def __init__(self, para, df):
        if not isinstance(para, dict):
            raise TypeError('para must be dict like type!')
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be DataFrame!')
        if len(df) <= 0:
            raise ValueError('df is empty!')

        self.timesteps = len(df)
        self.para = para
        x1, x2 = self.para['x']['range'][0], self.para['x']['range'][1]
        y1, y2 = self.para['y']['range'][0], self.para['y']['range'][1]
        if 'z' in self.para.keys():
            z1, z2 = self.para['z']['range'][0], self.para['z']['range'][1]
            left = min(x1, y1, z1)
            right = max(x2, y2, z2)
        else:
            left = min(x1, y1)
            right = max(x2, y2)

        distance = right - left
        start_point = 0
        end_point = self.timesteps - distance - 1
        self.n_sample = self.timesteps - right + left
        print('number of sample is {}...'.format(self.n_sample))

        x = Field()
        x.length = x2 - x1 + 1
        x.st = start_point + (x1 - left)
        x.ed = end_point + (x1 - left)
        x.values = df[self.para['x']['key']].values
        x_example = []
        for i in range(x.st, x.ed+1, self.para['step']):
            x_example.append(np.expand_dims(x.values[i: i + x.length], axis=0))
        x.values = np.concatenate(x_example, axis=0)

        y = Field()
        y.length = y2 - y1 + 1
        y.st = start_point + (y1 - left)
        y.ed = end_point + (y1 - left)
        y.values = df[self.para['y']['key']].values
        y_example = []
        for i in range(y.st, y.ed, self.para['step']):
            y_example.append(np.expand_dims(y.values[i: i + y.length], axis=0))
        y.values = np.concatenate(y_example, axis=0)

        if 'z' in self.para.keys():
            z = Field()
            z.length = z2 - z1 + 1
            z.st = start_point + (z1 - left)
            z.ed = end_point + (z1 - left)
            z.values = df[self.para['z']['key']].values
            z_example = []
            for i in range(z.st, z.ed, self.para['step']):
                z_example.append(np.expand_dims(z.values[i: i + z.length], axis=0))
            z.values = np.concatenate(z_example, axis=0)

        zip_list = []

        for i in [x, y, z]:
            if i is None:
                continue
            zip_list.append(i.values)

        dataset = []
        for data in zip(*zip_list):
            dataset.append(list(data))

        # if self.para['shuffle']:
        #     shuffle(dataset)

        data_size = len(dataset)
        split = [int(i * data_size) for i in self.para.split]
        self.dataset = []
        for i in range(3):
            self.dataset.append(dataset[split[i]:split[i + 1]])

        self.train_initalizer()
        self.validation_initializer()
        self.test_initalizer()

    def train_initalizer(self):
        train = self.dataset[0].copy()
        if self.para['shuffle']:
            shuffle(train)
        self.train_status = {'data': train, 'idx': 0, 'epoch': 0, 'size': len(train)}

    def validation_initializer(self):
        validation = self.dataset[1].copy()
        self.validation_status = {'data': validation, 'idx': 0, 'stop': False, 'size': len(validation)}

    def test_initalizer(self):
        test = self.dataset[2].copy()
        self.test_status = {'data': test, 'idx': 0, 'stop': False, 'size': len(test)}

    def get_train_batch(self):
        batch_size = self.para['batch_size']
        idx = self.train_status['idx']
        if idx + 2 * batch_size > self.train_status['size'] and idx + batch_size <= self.train_status['size']:
            self.train_status['epoch'] += 1
        if idx + batch_size > len(self.train_status['data']):
            idx = 0
            self.train_status['data'] = self.dataset[0].copy()
            if self.para['shuffle']:
                shuffle(self.train_status['data'])

        data = self.train_status['data'][idx:idx + batch_size]
        self.train_status['idx'] = idx + batch_size
        batch_data = []
        for n in zip(*data):
            batch_data.append(n)
        if len(batch_data) == 2:
            x = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[0]])
            y = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[1]])
            return x, y
        if len(batch_data) == 3:
            x = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[0]])
            y = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[1]])
            z = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[2]])
            return x, y, z
        else:
            raise ValueError('label and feature numbers is not support!')

    def get_validation_batch(self):
        batch_size = self.para['batch_size']
        idx = self.validation_status['idx']
        if idx + 2 * batch_size > len(self.validation_status['data']):
            self.validation_status['stop'] = True
        data = self.validation_status['data'][idx:idx + batch_size]
        self.validation_status['idx'] = idx + batch_size
        batch_data = []
        for n in zip(*data):
            batch_data.append(n)
        if len(batch_data) == 2:
            x = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[0]])
            y = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[1]])
            return x, y
        if len(batch_data) == 3:
            x = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[0]])
            y = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[1]])
            z = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[2]])
            return x, y, z
        else:
            raise ValueError('label and feature numbers is not support!')

    def get_test_batch(self):
        batch_size = 1
        idx = self.test_status['idx']
        if idx + batch_size >= len(self.test_status['data']):
            self.test_status['stop'] = True
        data = self.test_status['data'][idx:idx + batch_size]
        self.test_status['idx'] = idx + batch_size
        batch_data = []
        for n in zip(*data):
            batch_data.append(n)
        if len(batch_data) == 2:
            x = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[0]])
            y = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[1]])
            return x, y
        if len(batch_data) == 3:
            x = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[0]])
            y = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[1]])
            z = np.concatenate([np.expand_dims(i, axis=0) for i in batch_data[2]])
            return x, y, z
        else:
            raise ValueError('label and feature numbers is not support!')
