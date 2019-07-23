from random import shuffle
import numpy as np
import pandas as pd


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

        x = y = z = None

        if 'x' not in para.keys():
            raise ValueError('x is not exist')
        if 'y' not in para.keys():
            raise ValueError('y is not exist')

        x = Field()
        x.length = para['x']['range'][1] - para['x']['range'][0] + 1
        x.st = para['x']['range'][0] - para['z']['range'][0] \
            if 'z' in para.keys() and para['x']['range'][0] <= para['z']['range'][0] else 0
        x.ed = self.timesteps - (para['y']['range'][1] - para['x']['range'][0])
        x.values = df[para['x']['key']].values
        x_examples = []
        for i in range(x.st, x.ed, para['step']):
            x_examples.append(np.expand_dims(x.values[i:i + x.length], axis=0))
        x.values = np.concatenate(x_examples, axis=0)

        y = Field()
        y.length = para['y']['range'][1] - para['y']['range'][0] + 1
        y.st = para['y']['range'][0] - para['z']['range'][0] \
            if 'z' in para.keys() and para['x']['range'][0] <= para['z']['range'][0] \
            else para['y']['range'][0] - para['x']['range'][0]
        y.ed = self.timesteps - (para['y']['range'][1] - para['y']['range'][0])
        y.values = df[para['y']['key']].values
        y_examples = []
        for i in range(y.st, y.ed):
            y_examples.append(np.expand_dims(y.values[i:i + y.length], axis=0))
        y.values = np.concatenate(y_examples, axis=0)

        if 'z' in para.keys():
            z = Field()
            z.length = para['z']['range'][1] - para['z']['range'][0] + 1
            z.st = 0 if para['z']['range'][0] <= para['x']['range'][0] \
                else para['z']['range'][0] - para['x']['range'][0]
            z.ed = self.timesteps - (para['y']['range'][1] - para['z']['range'][0])
            z.values = df[para['z']['key']].values
            z_examples = []
            for i in range(z.st, z.ed):
                z_examples.append(np.expand_dims(z[i:i + z.length], axis=0))
            z.values = np.concatenate(z_examples, axis=0)

        zip_list = []

        for i in [x, y, z]:
            if i is None:
                continue
            zip_list.append(i.values)

        dataset = []
        for data in zip(*zip_list):
            dataset.append(list(data))

        data_size = len(dataset)
        split = [int(i * data_size) for i in self.para.split]
        self.dataset = []
        for i in range(3):
            self.dataset.append(dataset[split[i]:split[i + 1]])

        self.train_initalizer()
        self.validation_initalizer()
        self.test_initalizer()

    def train_initalizer(self):
        train = self.dataset[0].copy()
        if self.para['shuffle']:
            shuffle(train)
        self.train_status = {'data': train, 'idx': 0, 'epoch': 0}

    def validation_initalizer(self):
        validation = self.dataset[1].copy()
        self.validation_status = {'data': validation, 'idx': 0, 'stop': False}

    def test_initalizer(self):
        test = self.dataset[2].copy()
        self.test_status = {'data': test, 'idx': 0, 'stop': False}

    def get_train_batch(self):
        batch_size = self.para['batch_size']
        idx = self.train_status['idx']
        if idx + 2 * batch_size > len(self.train_status['data']) and idx + batch_size <= len(self.train_status['data']):
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
            return

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


class electricity(TimeSeriesDataset):
    def __init__(self, para, df):
        self.df = df.copy()
        input_size = df.values.shape[1]
        dat = np.zeros(df.values.shape, np.float32)
        self.scale = np.zeros(input_size, np.float32)
        self.mn = np.zeros(input_size, np.float32)
        for i in range(input_size):
            self.mn[i] = np.min(df.values[:, i])
            self.scale[i] = np.max(df.values[:, i]) - self.mn[i]
            dat[:, i] = (df.values[:, i] - self.mn[i]) / self.scale[i]
        norm_df = pd.DataFrame(dat, columns=df.columns)

        super(electricity, self).__init__(para, norm_df)

        validation = self.validation_status['data'].copy()
        y = None
        for i, val in enumerate(zip(*validation)):
            if i != 1:
                continue
            y = np.concatenate(val, axis=0)
        self.validation_rse = np.std(y * self.scale + self.mn)
