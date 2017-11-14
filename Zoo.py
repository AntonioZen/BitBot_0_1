import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from BitBot_0_1 import  CPM1 as cpm
from colorama import Fore


class TradeMonkey():

    def __init__(self, name, datacsv, X_list, Y, observation_size, wallet1, wallet2, coef, metric='metric'):
        self.name = name
        self.datacsv = datacsv
        self.X_list = X_list
        self.Y = Y
        print('here comes ' + str(name))
        self.data = pd.read_csv(datacsv)
        self.metric = metric
        self.observation_size = observation_size
        self.wallet1 = wallet1
        self.wallet2 = wallet2
        self.coef = coef
        # branch e.t.c
        pass

    def get_data(self, datacsv, past_index = 0):

        data = pd.read_csv(datacsv)
        self.data = data.iloc[: len(data) - past_index + 1]


    def create_row(self, observation):

        #print('monkey look')

        row = {}

        series_view = cpm.SeriesView(observation[self.metric], 0.5, 0.9)

        sv_data = series_view.getData()
        print(Fore.LIGHTRED_EX)
        print(sv_data)
        print(Fore.GREEN)
        mean = observation[self.metric].mean()
        dataset_row = sv_data.iloc[len(sv_data) - 1, :]

        row['mean'] = mean
        row['mean_in_range'] = dataset_row['mean_in_range']
        row['metric'] = dataset_row['metric']


        return row

    def target_row(self, row, future_index):

        target = float()

        target = self.data.iloc[row['observation_index'] + future_index][self.metric]

        return target

    def create_dataset(self, size, future_index):

        dataset = pd.DataFrame()
        observation_index = len(self.data)

        for i in range(size):

            observation = self.data.iloc[observation_index - self.observation_size - i - future_index  : observation_index - i - future_index].reset_index(drop=True)

            row = self.create_row(observation)
            row['observation_index'] = int(observation_index - i - 1 - future_index)
            row['target'] = self.target_row(row, future_index)

            #print(i, row)
            dataset = dataset.append(row, ignore_index=True)

        print(dataset)
        return dataset

    def see(self, size, future_index, past_index):
        print(Fore.GREEN)
        print('monkey see')

        self.get_data(self.datacsv, past_index)
        self.dataset = self.create_dataset(size, future_index)
        self.clf = RandomForestRegressor(n_estimators=500, max_features='sqrt', max_depth=10, min_samples_leaf=10,
                                    random_state=0)

        self.clf.fit(self.dataset[self.X_list], self.dataset[self.Y])

    def do(self):
        print(Fore.LIGHTYELLOW_EX)
        print('monkey do')

        horizont = self.create_dataset(1, 0)
        horizont['predicted'] = self.clf.predict(horizont[self.X_list])

        return horizont


class TradeMonkey2():
    def __init__(self, name, datacsv, X_list, Y, observation_size, wallet1, wallet2, coef, metric='metric'):
        self.name = name
        self.datacsv = datacsv
        self.X_list = X_list
        self.Y = Y
        print('here comes ' + str(name))
        self.data = pd.read_csv(datacsv)
        self.metric = metric
        self.observation_size = observation_size
        self.wallet1 = wallet1
        self.wallet2 = wallet2
        self.coef = coef
        # branch e.t.c
        pass

    def get_data(self, datacsv, past_index=0):
        data = pd.read_csv(datacsv)
        self.data = data.iloc[: len(data) - past_index + 1]

    def create_row(self, observation):
        # print('monkey look')

        dataset_row = {}

        series_view = cpm.SeriesView(observation[self.metric], 0.5, 0.9)
        sv_data = series_view.getData()
        mean = observation[self.metric].mean()
        macd_mean26 =observation.tail(26)[self.metric].mean()
        macd_mean12 = observation.tail(12)[self.metric].mean()
        macd_mean9 = observation.tail(9)[self.metric].mean()
        cpcount = 0
        # ------------------
        obdata = {}
        cpdata = {}
        for i, row in sv_data.iterrows():
            if i == 0:
                pass
            elif i == len(sv_data) - 1:
                obdata = {
                    # "observer_id": self.observer.get('id'),
                    "observation_index": i,
                    # "property_value": column,
                    # "confidence_interval": confidence_interval,
                    'change_point_index': i,
                    'value_from': sv_data.ix[i - 1, 'mean_in_range'],
                    'value_to': sv_data.ix[i, 'mean_in_range'],
                    'signal_0': sv_data.ix[i - 1, 'metric'],
                    'signal_1': sv_data.ix[i, 'metric'],
                    'metric_alpha_1': sv_data.ix[i, 'metric_alpha_1'],
                    'observation_mean': mean,
                    'change_point_weight': row['change_point_weight'],
                    'mean_delta_percent': row['mean_delta_percent'],
                    'mean_in_range_running': row['mean_in_range_running']

                    # 'series_view_index_value': row['series_view_index_value']
                }

            elif row['change_point'] == 1:
                cpcount = cpcount + 1
                # print(i)
                # print(Fore.GREEN)
                cpdata = {
                    # "observer_id": self.observer.get('id'),
                    "observation_index": i,
                    # "property_value": column,
                    # "confidence_interval": confidence_interval,
                    'change_point_index': i,
                    'value_from': sv_data.ix[i - 1, 'mean_in_range'],
                    'value_to': sv_data.ix[i, 'mean_in_range'],
                    'signal_0': sv_data.ix[i - 1, 'metric'],
                    'signal_1': sv_data.ix[i, 'metric'],
                    'metric_alpha_1': sv_data.ix[i, 'metric_alpha_1'],
                    'observation_mean': mean,
                    'change_point_weight': row['change_point_weight'],
                    'mean_delta_percent': row['mean_delta_percent'],
                    'mean_in_range_running': row['mean_in_range_running']
                    # 'series_view_index_value': row['series_view_index_value']
                }

        dataset_row['mean'] = mean

        dataset_row['value_to'] = obdata['value_to']
        dataset_row['value_from'] = cpdata['value_from']
        dataset_row['macd_mean26'] = macd_mean26
        dataset_row['macd_mean12'] = macd_mean12
        dataset_row['macd_mean9'] = macd_mean9
        dataset_row['metric'] = obdata['signal_1']

        # print(dataset_row)
        # print(obdata)
        # print(cpdata)

        return dataset_row

    def target_row(self, row, future_index):
        target = float()

        target = self.data.iloc[row['observation_index'] + future_index][self.metric]

        return target

    def create_dataset(self, size, future_index):
        dataset = pd.DataFrame()
        observation_index = len(self.data)

        for i in range(size):
            observation = self.data.iloc[
                          observation_index - self.observation_size - i - future_index: observation_index - i - future_index].reset_index(
                drop=True)

            row = self.create_row(observation)
            row['observation_index'] = int(observation_index - i - 1 - future_index)
            row['target'] = self.target_row(row, future_index)
            row['future_signal'] = (row['target'] - row['metric'])/row['metric']

            # print(i, row)
            dataset = dataset.append(row, ignore_index=True)

        print(Fore.LIGHTRED_EX)
        print(dataset)

        return dataset

    def see(self, size, future_index, past_index):
        print(Fore.GREEN)
        print('monkey see')

        self.get_data(self.datacsv, past_index)
        self.dataset = self.create_dataset(size, future_index)
        self.clf = RandomForestRegressor(n_estimators=500, max_features='sqrt', max_depth=10, min_samples_leaf=10,
                                          random_state=0)
        # self.clf = MLPRegressor(hidden_layer_sizes=(8, 8, 2),
        #          activation='tanh', solver='lbfgs')

        self.clf.fit(self.dataset[self.X_list], self.dataset[self.Y])

    def do(self):
        print(Fore.LIGHTYELLOW_EX)
        print('monkey do')

        horizont = self.create_dataset(1, 0)
        horizont['predicted'] = self.clf.predict(horizont[self.X_list])

        return horizont


class TradeMonkey3():
    def __init__(self, name, datacsv, X_list, Y, observation_size, wallet1, wallet2, coef, metric='metric'):
        self.name = name
        self.datacsv = datacsv
        self.X_list = X_list
        self.Y = Y
        print('here comes ' + str(name))
        self.data = pd.read_csv(datacsv)
        self.metric = metric
        self.observation_size = observation_size
        self.wallet1 = wallet1
        self.wallet2 = wallet2
        self.coef = coef
        # branch e.t.c
        pass

    def get_data(self, datacsv, past_index=0):
        data = pd.read_csv(datacsv)
        self.data = data.iloc[: len(data) - past_index + 1]

    def create_row(self, observation):
        # print('monkey look')

        dataset_row = {}

        series_view = cpm.SeriesView(observation[self.metric], 0.5, 0.8)
        sv_data = series_view.getData()
        mean = observation[self.metric].mean()
        macd_mean26 =observation.tail(26)[self.metric].mean()
        macd_mean12 = observation.tail(12)[self.metric].mean()
        macd_mean9 = observation.tail(9)[self.metric].mean()
        cpcount = 0
        # ------------------
        obdata = {}
        cpdata = {}
        for i, row in sv_data.iterrows():
            if i == 0:
                pass
            elif i == len(sv_data) - 1:
                obdata = {
                    # "observer_id": self.observer.get('id'),
                    "observation_index": i,
                    # "property_value": column,
                    # "confidence_interval": confidence_interval,
                    'change_point_index': i,
                    'value_from': sv_data.ix[i - 1, 'mean_in_range'],
                    'value_to': sv_data.ix[i, 'mean_in_range'],
                    'signal_0': sv_data.ix[i - 1, 'metric'],
                    'signal_1': sv_data.ix[i, 'metric'],
                    'metric_alpha_1': sv_data.ix[i, 'metric_alpha_1'],
                    'observation_mean': mean,
                    'change_point_weight': row['change_point_weight'],
                    'mean_delta_percent': row['mean_delta_percent'],
                    'mean_in_range_running': row['mean_in_range_running']

                    # 'series_view_index_value': row['series_view_index_value']
                }

            elif row['change_point'] == 1:
                cpcount = cpcount + 1
                # print(i)
                # print(Fore.GREEN)
                cpdata = {
                    # "observer_id": self.observer.get('id'),
                    "observation_index": i,
                    # "property_value": column,
                    # "confidence_interval": confidence_interval,
                    'change_point_index': i,
                    'value_from': sv_data.ix[i - 1, 'mean_in_range'],
                    'value_to': sv_data.ix[i, 'mean_in_range'],
                    'signal_0': sv_data.ix[i - 1, 'metric'],
                    'signal_1': sv_data.ix[i, 'metric'],
                    'metric_alpha_1': sv_data.ix[i, 'metric_alpha_1'],
                    'observation_mean': mean,
                    'change_point_weight': row['change_point_weight'],
                    'mean_delta_percent': row['mean_delta_percent'],
                    'mean_in_range_running': row['mean_in_range_running']
                    # 'series_view_index_value': row['series_view_index_value']
                }

        dataset_row['mean'] = mean

        dataset_row['value_to'] = obdata['value_to']
        dataset_row['value_from'] = cpdata['value_from']
        dataset_row['macd_mean26'] = macd_mean26
        dataset_row['macd_mean12'] = macd_mean12
        dataset_row['macd_mean9'] = macd_mean9
        dataset_row['metric'] = obdata['signal_1']

        # print(dataset_row)
        # print(obdata)
        # print(cpdata)

        return dataset_row

    def target_row(self, row, future_index):
        target = float()

        target = self.data.iloc[row['observation_index'] + future_index][self.metric]

        return target

    def create_dataset(self, size, future_index):
        dataset = pd.DataFrame()
        observation_index = len(self.data)

        for i in range(size):
            observation = self.data.iloc[
                          observation_index - self.observation_size - i - future_index: observation_index - i - future_index].reset_index(
                drop=True)

            row = self.create_row(observation)
            row['observation_index'] = int(observation_index - i - 1 - future_index)
            row['target'] = self.target_row(row, future_index)
            row['future_signal'] = (row['target'] - row['metric'])/row['metric']

            # print(i, row)
            dataset = dataset.append(row, ignore_index=True)

        print(Fore.LIGHTRED_EX)
        print(dataset)

        return dataset

    def normalize(self, dataset):

        dataset_tmp = pd.DataFrame()
        dataset_tmp['future_signal'] = dataset['future_signal']
        for X in self.X_list:
            dataset_tmp[X] = dataset[X] - dataset['mean']

        print(Fore.LIGHTYELLOW_EX)
        print(dataset_tmp)
        return dataset_tmp


    def see(self, size, future_index, past_index):
        print(Fore.GREEN)
        print('monkey see')

        self.get_data(self.datacsv, past_index)
        dataset_tmp = self.normalize(self.create_dataset(size, future_index))

        self.clf = RandomForestRegressor(n_estimators=500, max_features='sqrt', max_depth=10, min_samples_leaf=10,
                                          random_state=0)
        # self.clf = MLPRegressor(hidden_layer_sizes=(8, 8, 2),
        #          activation='tanh', solver='lbfgs')

        self.clf.fit(dataset_tmp[self.X_list], dataset_tmp[self.Y])

    def do(self):
        print(Fore.LIGHTYELLOW_EX)
        print('monkey do')

        horizont = self.create_dataset(1, 0)
        horizont['predicted'] = self.clf.predict(self.normalize(horizont)[self.X_list])

        return horizont


class TradeMonkey4():
    def __init__(self, name, datacsv, X_list, Y, observation_size, wallet1, wallet2, coef, metric='metric'):
        self.name = name
        self.datacsv = datacsv
        self.X_list = X_list
        self.Y = Y
        print('here comes ' + str(name))
        self.data = pd.read_csv(datacsv)
        self.metric = metric
        self.observation_size = observation_size
        self.wallet1 = wallet1
        self.wallet2 = wallet2
        self.coef = coef
        # branch e.t.c
        pass

    def get_data(self, datacsv, past_index=0):
        data = pd.read_csv(datacsv)
        self.data = data.iloc[: len(data) - past_index + 1]

    def create_row(self, observation):
        # print('monkey look')

        dataset_row = {}

        series_view = cpm.SeriesView(observation[self.metric], 0.5, 0.8)
        sv_data = series_view.getData()
        mean = observation[self.metric].mean()
        macd_mean26 = observation.tail(26)[self.metric].mean()
        macd_mean12 = observation.tail(12)[self.metric].mean()
        macd_mean9 = observation.tail(9)[self.metric].mean()
        cpcount = 0
        # ------------------
        obdata = {}
        cpdata = {}
        for i, row in sv_data.iterrows():
            if i == 0:
                pass
            elif i == len(sv_data) - 1:
                obdata = {
                    # "observer_id": self.observer.get('id'),
                    "observation_index": i,
                    # "property_value": column,
                    # "confidence_interval": confidence_interval,
                    'change_point_index': i,
                    'value_from': sv_data.ix[i - 1, 'mean_in_range'],
                    'value_to': sv_data.ix[i, 'mean_in_range'],
                    'signal_0': sv_data.ix[i - 1, 'metric'],
                    'signal_1': sv_data.ix[i, 'metric'],
                    'metric_alpha_1': sv_data.ix[i, 'metric_alpha_1'],
                    'observation_mean': mean,
                    'change_point_weight': row['change_point_weight'],
                    'mean_delta_percent': row['mean_delta_percent'],
                    'mean_in_range_running': row['mean_in_range_running']

                    # 'series_view_index_value': row['series_view_index_value']
                }

            elif row['change_point'] == 1:
                cpcount = cpcount + 1
                # print(i)
                # print(Fore.GREEN)
                cpdata = {
                    # "observer_id": self.observer.get('id'),
                    "observation_index": i,
                    # "property_value": column,
                    # "confidence_interval": confidence_interval,
                    'change_point_index': i,
                    'value_from': sv_data.ix[i - 1, 'mean_in_range'],
                    'value_to': sv_data.ix[i, 'mean_in_range'],
                    'signal_0': sv_data.ix[i - 1, 'metric'],
                    'signal_1': sv_data.ix[i, 'metric'],
                    'metric_alpha_1': sv_data.ix[i, 'metric_alpha_1'],
                    'observation_mean': mean,
                    'change_point_weight': row['change_point_weight'],
                    'mean_delta_percent': row['mean_delta_percent'],
                    'mean_in_range_running': row['mean_in_range_running']
                    # 'series_view_index_value': row['series_view_index_value']
                }

        dataset_row['mean'] = mean

        dataset_row['value_to'] = obdata['value_to']
        dataset_row['value_from'] = cpdata['value_from']
        dataset_row['macd_mean26'] = macd_mean26
        dataset_row['macd_mean12'] = macd_mean12
        dataset_row['macd_mean9'] = macd_mean9
        dataset_row['metric'] = obdata['signal_1']
        dataset_row['metric_alpha_1'] = obdata['metric_alpha_1']
        # print(dataset_row)
        # print(obdata)
        # print(cpdata)

        return dataset_row

    def target_row(self, row, future_index):
        target = float()

        target = self.data.iloc[row['observation_index'] + future_index][self.metric]

        return target

    def create_dataset(self, size, future_index):
        dataset = pd.DataFrame()
        observation_index = len(self.data)

        for i in range(size):
            observation = self.data.iloc[
                          observation_index - self.observation_size - i - future_index: observation_index - i - future_index].reset_index(
                drop=True)

            row = self.create_row(observation)
            row['observation_index'] = int(observation_index - i - 1 - future_index)
            # row['target'] = self.target_row(row, future_index)
            # row['future_signal'] = (row['target'] - row['metric'])/row['metric']

            # print(i, row)
            dataset = dataset.append(row, ignore_index=True)

        print(Fore.LIGHTRED_EX)
        print(dataset)

        return dataset

    def normalize(self, dataset):

        dataset_tmp = pd.DataFrame()
        dataset_tmp['future_signal'] = dataset['future_signal']
        for X in self.X_list:
            dataset_tmp[X] = dataset[X] - dataset['mean']

        print(Fore.LIGHTYELLOW_EX)
        print(dataset_tmp)
        return dataset_tmp


    def see(self, size, future_index, past_index):
        print(Fore.GREEN)
        print('monkey see')

        self.get_data(self.datacsv, past_index)
        # dataset_tmp = self.normalize(self.create_dataset(size, future_index))

        # self.clf = RandomForestRegressor(n_estimators=500, max_features='sqrt', max_depth=10, min_samples_leaf=10,
        #                                   random_state=0)
        # self.clf = MLPRegressor(hidden_layer_sizes=(8, 8, 2),
        #          activation='tanh', solver='lbfgs')

        # self.clf.fit(dataset_tmp[self.X_list], dataset_tmp[self.Y])

    def do(self):
        print(Fore.LIGHTYELLOW_EX)
        print('monkey do')

        horizont = self.create_dataset(1, 0)
        # horizont['predicted'] = self.clf.predict(self.normalize(horizont)[self.X_list])

        return horizont
