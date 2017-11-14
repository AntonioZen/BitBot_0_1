import pandas as pd
from colorama import Fore
from BitBot_0_1 import  CPM1 as cpm
from BitBot_0_1 import Zoo
from sklearn.neural_network import MLPRegressor
from bokeh.plotting import figure, show, output_file
from math import pi
# ------------------

def simulation2():

    coin1 = 'USDT'
    coin2 = 'BTC'
    pair = coin1 + '-' + coin2
    # df = pd.DataFrame(read_jsonline(datasource))
    # df.to_csv(datacsv)
    datacsv = 'C:/BitBot/data_test_9/%s_output.csv' % pair
    path = 'C:/BitBot/BitBot_0_1/'
    # datatest = 'C: /BitBot/data_test_2/BTC-BTS.csv'


    df = pd.read_csv(datacsv)

    # estimation_period = 1128

    high = 0.005
    low = -0.005
    coef = 0.05
    wallet1 = 1
    wallet2 = 0
    name = 'Bob'
    X_list = ['mean','value_from','value_to','macd_mean26','macd_mean12','macd_mean9','metric']
    Y = 'future_signal'
    #Y = 'target'

    dataset_size = 2000
    observation_size = 100
    future_index = 5
    estimation_period =300

    Monkey1 = Zoo.TradeMonkey2(name, datacsv, X_list, Y, observation_size, wallet1, wallet2, coef, metric='metric')
    Monkey1.get_data(datacsv)
    #Monkey1.create_dataset(10, 1)

    #dataset = Monkey1.create_dataset(1000, 3)
    answers = pd.DataFrame()

    for i in range(estimation_period):

        Monkey1.see(dataset_size, future_index, estimation_period - i)
        monkey_shit = Monkey1.do()
        answers = answers.append(monkey_shit, ignore_index=True)
        print(Fore.CYAN)
        print(answers)

    answers.to_csv('C:/BitBot/' + 'Monkey_test_ds500_os500_fi5_test5.csv')
    return answers


def simulation3():

    coin1 = 'USDT'
    coin2 = 'BTC'
    pair = coin1 + '-' + coin2
    # df = pd.DataFrame(read_jsonline(datasource))
    # df.to_csv(datacsv)
    datacsv = 'C:/BitBot/data_test_9/%s_output.csv' % pair
    path = 'C:/BitBot/BitBot_0_1/'
    # datatest = 'C: /BitBot/data_test_2/BTC-BTS.csv'


    df = pd.read_csv(datacsv)

    # estimation_period = 1128

    high = 0.005
    low = -0.005
    coef = 0.05
    wallet1 = 1
    wallet2 = 0
    name = 'Bob'
    X_list = ['mean','value_from','value_to','macd_mean26','macd_mean12','macd_mean9','metric'] # ['mean',
    Y = 'future_signal'
    #Y = 'target'

    dataset_size = 2000
    observation_size = 100
    future_index = 5
    estimation_period =300

    Monkey1 = Zoo.TradeMonkey3(name, datacsv, X_list, Y, observation_size, wallet1, wallet2, coef, metric='metric')
    Monkey1.get_data(datacsv)
    #Monkey1.create_dataset(10, 1)

    #dataset = Monkey1.create_dataset(1000, 3)
    answers = pd.DataFrame()

    for i in range(estimation_period):

        Monkey1.see(dataset_size, future_index, estimation_period - i)
        monkey_shit = Monkey1.do()
        answers = answers.append(monkey_shit, ignore_index=True)
        print(Fore.CYAN)
        print(answers)

    answers.to_csv('C:/BitBot/' + 'Monkey_test_ds500_os500_fi5_test6.csv')
    return answers


def simulation_cpm():

    coin1 = 'USDT'
    coin2 = 'BTC'
    pair = coin1 + '-' + coin2
    # df = pd.DataFrame(read_jsonline(datasource))
    # df.to_csv(datacsv)
    datacsv = 'C:/BitBot/data_test_10/%s_output.csv' % pair
    path = 'C:/BitBot/BitBot_0_1/'
    # datatest = 'C: /BitBot/data_test_2/BTC-BTS.csv'


    # df = pd.read_csv(datacsv, sep = '')

    # estimation_period = 1128

    high = 0.005
    low = -0.005
    coef = 0.05
    wallet1 = 1
    wallet2 = 0
    name = 'Bob'
    X_list = ['mean','value_from','value_to','macd_mean26','macd_mean12','macd_mean9','metric'] # ['mean',
    Y = 'future_signal'
    #Y = 'target'

    dataset_size = 2000
    observation_size = 200
    future_index = 5
    estimation_period = 1000

    Monkey1 = Zoo.TradeMonkey4(name, datacsv, X_list, Y, observation_size, wallet1, wallet2, coef, metric='metric')
    Monkey1.get_data(datacsv)
    #Monkey1.create_dataset(10, 1)

    #dataset = Monkey1.create_dataset(1000, 3)
    answers = pd.DataFrame()

    for i in range(estimation_period):

        Monkey1.see(dataset_size, future_index, estimation_period - i)
        monkey_shit = Monkey1.do()
        answers = answers.append(monkey_shit, ignore_index=True)
        print(Fore.CYAN)
        print(answers)

    answers.to_csv('C:/BitBot/' + 'Monkey_test_ds500_os500_cusum_engine_test2.csv')
    return answers


def visualize_frame(df):

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    #s = file
    p = figure(tools=TOOLS, plot_width=1000, title = ' file: ' + 'name') # , x_axis_type="datetime")
    p.xaxis.major_label_orientation = pi/4
    p.grid.grid_line_alpha=0.1

    p.line(df.index.values,df.target, line_width=1, color = '#a1d99b')
    p.line(df.index.values, df.predicted, line_width=1, color='#fc9272')
    show(p)


# Monkey1.create_row(df)

# print(Fore.LIGHTRED_EX)
# print(Monkey1.data.tail(10))

# 1229  4280.800
# 1228  4270.957
# 1227  4280.000
# 1226  4284.000
# 1225  4250.100