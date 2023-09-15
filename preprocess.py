from utils import load_data

#FIXME: add argparse

#load = load_data(dataset="spirit")
#load.load_and_split(data_range=[0,7983345],train_ratio=0.8)


#load = load_data(dataset="tbird")
#load.load_and_split(data_range=[0,10000000],train_ratio=0.8)

#load = load_data(dataset="liberty")
#load.load_and_split(data_range=[0,10000000],train_ratio=0.8)

#load = load_data(dataset="bgl")
#load.load_and_split(data_range=[0,4747963],train_ratio=0.8)

load = load_data(dataset="hdfs", eliminated=False)
load.load_and_split_hdfs(train_ratio=0.8,shuffle=True,times=5)