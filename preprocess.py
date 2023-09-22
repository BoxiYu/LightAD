from utils import load_data
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="hdfs",
    choices = ["hdfs", "spirit", "tbird", "liberty", "bgl"],
    help="The dataset you want to preprocess"
)

parser.add_argument(
    "--eliminate",
    type=bool,
    default=False,
    help="Whether to deduplicate the data"
)

args = parser.parse_args()
dataset = args.dataset
eliminate = args.eliminate

if dataset == "spirit":
    load = load_data(dataset="spirit")
    load.load_and_split(data_range=[0,7983345],train_ratio=0.8)

if dataset == "tbird":
    load = load_data(dataset="tbird")
    load.load_and_split(data_range=[0,10000000],train_ratio=0.8)

if dataset == "liberty":
    load = load_data(dataset="liberty")
    load.load_and_split(data_range=[0,10000000],train_ratio=0.8)

if dataset == "bgl":
    load = load_data(dataset="bgl")
    load.load_and_split(data_range=[0,4747963],train_ratio=0.8)

if dataset == "hdfs":
    if eliminate:
        load = load_data(dataset="hdfs", eliminated=True)
        load.load_and_split_hdfs(train_ratio=0.8,times=1)
    else:
        load = load_data(dataset="hdfs", eliminated=False)
        load.load_and_split_hdfs(train_ratio=0.8,times=5)