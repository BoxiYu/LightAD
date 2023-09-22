import argparse
import numpy as np
from models.classifiers import MLP
from models.classifiers import KNN
from models.classifiers import decision_tree
from utils import evaluation

#FIXME: use reload_data

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="knn",
    help="The model you want to used (default: knn)"
)

parser.add_argument(
    "--eliminate",
    type=str,
    default=False,
    help="Whether you want to run the model on eliminated dataset"
)

args = parser.parse_args()

model = args.model
eliminate = args.eliminate

x_train_arr, x_test_arr, y_train_arr, y_test_arr = [], [], [], []
#load = load_data(dataset="hdfs", vectorized=True, eliminated=eliminate)

data_len = 5

for i in range(data_len):
    data = np.load(f"/mnt/ssd1/yjy/log_icse/simple_methods/splited_datasets/hdfs/shuffle_{i}.npz", allow_pickle=True)
    x_train_arr.append(data["x_train"])
    y_train_arr.append(data["y_train"])
    x_test_arr.append(data["x_test"])
    y_test_arr.append(data["y_test"])




p_all = []
r_all = []
f_all = []
s_all = []
b_all = []
p_all_train = []
r_all_train = []
f_all_train = []
time_train_all = []
time_infer_all = []

for i in range(data_len):
    x_train, x_test, y_train, y_test = x_train_arr[i].tolist(), x_test_arr[i].tolist(), y_train_arr[i].tolist(), y_test_arr[i].tolist()
    u = 0
    d = 0
    for x in x_test:
        if x in x_train:
            d += 1
        else:
            u += 1
    print(f"Unique sequences: {u}; Duplicate sequences: {d}")

    # DT
    if model == "dt":
        params = {}
        labels_pre_d, train_time, infer_time = decision_tree(x_train, x_test, y_train, **params)

    # KNN
    if model == "knn":
        params = {"n_neighbors":1}
        labels_pre_d, train_time, infer_time = KNN(x_train, x_test, y_train, **params)
    
    # SLFN
    if model == "slfn":
        params = {"hidden_layer_sizes":(25,)}
        labels_pre_d, train_time, infer_time = MLP(x_train, x_test, y_train, **params)
    
    

    evaluate = evaluation(y_test,labels_pre_d)
    matched = []
    p, r, f, s, b = evaluate.calc_metrics()

    print(p, r, f, s, b)
    print(train_time, infer_time)

    p_all.append(p)
    r_all.append(r)
    f_all.append(f)
    s_all.append(s)
    b_all.append(b)

    time_train_all.append(train_time)
    time_infer_all.append(infer_time)


print("Current Model:",model+";","Precision:",np.mean(p_all),"Recall:",np.mean(r_all),"F1-score:",np.mean(f_all), "Specificity:", np.mean(s_all), "Balanced Acc:", np.mean(b_all))
print(f"Train time: {sum(time_train_all)/5}")
print(f"Inference time: {sum(time_infer_all)/5}")



