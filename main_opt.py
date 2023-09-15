import pdb
import argparse
import numpy as np
from models.classifiers import KNN
from models.classifiers import decision_tree
from models.classifiers import MLP

from models.optimizer import Bayes_optimizer
from utils import load_data
from utils import evaluation

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="knn",
    help="The model you want to used (default: knn)"
)

parser.add_argument(
    "--l1",
    type=float,
    help="Relative importance of model accuracy"
)

parser.add_argument(
    "--l2",
    type=float,
    help="Relative importance of train time"
)

parser.add_argument(
    "--l3",
    type=float,
    help="Relative importance of infer time"
)

args = parser.parse_args()

model = args.model
l1 = args.l1
l2 = args.l2
l3 = args.l3

load = load_data(dataset="hdfs", eliminated=True)
x_train_arr, x_test_arr, y_train_arr, y_test_arr = load.reload_data()

p_all = []
r_all = []
f_all =[]
t_time, i_time = [], []

for i in range(5):
    x_train, x_test, y_train, y_test = x_train_arr[i].tolist(), x_test_arr[i].tolist(), y_train_arr[i].tolist(), y_test_arr[i].tolist()
    
    # SLFN
    if model == 'slfn':
        params_range_dict = {"hidden_neurons": (5,100), "solver":(0,3),\
                    "activation":(0,4), "alpha":(1e-6,1e-1), "tol":(1e-6,1e-1),"max_iter":(50,400)}

        optimizer = Bayes_optimizer(l1,l2,l3,MLP, x_train, y_train, params_range_dict)
        best_params = optimizer.optimize()
        print(best_params)
        labels_pre, train_time, infer_time = MLP(x_train, x_test, y_train, **best_params)
    

    # DT
    if model == 'dt':
        params_range_dict = {'criterion': (0,2), 'min_samples_leaf':(1,5),'max_depth':(5,70),\
            'min_samples_split':(2,5)}

        optimizer = Bayes_optimizer(l1,l2,l3,decision_tree, x_train, y_train, params_range_dict)
        best_params = optimizer.optimize()
        print(best_params)
        labels_pre, train_time, infer_time = decision_tree(x_train, x_test, y_train, **best_params)

    # KNN
    if model == 'knn':
        params_range_dict = {'n_neighbors': (1,11), 'metric':(0,3)}
        optimizer = Bayes_optimizer(l1,l2,l3,KNN, x_train, y_train, params_range_dict)
        best_params = optimizer.optimize()
        print(best_params)
        labels_pre, train_time, infer_time = KNN(x_train, x_test, y_train, **best_params)

    t_time.append(train_time/len(x_train))
    i_time.append(infer_time/len(x_test))

    evaluate = evaluation(y_test,labels_pre)
    matched = []
    print("Testing process finished!!!")
    _, _, f, _, _ = evaluate.calc_metrics()
    #p_all.append(p)
    #r_all.append(r)
    f_all.append(f)

print("Current Model:",model+";","ModelScore:",l1*(np.mean(f_all)-0.8)/0.2-l2*(np.mean(t_time))/3e-2-l3*(np.mean(i_time))/2e-3)

