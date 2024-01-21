import argparse
from utils import load_data
from utils import evaluation
from models.match import semantics_match
from models.classifiers import MLP
from models.classifiers import KNN
from models.classifiers import decision_tree
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="tbird",
    help="The supercomputer dataset you want to use (default: bgl)"
)

args = parser.parse_args()
dataset = args.dataset


load = load_data(dataset=dataset)
x_train, x_test, y_train, y_test = load.reload_data()
x_train, x_test, y_train, y_test = x_train[0], x_test[0], y_train[0], y_test[0]

idx_list = np.random.choice(len(x_train), int(0.1*len(x_train)), replace=False)
xx_train = []
yy_train = []
for idx in idx_list:
    xx_train.append(x_train[idx])
    yy_train.append(y_train[idx])
x_train = xx_train
y_train = yy_train       

print(len(x_train))

model = semantics_match()
labels_pre,matched, train_time, infer_time = model.run(x_train, x_test, y_train, dist_type='Jaccard')
evaluate = evaluation(y_test,labels_pre)

print("Testing process finished!!!")
p, r, f, s, b = evaluate.calc_with_windows(window=10,step=10)
 
print("Currently Used Dataset:",dataset+";","Precision:",p,"Recall:",r,"F1-score:",f,"Specificity:",s,"Banlanced Acc: ",b)
