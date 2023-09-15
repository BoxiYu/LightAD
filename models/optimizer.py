from utils import evaluation
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score


class Bayes_optimizer():

    def __init__(self, l1, l2, l3, model, train_data, train_labels, params_range_dict):
        self.model = model
        self.params_range_dict = params_range_dict
        self.x = train_data
        self.y = train_labels
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
    
    def cv_split(self, i):
        train_data = []
        train_label = []
        train_data.extend(self.x[:int((1-(i+1)/10)*len(self.x))])
        train_data.extend(self.x[int((1-i/10)*len(self.x)-1):])
        train_label.extend(self.y[:int((1-(i+1)/10)*len(self.x))])
        train_label.extend(self.y[int((1-i/10)*len(self.x)-1):])
        validation_data = self.x[int((1-(i+1)/10)*len(self.x)):int((1-i/10)*len(self.x))]
        validation_label = self.y[int((1-(i+1)/10)*len(self.x)):int((1-i/10)*len(self.x))]

        return train_data, train_label, validation_data, validation_label
    
    def dt_opt(self, criterion, min_samples_leaf, max_depth, min_samples_split):
        if criterion <= 1:
            criterion = 'entropy'
        else:
            criterion = 'gini'
        params_dict = {'criterion':criterion, 'min_samples_leaf':int(min_samples_leaf),\
            'max_depth':int(max_depth),'min_samples_split':int(min_samples_split)}
        val = []
        t_time = []
        i_time = []
        for i in range(10):
            train_data, train_label, validation_data, validation_label = self.cv_split(i)
            labels_pre, train_time, infer_time = self.model(train_data, validation_data, train_label,**params_dict)
            evaluate = evaluation(labels_pre, validation_label)
            val.append(evaluate.calc_metrics()[2])
            t_time.append(train_time/len(train_data))
            i_time.append(infer_time/len(validation_data))
        tol_len = len(val)
        return (self.l1*(sum(val)/tol_len-0.8)/0.2-self.l2*(sum(t_time)/tol_len)/3e-2-self.l3*(sum(i_time)/tol_len)/2e-3)
    
    def knn_opt(self, n_neighbors, metric):

        if metric <= 1:
            metric = 'minkowski'
        elif metric <= 2:
            metric = 'manhattan'
        else:
             metric = 'cosine'
        
        params_dict = {"n_neighbors":int(n_neighbors),"metric":metric}
        val,t_time,i_time = [], [], []
        for i in range(10):
            train_data, train_label, validation_data, validation_label = self.cv_split(i)
            labels_pre, train_time, infer_time = self.model(train_data, validation_data, train_label,**params_dict)
            evaluate = evaluation(labels_pre, validation_label)
            val.append(evaluate.calc_metrics()[2])
            t_time.append(train_time/len(train_data))
            i_time.append(infer_time/len(validation_data))
        tol_len = len(val)
        return (self.l1*(sum(val)/tol_len-0.8)/0.2-self.l2*(sum(t_time)/tol_len)/3e-2-self.l3*(sum(i_time)/tol_len)/2e-3)

    def mlp_opt(self, hidden_neurons, solver, activation, alpha, tol, max_iter):
        if solver <= 1:
            solver = 'lbfgs'
        elif solver <= 2:
            solver = 'sgd'
        else:
            solver = 'adam'
        
        if activation <= 1:
            activation = 'identity'
        elif activation <= 2:
            activation = 'logistic'
        elif activation <= 3:
            activation = 'relu'
        else:
            activation = 'tanh'
        
        params_dict = {"hidden_layer_sizes": (int(hidden_neurons),), "solver":solver,\
             "activation":activation, "alpha":alpha, "tol":tol,"max_iter":int(max_iter)}
        val, t_time, i_time = [], [], []
        for i in range(10):
            train_data, train_label, validation_data, validation_label = self.cv_split(i)
            labels_pre, train_time, infer_time = self.model(train_data, validation_data, train_label,**params_dict)
            evaluate = evaluation(labels_pre, validation_label)
            val.append(evaluate.calc_metrics()[2])
            t_time.append(train_time/len(train_data))
            i_time.append(infer_time/len(validation_data))
        tol_len = len(val)
        return (self.l1*(sum(val)/tol_len-0.8)/0.2-self.l2*(sum(t_time)/tol_len)/3e-2-self.l3*(sum(i_time)/tol_len)/2e-3)

    def optimize(self):
        if self.model.__name__ == 'decision_tree':
            optimizer = BayesianOptimization(self.dt_opt,self.params_range_dict)
            optimizer.maximize(init_points=30, n_iter=100)
            
            best_params = {}
            best_score = optimizer.max['target']
            temp_dict = optimizer.max['params']
            if temp_dict['criterion'] <= 1:
                best_params['criterion'] = 'entropy'
            else:
                best_params['criterion'] = 'gini'
            best_params['max_depth'] = int(temp_dict['max_depth'])
            best_params['min_samples_leaf'] = int(temp_dict['min_samples_leaf'])
            best_params['min_samples_split'] = int(temp_dict['min_samples_split'])
        
        if self.model.__name__ == 'KNN':
            optimizer = BayesianOptimization(self.knn_opt,self.params_range_dict)
            optimizer.maximize(init_points=30, n_iter=100)
            
            best_params = {}
            best_score = optimizer.max['target']
            temp_dict = optimizer.max['params']
            if temp_dict['metric'] <= 1:
                best_params['metric'] = 'minkowski'
            elif temp_dict['metric'] <= 2:
                best_params['metric'] = 'manhattan'
            else:
                best_params['metric'] = 'cosine'
            best_params['n_neighbors'] = int(temp_dict['n_neighbors'])
        
        elif self.model.__name__ == 'MLP':
            optimizer = BayesianOptimization(self.mlp_opt,self.params_range_dict)
            optimizer.maximize(init_points=50, n_iter=100)
            
            best_params = {}
            temp_dict = optimizer.max['params']

            best_score = optimizer.max['target']
            if temp_dict['solver'] <= 1:
                best_params['solver'] = 'lbfgs'
            elif temp_dict['solver'] <= 2:
                best_params['solver'] = 'sgd'
            else:
                best_params['solver'] = 'adam'

            if temp_dict['activation'] <= 1:
                best_params['activation'] = 'identity'
            elif temp_dict['activation'] <= 2:
                best_params['activation'] = 'logistic'
            elif temp_dict['activation'] <= 3:
                best_params['activation'] = 'relu'
            else:
                best_params['activation'] = 'tanh'
            best_params['hidden_layer_sizes'] = (int(temp_dict['hidden_neurons']),)
            best_params['alpha'] = temp_dict['alpha']
            best_params['tol'] = temp_dict['tol']
            best_params['max_iter'] = int(temp_dict['max_iter'])

        print(best_score)
        return best_params


