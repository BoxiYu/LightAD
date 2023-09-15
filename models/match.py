import distance
import numpy as np
import time

class semantics_match:

    def __init__(self):
        pass

    def update_training_database(self, database, input, label):
        input = str(input)
        if label == '-':
            if input in database['normal'].keys():
                database['normal'][input] += 1
            else:
                database['normal'][input] = 1
        else:
            if input in database['abnormal'].keys():
                database['abnormal'][input] += 1
            else:
                database['abnormal'][input] = 1
        return database

    def search_min_dist(self, training_database, input, matched, distance_type='Jaccard'):

        # set min dist with a large value
        current_dist = 100000000
        current_label = None
        vote_normal = 0
        vote_abnormal = 0
        current_matched = None
        current_matched_label = []
        for train_data in training_database['normal'].keys():
            if distance_type == 'Jaccard':
                dist_temp = distance.jaccard(eval(train_data), input)
            elif distance_type == 'Manhattan':
                dist_temp = np.sum(abs(np.array(eval(train_data))-np.array(input)))
            elif distance_type == 'Levenshtein':
                dist_temp = distance.levenshtein(eval(train_data), input)
            if dist_temp < current_dist:
                current_label = '-'
                current_dist = dist_temp
                vote_normal = 1
                current_matched = [eval(train_data)]
                current_matched_label = ["-"]
            elif dist_temp == current_dist:
                vote_normal += 1
                current_matched.append(eval(train_data))
                current_matched_label.append("-")
        
        tie = True
        flag = False

        # traverse and find nearest
        for train_data in training_database['abnormal'].keys():
            if distance_type == 'Jaccard':
                dist_temp = distance.jaccard(eval(train_data), input)
            elif distance_type == 'Manhattan':
                dist_temp = np.sum(abs(np.array(eval(train_data))-np.array(input)))
            elif distance_type == 'Levenshtein':
                dist_temp = distance.levenshtein(eval(train_data), input)
            if dist_temp < current_dist:
                current_label = '+'
                current_dist = dist_temp
                tie = False
                vote_abnormal = 1
                vote_normal = 0
                current_matched = [eval(train_data)]
                current_matched_label = ["+"]
            elif dist_temp == current_dist:
                vote_abnormal += 1
                flag = True
                current_matched.append(eval(train_data))
                current_matched_label.append("+")
        
        # solve tie
        if tie and flag:
            print("tie exist")
            if vote_abnormal == vote_normal:
                min_dist = 100000000
                for idx,x in enumerate(current_matched):
                    current_min_dist = distance.levenshtein(x,input)
                    if current_min_dist < min_dist:
                        min_dist = current_min_dist
                        current_label = current_matched_label[idx]
            elif vote_normal > vote_normal:
                current_label = '-'
            else:
                current_label = '+'
        matched.append(current_matched)
        return current_label, matched


    def label_and_update_testing_database(self, testing_database, training_database, input, labels_pre, matched, dist_type='Jaccard'):
        exist = False
        vote_normal = 0
        vote_abnormal = 0
        if input in testing_database['normal']:
            labels_pre.append("-")
            matched.append("exist_test")
            return labels_pre, testing_database,matched
        if input in testing_database['abnormal']:
            labels_pre.append("+")
            matched.append("exist_test")
            return labels_pre, testing_database,matched
        if str(input) in training_database['normal']:
            exist = True
            vote_normal = training_database['normal'][str(input)]
        if str(input) in training_database['abnormal']:
            exist = True
            vote_abnormal = training_database['abnormal'][str(input)]
        
        # solve .duplicate/unique test sequence 
        if exist:
            matched.append('exist')
            if vote_normal > vote_abnormal:
                testing_database['normal'].append(input)
                labels_pre.append("-")
                return labels_pre, testing_database,matched
            else:
                # solve tie
                if vote_normal==vote_abnormal:
                    print("tie exists 1")
                testing_database['abnormal'].append(input)
                labels_pre.append("+")
                return labels_pre, testing_database,matched
        else:
            current_label, matched= self.search_min_dist(training_database, input, matched, distance_type=dist_type)
            if current_label == '-':
                testing_database['normal'].append(input)
                labels_pre.append("-")
                return labels_pre, testing_database,matched
            else:
                testing_database['abnormal'].append(input)
                labels_pre.append("+")
                return labels_pre, testing_database, matched

    def run(self, x_train, x_test, y_train, dist_type):
        start_time = time.time()
        training_database = {"normal":{},"abnormal":{}}
        testing_database = {"normal":[],"abnormal":[]}
        labels_pre = []
        matched = []
        for idx, x in enumerate(x_train):
            training_database = self.update_training_database(training_database, x, y_train[idx])
            if idx % 100000 == 0:
                print("training progress: "+str(idx)+"/"+str(len(x_train)))
        print("Training process finished!!!")
        end_time = time.time()
        train_time = end_time-start_time

        start_time = time.time()
        for idx, x in enumerate(x_test):
            labels_pre, testing_database, matched =\
                 self.label_and_update_testing_database(testing_database, training_database, x, labels_pre, matched, dist_type)
            if idx % 10000 == 0:
                print("testing progress: "+str(idx)+"/"+str(len(x_test)))
        end_time = time.time()
        infer_time = end_time - start_time
        return labels_pre, matched, train_time, infer_time