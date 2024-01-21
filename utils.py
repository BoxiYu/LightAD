import numpy as np
import os
import time
from sklearn.model_selection import train_test_split

class load_data:
    
    def __init__(self, dataset, eliminated=False):
        self.dataset = dataset
        self.eliminated = eliminated
        self.path_original = __file__.replace("utils.py","datasets/original_datasets/"+dataset)

        if dataset == 'hdfs':
            self.path_original_label = __file__.replace("utils.py","datasets/original_datasets/anomaly_label.csv")
    
        if self.eliminated:
            self.path = __file__.replace("utils.py","datasets/splited_datasets/"+dataset+"_eli")
        else:
            self.path = __file__.replace("utils.py","datasets/splited_datasets/"+dataset)
        print(self.path)


    def reload_data(self):
        dirs = os.listdir(self.path)
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        print(dirs)
        for dir in dirs:
            data = np.load(os.path.join(self.path,dir), allow_pickle=True)
            x_train.append(data["x_train"])
            x_test.append(data["x_test"])
            y_train.append(data["y_train"])
            y_test.append(data["y_test"])
            
        return x_train, x_test, y_train, y_test
    

    def eliminate(self, data_eliminated, data_formed, total_label):
        matrix = []
        new_data = [] 
        label = []
        for idx_r, x_r in enumerate(data_formed):
            temp_vector = [0]*len(data_eliminated)
            for idx, x in enumerate(x_r):
                vec_idx = data_eliminated.index(x)
                tf = x_r.count(x)
                temp_vector[vec_idx] = tf
            if temp_vector not in matrix:
                matrix.append(temp_vector)
                new_data.append(x_r)
                label.append(total_label[idx_r])
        return new_data, label

    def naive_agg(self, data_eliminated, data_total):
        vec_data = []
        for data in data_total:
            temp_vector = [0]*len(data_eliminated)
            for d in data:
                vec_idx = data_eliminated.index(d)
                temp_vector[vec_idx] += 1
            vec_data.append(temp_vector)
        return vec_data


    def load_and_split_hdfs(self, train_ratio=0.8, times=1):

        f_label = open(self.path_original_label,'rb')
        lines = f_label.readlines()
        label_dict = {}
        count_label = 0
        for line in lines:
            if count_label == 0:
                count_label += 1
                continue
            line = line.decode('utf-8')[:-2].split(',')
            count_label += 1
            if line[1] == 'Normal':
                current_label = "-"
            else:
                current_label = "+"
            label_dict[line[0]] = current_label
            count_label += 1

        total_time = 0
        f = open(self.path_original,'rb')
        total_data = {}
        data_eliminated = []
        total_label = []
        lines = f.readlines()
        count = 0
        debug_content = []
        for line in lines:
            #print(line.decode('utf-8'))
            line = line.decode('utf-8')[:-2].split(' ')
            start_time = time.time()
            content = line[5:]
            new_content = []
            for c in content:
                flag = True
                if c.startswith('blk_') and flag:
                    flag = False
                    current_blk = c.strip('.')
                    if current_blk not in total_data.keys():
                        total_data[current_blk] = []
                        #print(count,current_blk)
                for element in c:
                    if element.isdigit(): #whether token contains digits
                        flag = False
                        break
                if flag:
                    new_content.append(c)
            if new_content not in debug_content:
                #print(new_content)
                debug_content.append(new_content)
            total_data[current_blk].append(new_content)
            
            if new_content not in data_eliminated:
                data_eliminated.append(new_content)
            end_time = time.time()
            total_time += (end_time-start_time)
            count += 1
            if count % 500000 == 0:
                print("Data pre_loading:",count,"/",len(lines), "Current event types:",len(data_eliminated))
        total_data_temp = []
        start_time = time.time()
        for key in total_data.keys():
            total_label.append(label_dict[key])
            total_data_temp.append(total_data[key])
        end_time = time.time()
        total_time += (end_time-start_time)

        # whether eliminate (deduplicate)
        if self.eliminated:
            total_data_temp, total_label = self.eliminate(data_eliminated, total_data_temp, total_label)


        start_time = time.time()
        total_data_temp = self.naive_agg(data_eliminated, total_data_temp)
        end_time = time.time()
        total_time+=(end_time-start_time)
        for t in range(times):
            #pdb.set_trace()
            x_train_temp, x_test_temp, y_train, y_test = train_test_split(total_data_temp, total_label, test_size=1-train_ratio)
            start_time = time.time()

            x_train, x_test = x_train_temp, x_test_temp


            end_time = time.time()
            total_time += (end_time-start_time)
            print("Total preprocessing time is:",total_time)
            print(t, len(x_train))
            if not os.path.exists(self.path):
                os.mkdir(self.path)
            np.savez(os.path.join(self.path,"shuffle_"+str(t)+".npz"), x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    def load_and_split(self, preprocess=True, data_range=[0,10000000], train_ratio=0.8,  window_size=1, step_size=1):

        f = open(self.path_original,'rb')
        total_data = []
        total_data_window = []
        total_label = []
        total_label_window = []
        count = 0
        first_occurences = []
        error_count = 0
        error_types = []
        total_time = 0

        lines = line = f.readlines()
        for idx, line in enumerate(lines):
            try:  
                if count < data_range[0]:
                    count += 1
                    continue
                line = line.decode('utf-8')[:-1]
                line = line.split(' ')
   
                start_time = time.time()
                if self.dataset in ['spirit','tbird','bgl']:
                    content = line[8:]
                elif self.dataset == 'liberty':
                    content = line[7:]
                label = line[0]
                if preprocess:
                    new_content = []
                    for c in content:
                        flag = True
                        char = False
                        for element in c:
                            if element.isdigit():
                                flag = False
                                break
                            if element.isalpha():
                                char = True
                        if flag and char:
                            new_content.append(c)
                    total_data.append(new_content)
                else:
                    total_data.append(content)
                if label == '-':
                    total_label.append(label)
                else:
                    total_label.append('+')
                if label != '-':
                    if label not in error_types:
                        error_types.append(label)
                        first_occurences.append(count)
                    error_count += 1

                if count >= window_size+data_range[0]-1 and (count-data_range[0]-window_size+1)%step_size==0:
                    temp_data = total_data[count-data_range[0]-window_size+1:count-data_range[0]+1]

                    total_data_window.append(temp_data[0])

                    if '+' in total_label[count-data_range[0]-window_size+1:count-data_range[0]+1]:
                        total_label_window.append("+")
                    else:
                        total_label_window.append("-")
                
                end_time = time.time()
                total_time += (end_time-start_time)
                count += 1
                if count % 500000 == 0:
                    print("Data pre_loading:",count-data_range[0],"/",data_range[1]-data_range[0])
                if count == data_range[1] or idx == len(lines)-1:
                    print("Number of error types in current data:", error_count)
                    print("Error types in current data:", error_types)
                    print("Their first occurences are:", first_occurences)
                    break
            except Exception as e:
                print('decode error!!!')
                print(len(total_data_window))
        print("Preprocessing time (w./o vectorization):",total_time)
        
    
        # No shuffle
        train_size = int(train_ratio*len(total_data_window))
        x_train, x_test, y_train, y_test = total_data_window[:train_size], total_data_window[train_size:],\
            total_label_window[:train_size], total_label_window[train_size:]
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        np.savez(os.path.join(self.path,"unshuffle.npz"),x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    



class evaluation:

    def __init__(self, labels, labels_pre):
        self.labels = labels
        self.labels_pre = labels_pre

    def calc_metrics(self):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for idx, l in enumerate(self.labels_pre):
            if l=="-" and self.labels[idx]=="-":
                tn += 1
            elif l == "-" and self.labels[idx] != "-":
                fn += 1
            elif l != "-" and self.labels[idx] == "-":
                fp += 1
            else:
                tp += 1
        #print(tp,fp,tn,fn)
        if (tp+fp)==0:
            prec = 0
        else:
            prec = tp/(tp+fp)
        if (tp+fn)==0:
            rec = 0
        else:
            rec = tp/(tp+fn)
        if (prec+rec)==0:
            f=0
        else:
            f = 2*prec*rec/(prec+rec)
        if (fp + tn) == 0:
            spec = 0
        else:
            spec = tn / (fp + tn)
        bal = (spec + rec) / 2
        return prec, rec, f, spec, bal
        

    
    def calc_with_windows(self,window=20,step=1):

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        idx = 0
        while True:
            if (idx+window) >= len(self.labels_pre):
                break
            flag_pre = "-"
            for l_pre in self.labels_pre[idx:idx+window]:
                if l_pre != "-":
                    flag_pre = "+"
            flag_true = "-"
            for l_true in self.labels[idx:idx+window]:
                if l_true != "-":
                    flag_true = "+"
            if flag_pre=="-" and flag_true=="-":
                tn += 1
            elif flag_pre == "-" and flag_true != "-":
                fn += 1
            elif flag_pre != "-" and flag_true == "-":
                fp += 1
            else:
                tp += 1
            idx += step
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        f = 2*prec*rec/(prec+rec)
        # spec = tn / (fp + tn)
        if (fp + tn) == 0:
            spec = 0
        else:
            spec = tn / (fp + tn)
        bal = (spec + rec) / 2
        return prec, rec, f, spec, bal
