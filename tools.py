from sklearn import impute, cluster
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from time import time
from collections import Counter

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import GridSearchCV 
# import lightgbm as lgb
from sklearn.externals import joblib 

class TF_IDF():
    def __init__(self, stop_words=[]):
        self.stop_words = stop_words
        self.list_doc = None
        self.set_voc = None
        self.counter_WD = None
        self.tf, self.idf = None, None

    def fit(self, list_doc):
        self.list_doc = list_doc
        self.set_voc, self.counter_WD = self.__build_dictionary()
        self.tf , self.idf = self.__calculate(self.counter_WD)

    # def _write_file(self):

    def __remove_stopwords(self, words : "string"):
        cleaned_text = [w.lower() for w in words if w not in self.stop_words]
        return cleaned_text

    def __build_dictionary(self):
        print('Building dictionary...', end='')
        start = time()
        k = len(self.list_doc)
        set_voc = set()
        counter_word_doc = []
        cnt = 0 
        for index in range(k):
            context_clean = self.__remove_stopwords(self.list_doc[index])
            counter_word_doc.append(Counter(context_clean))
            cnt += len(context_clean)
            set_voc.update(context_clean)

        set_voc = sorted(list(set_voc))
        print('finish: {:.4}s'.format(time() - start))

        return set_voc, counter_word_doc

    def __call__(self):
        return self.tf, self.idf

    def __calculate(self, counter_WD):
        
        k, n = len(self.list_doc), len(self.set_voc)
        tf, idf = np.zeros((n,k)), np.zeros(n)

        for i in range(k):
            counter = counter_WD[i]

            for word, count in counter.items():
                index = self.set_voc.index(word)
                tf[index][i] = count
                idf[index] += 1

        self.idf = np.log(k/idf)
        tf = tf/tf.max()
        return tf, idf

    def predict(self, list_doc):
        counter_word_doc = []
        k, n = len(list_doc), len(self.set_voc)
        tf = np.zeros((n,k))
        for i in range(k):
            context_clean = self.__remove_stopwords(list_doc[i])

            counter = Counter(context_clean)


            for word, count in counter.items():
                index = self.set_voc.index(word)
                tf[index][i] = count
        tf = tf/tf.max()
        return tf.T*self.idf.T

class Cluster_address():
    def __init__(self, path_stopwords, k=4, distance='L2'):
        self.stop_words = None
        with open(path_stopwords, 'r') as f:
            self.stop_words = f.read().strip()

        self.k = k

        # self.distance = distance

    def fit(self, address_list):

        self.address = address_list

        self.tf_idf = TF_IDF(self.stop_words)
        
        self.tf_idf.fit(self.address)
        
        tf, idf = self.tf_idf()
        print('tf.shape, idf.shape :', tf.shape, idf.shape)
        tf = tf.T * idf.T
    
        self.clustering = cluster.KMeans(n_clusters=self.k).fit(tf)

        # clustering = cluster.MeanShift().fit(tf)
    
        y_pred = self.clustering.predict(tf)

        return y_pred

    def predict(self, address_list):
        tf = self.tf_idf.predict(address_list)
        y_pred = self.clustering.predict(tf)
        return y_pred

class dataloader():
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_cluster =  self.cfg['num_cluster']
        self.class_name = ['t', 'h', 'u']

    def read_data(self):
        self.origin_df = pd.read_csv(self.cfg['path_data'])


    def excute(self):
        self.transform()

        # normalize 
        if  self.cfg['cfgNormal']['isNormal']:
            self.data= self.normalize(self.data)

        self.split_data()
        self.remove_outlier()

    def transform(self):
        # __init__
        path_stopwords =  self.cfg['path_stopwords']
        df = self.origin_df.copy()
        col_object = df.select_dtypes(['object']).columns
        col_numeric = df.columns.difference(col_object)
        
        self.cluster_address = Cluster_address(path_stopwords, self.num_cluster)

        # ____excute____

        # address
        df['Address'] = self.cluster_address.fit(df['Address'])

        # Add new features
        # df['cnt_NaN'] = df[columns].isna().sum(axis=1)

        # Impute data
        self.imputer_object = impute.SimpleImputer(strategy= self.cfg['impute_object'], fill_value=-1)
        self.imputer_numeric = impute.SimpleImputer(strategy= self.cfg['impute_numeric'], fill_value=-1)

        df[col_object] = self.imputer_object.fit_transform(df[col_object])
        df[col_numeric] = self.imputer_numeric.fit_transform(df[col_numeric])

        unique_col = [list(set(df[col])) for col in col_object]

        for i_col in range(len(col_object)):
            col = col_object[i_col]
            if col == 'Type':
                df[col] = df[col].apply(lambda x: self.class_name.index(x))
            else:
                df[col] = df[col].apply(lambda x: unique_col[i_col].index(x))


        self.labels = df.pop('Type')
        self.columns_name = set(df.columns)

        self.data = df

    def normalize(self, df):
        cfgNormal = self.cfg['cfgNormal']
        esilon = 1e-9
        if cfgNormal['type'] == 'min_max':
            new_nim, new_max =  self.cfg['cfgNormal']['min_max']
            df=(df-df.min())/(df.max()-df.min() + esilon)* (new_max - new_nim) + new_nim
        return df

    def detect_outliers(self, numpy_data):
        lof = LocalOutlierFactor()
        yhat = lof.fit_predict(numpy_data)
        mask = yhat != -1
        num_of_out= numpy_data.shape[0] - len(numpy_data[mask, :])
        log = 'Num of outliers: %d (%.3f%%)\nNum of samples after removed outliers: %d' % (num_of_out, num_of_out/numpy_data.shape[0]*100  , len(numpy_data[mask, :]))
        return mask, log

    def split_data(self):
        size = list(map(int, self.cfg['train_test'].split('/')))
        check =  self.cfg['check_split_data']
        random_state =  self.cfg['random_state']
        X = self.data
        if isinstance(X, pd.DataFrame):
            X = self.data.to_numpy()

        Y = self.labels.to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, train_size=size[0]/10, random_state= random_state)
    
        if check:
            for i in [X_train, X_test]: print('%d (%.3f)' % (len(i), len(i)/len(X)))

        self.X_train, self.X_test, self.y_train, self.y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def remove_outlier(self):
        mask, log = self.detect_outliers(self.X_train)
        self.X_train, self.y_train = self.X_train[mask, :], self.y_train[mask]
        return log

    def load_data(self):
        return (self.X_train,self.y_train), (self.X_test,  self.y_test)

    def entropy(self, data_col : "data_frame", labels, is_data=False):
        s = 0
        if is_data:
            c = Counter(labels)
            row = np.array([c[i] for i in c])/len(labels)
            s -= np.sum(row*np.log2(row))
        else:

            unique= list(set(data_col))
            unique_lb = list(set(labels))
            table = np.zeros((len(unique), len(unique_lb)))
            
            for i in range(len(unique_lb)):
                index = labels==unique_lb[i]
                cnt = Counter(data_col[index])
                try:
                    
                    for key, val in cnt.items():
                        j = unique.index(key)
                        table[j][i] += val
                except :
                    print(unique)
                    print(Counter)
            for row in table:
                if row.all()>0:
                    n = row.sum()
                    row/= n
                    s -= np.sum(row*np.log2(row))*n/len(labels)

        return s

    def information_gain(self, data=None, labels=None):

        if not isinstance(data, pd.DataFrame):
            data, labels =self.data, self.labels
        info_D = self.entropy('',labels, True)

        self.infor_gain={}
        for col in data:
            info = self.entropy(data[col], labels)
            info_gain= info_D - info
            self.infor_gain[col] = info_gain
        return self.infor_gain
   
    def save(self):
        path = 'data_pre/'
        if not os.path.isdir(path): 
            os.mkdir(path)
        name = path + 'X_train.npy'
        np.save(name, self.X_train)
        name = path + 'y_train.npy'
        np.save(name, self.y_train)

        name = path + 'X_test.npy'
        np.save(name, self.X_test)
        name = path + 'y_test.npy'
        np.save(name, self.y_test)

    def load(self, path='data_pre/'):

        self.X_train = np.load(path + 'X_train.npy')
        self.y_train = np.load(path + 'y_train.npy')

        self.X_test = np.load(path + 'X_test.npy')
        self.y_test = np.load(path + 'y_test.npy')

    def predict(self, path):
        df = pd.read_csv(path)

        col_object = df.select_dtypes(['object']).columns
        col_numeric = df.columns.difference(col_object)

        # ____excute____


        # address
        df['Address'] = self.cluster_address.predict(df['Address'])


        # Impute data
        df[col_object] = self.imputer_object.fit_transform(df[col_object])
        df[col_numeric] = self.imputer_numeric.fit_transform(df[col_numeric])
        unique_col = [list(set(df[col])) for col in col_object]
        for i_col in range(len(col_object)):
            col = col_object[i_col]
            df[col] = df[col].apply(lambda x: unique_col[i_col].index(x))

        if self.cfg['cfgNormal']['isNormal']:
            df= self.normalize(df)
        return df.to_numpy()


class visualize():
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def distribution(self):
        dir_name = 'hist/'
        if not os.path.isdir(dir_name):
           os.mkdir('hist/')
        for i in self.dataloader.data.columns:
            ax = self.dataloader.data.hist(i)
            ax[0][0].get_figure().savefig(dir_name+'hist_{}.png'.format(i))

        return self.dataloader.data.hist()
    

    def temp():


        self.dataloader.transform()

        # normalize 
        if  self.dataloader.cfg['cfgNormal']['isNormal']:
            self.dataloader.data= self.dataloader.normalize(self.dataloader.data)

        self.dataloader.split_data()
        
        self.dataloader.remove_outlier()

class model_team_7():

    def __init__(self, cfg):
        self.cfg = cfg
        self.algorithm =  self.cfg['algorithm']
        self.clf = None
        self.__classifier()
        
    def __classifier(self):
        algorithm = self.algorithm
        clf= None
        if algorithm=='svm':
            clf =SVC(kernel='rbf',gamma='scale',C=1.0) #svm.LinearSVC(max_iter=10000) #SVC(gamma='auto', max_iter=100000)
        elif algorithm=='decision_tree':
            clf = DecisionTreeClassifier()
        elif algorithm == 'LogisticRegression':
            clf = LogisticRegression(C=1e5)
        elif algorithm == 'random':
            clf = RandomForestClassifier(n_estimators=10, random_state=30)
        elif algorithm == 'naive':
            clf = GaussianNB()
        assert clf!=None
        
        self.clf = clf


    def fit(self, dataloader):
        (self.X_train,self.y_train), (self.X_test,  self.y_test) = dataloader.load_data()
        self.clf.fit(self.X_train, self.y_train)
        
    def eval(self, y_test, y_pred, phrase, isBool=True):
        assert len(y_test) == len(y_pred),'{} {}'.format(len(y_test), len(y_pred))
        n = 3 # class 
        confusion_matrix = np.zeros((n, n), dtype=int)
        for i in range(len(y_pred)):
            confusion_matrix[y_test[i], y_pred[i]] += 1
        normal_confusion_matrix = confusion_matrix/confusion_matrix.sum()

        if isBool:
            print('matrix: \n', confusion_matrix, end='\n\n')
            print('normalize matrix: \n', normal_confusion_matrix, end='\n\n')

        print('acc {}: '.format(phrase), normal_confusion_matrix.trace())

        return normal_confusion_matrix.trace()

    def eval_2(self, y, y_pred):
        log = 'confusion matrix:\n{}\n{}\n'.format(confusion_matrix(y, y_pred), classification_report(y, y_pred))
        return log
    def predict(self, X):
        return self.clf.predict(X)
    def evaluation(self, train_set = None, test_set=None):


        X_train, y_train , X_test,  y_test = None, None, None, None

        if train_set == None:
            X_train, y_train , X_test,  y_test = self.X_train,self.y_train , self.X_test,  self.y_test            
        else:
           (X_train, y_train) ,(X_test,  y_test) = train_set, test_set      
            
        y_pred = None

        y_pred = np.array(self.clf.predict(X_test))
        
        y_pred_train = np.array(self.clf.predict(X_train))
        
        self.eval_2(y_train, y_pred_train)
        temp = self.eval_2(y_test, y_pred)
        print('-'*10)
        return temp
        # return (acc_train, acc_test)

    def save(self):
        path ='save_model/'
        if not os.path.isdir(path): 
            os.mkdir(path)
        path += '{}.pkl'.format(self.algorithm)
        joblib.dump(self.clf, path) 
        return 'Saved in {}'.format(path)

    def load(self, file):
        self.clf = joblib.load(file)  

        return 'Loaded in {}'.format(file)



def main():
    cfg_normal = {
        'isNormal' : True,
        'type': 'min_max' , # z_score, min_max
        'min_max': [-1, 1]
    }


    config = {}
    config['path_data'] = "data/Melbourne_housing_FULL.csv"
    config['path_stopwords'] = '/data/stopwords_en.txt'

    config['random_state'] = 432751
    config['train_test']= '7/3'
    config['check_split_data'] = False
    
    config['cfgNormal'] = cfg_normal
    config['impute_numeric'] ='mean'
    config['impute_object'] ='most_frequent'

    config['num_cluster'] = 8
    config['algorithm'] = 'svm'
    cfg = config

    data = dataloader(cfg)
    temp1 = list(data.columns_name)
    data.information_gain(data.origin_df[[i for i in temp1[1:]]], data.origin_df[temp1[0]])

if __name__ == "__main__":
    main()
