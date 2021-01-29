#!/usr/bin/env python
# coding: utf-8

# In[11]:


from core.DataLoader import *
# from core.models.MLP import * 
# from core.models.GAM import *
# from core.models.SVM import *
# from core.models.EBM import *
# from core.models.RandomForest import *
# from core.Visualizer import *

import itertools
from multiprocessing import Pool
import json

def read_data(engine='postgres'):
    dl = DataLoader(engine)
    one_file_dss = dl.get_one_file_ds()
    clustered_file_ds = dl.get_clustered_files_ds()
    all_file_ds = dl.get_all_files_ds()
    return one_file_dss, clustered_file_ds, all_file_ds

class classification_model_runner():
    def __init__(self, original_df):
        self.df = original_df
        self.models = {}
        self.model_acc = {}

    def set_dataframe(self, df):
        self.df = df

    def prepare_features(self, train_features, target_feature='optimal_decision', cost_features=DataLoader().regression_targets):
        self.train_features = train_features
        X = self.df[train_features]
        y = self.df[target_feature]
        X_costs = self.df[cost_features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
        X_train_costs, _, _, _, = train_test_split(X_costs, y, train_size=0.8, random_state=1)
        X_train, X_test, y_train, y_test =             X_train.to_numpy(),X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

        X_train_weights = calculate_importance_from_costs(X_train_costs.to_numpy())

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        # scaler = preprocessing.StandardScaler().fit(X_test)
        X_test = scaler.transform(X_test)

        X_train_weights = preprocessing.MinMaxScaler().fit_transform(X_train_weights.reshape(-1,1))

        self.X_train = X_train
        self.X_test = X_test
        self.X_test = X_test
        self.X_train_weights = X_train_weights
        self.X_scaler = scaler
        
        self.y_train = y_train
        self.y_test = y_test
        

    def set_model_list(self, model_name_list):
        self.model_name_list = model_name_list

    def train_models(self):
        for model_name in self.model_name_list:
            if model_name == MLPClassifier:
                clf = model_name(n_features=len(self.train_features))
            elif model_name == sk_nn.MLPClassifier:
                clf = model_name(hidden_layer_sizes=(100,100,100,100))
            else:
                clf = model_name()
            clf = clf.fit(self.X_train, self.y_train)
            acc = clf.score(self.X_test, self.y_test)
#             print(f"Accuray of {model_name}: {acc}")
            # if len(features) == 2:
            #     plot_2d_decision_boundaries(clf, X_train, X_train_costs, y_train, title=f'{model} on {i}-th one table pair')
            self.models[model_name] = clf
            self.model_acc[model_name] = acc    


# In[14]:

def run_all_features():
    one_file_dss, clustered_file_ds, all_file_ds = read_data()
    # all_dfs = [ds for ds in one_file_dss] + [clustered_file_ds, all_file_ds]
    all_dfs = [all_file_ds]

    all_features = DataLoader().all_features
    model_name_list=[sk_nn.MLPClassifier, m_RandomForestClassifier,
                            SVMClassifier, LinearGAMClassifier, EBMClassifier]

    from multiprocessing import Pool
    global performances
    performances = {}

    def eval_features(train_features):
        accs = []
        for ds in all_dfs:
            runner = classification_model_runner(ds)
            runner.prepare_features(list(train_features))
            runner.set_model_list(model_name_list=model_name_list)
            runner.train_models()
            accs += list(runner.model_acc.values())
            # accs.append(np.average(list(runner.model_acc.values())))
        accs = [str(i) for i in accs]
        with open(f'all_perfomances_{len(train_features)}/{"|".join(train_features)}.txt', 'w') as fp:
            fp.write('\n'.join(accs))


    for n in [2,3,4,5]:
        performances = {}
        args = []

        for train_features in tqdm(list(itertools.combinations(all_features, n))):
            args.append(list(train_features))

        with Pool(50) as p:
            p.map(eval_features, args)



def eval_features_results(base_dir='all_perfomances_{}', nums=[2,3]):

    for n in nums:
        m_dir = base_dir.format(n)
        files = os.listdir(m_dir)

        perf = {}
        for txt_file in files:
            accs = []
            with open(os.path.join(m_dir, txt_file) , 'r') as f:
                line = f.readline()
                while line:
                    accs.append(float(line.replace('\n', '')))
                    line = f.readline()

            perf[tuple(txt_file.replace('.txt', '').split('|'))] = accs # np.average(accs)

        max_key = max(perf, key=perf.get)
        print(max_key)
        print(perf[max_key])
        print(perf[('left_cardinality', 'base_cardinality', 'sel_of_pred_on_indexed_attr' )])




if __name__ == "__main__":
    eval_features_results(nums=[3])




