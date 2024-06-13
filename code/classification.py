import torch
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def train_test_MLP(X_train, y_train, X_test, y_test, params):

    X_train = torch.stack(X_train)  
    X_test = torch.stack(X_test)

    clf = MLPClassifier(random_state=12).set_params(**params).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return predictions, accuracy

@ignore_warnings(category=ConvergenceWarning)
def MLP_GridSearch(X_dev, y_dev, 
                   param_grid = {
                       'hidden_layer_sizes':[(300, 200, 100, 50), (200, 100, 50), (100, 50)], 
                       'learning_rate_init':[0.0005, 0.001, 0.005],
                       'max_iter': [10, 20, 40, 80, 100, 200]}):

    mlp = MLPClassifier(random_state=12)
    clf = GridSearchCV(mlp, param_grid)
    clf.fit(torch.stack(X_dev), y_dev)

    return clf.best_params_


def train_test_DimProj(train_data, X_train, y_train, X_test, y_test, params):
    # binary classification only

    pos_vecs, neg_vecs = [], []
    pos_data_ids, neg_data_ids = [], []
    for i, y in enumerate(y_train):
        if y == 0:
            pos_vecs.append(X_train[i])
            #pos_data_ids.append(train_data[i]['id'])
        else:
            neg_vecs.append(X_train[i])
            #neg_data_ids.append(train_data[i]['id'])

    # create dimension vector
    pairwise_dist = pairwise_cosine_similarity(torch.stack(pos_vecs), torch.stack(neg_vecs))
    pairwise_dist_dict = dict()
    for p_id in range(len(pos_vecs)):
        for n_id in range(len(neg_vecs)):
            pairwise_dist_dict[(p_id, n_id)] = pairwise_dist[p_id, n_id].item()
    top_similar_pairs = [pair for pair, sim in pairwise_dist_dict.items() if sim >= params['sim_thres']]
    
    if len(top_similar_pairs) > 0:
        diff_vecs = [pos_vecs[p_id] - neg_vecs[n_id] for (p_id, n_id) in top_similar_pairs]
        dimension = torch.mean(torch.stack(diff_vecs), 0)

        #dimension_data = [(pos_data_ids[pos_id], neg_data_ids[neg_id], 
                #pairwise_dist_dict[(pos_id, neg_id)]) for (pos_id, neg_id) in top_similar_pairs]

        # project test embeddings
        cos = torch.nn.CosineSimilarity(dim=0)
        threshold = 0 # make this decision threshold a parameter?
        predictions = []
        for x in X_test:
            cossim = cos(x, dimension).item()
            predictions.append(1 if cossim > threshold else 0)
        accuracy = accuracy_score(y_test, predictions)
    
        return predictions, accuracy, len(top_similar_pairs)
    
    print(f"No pairs with similarity above threshold {params['sim_thres']}")
    return [], 0, 0

def DimProj_SimThresSearch(X_train, y_train, X_dev, y_dev, 
                            sim_thresholds=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    
    best_acc = 0
    for st in sim_thresholds:
        params = {'sim_thres': st}
        _, acc = train_test_DimProj([], X_train, y_train, X_dev, y_dev, params)
        #print('similarity threshold', st, '- accuracy:', acc)
        if acc > best_acc:
            best_acc = acc
            best_params = {'sim_thres': st}
    
    return best_params
    