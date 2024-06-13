import pickle, more_itertools, random
from tqdm import tqdm
import pandas as pd
from classification import train_test_MLP, MLP_GridSearch, \
                            train_test_DimProj, DimProj_SimThresSearch
from sklearn.metrics import accuracy_score, classification_report

def tenfoldsplits2instances(instances):
    random.shuffle(instances)
    tenfolds = [set(fold) for fold in more_itertools.divide(10, instances)]
    fold2instances = {i: {'test': tenfolds[i-1],
                    'dev': tenfolds[i-2], 
                    'train': set().union(*[f for j, f in enumerate(tenfolds) if j not in (i-1, 9 if i == 1 else i-2)])
                    } for i in range(1, 11)}
    return fold2instances


def get_split_data(data, id2embeddings, split2items, item_column, id_column):

    train_X, train_y, dev_X, dev_y, test_X, test_y = [], [], [], [], [], []
    train_data, dev_data, test_data = [], [], []
    
    for _, row in data.iterrows():
        if row[item_column] in split2items['train']:
            train_X.append(id2embeddings[row[id_column]])
            train_y.append(row['encoded_label'])
            train_data.append(row)
        elif row[item_column] in split2items['dev']:
            dev_X.append(id2embeddings[row[id_column]])
            dev_y.append(row['encoded_label'])
            dev_data.append(row)
        elif row[item_column] in split2items['test']:
            test_X.append(id2embeddings[row[id_column]])
            test_y.append(row['encoded_label'])
            test_data.append(row)

    return train_X, train_y, dev_X, dev_y, test_X, test_y,\
          train_data, dev_data, test_data


def evaluate(data_path, id_column, label_column, label_encoder, embedding_path, 
             output_path, clf, params, random_split_seed, splitby):

    logs = []
    random.seed(random_split_seed)

    # load sense representations of model
    with open(embedding_path, 'rb') as infile:
        id2embeddings = pickle.load(infile)
    
    # load and shuffle data and encode labels
    data = pd.read_csv(data_path).sample(frac=1, random_state=12, ignore_index=True)
    #data[label_column] = data[label_column].fillna('NaN') # uncomment if Wiktionary labels
    data['encoded_label'] = data[label_column].replace(label_encoder)
    # exclude data for which no model representations exist and no label for exist
    data = data[data[id_column].isin(id2embeddings)].dropna(subset=['encoded_label']) 
    data = data.drop(data[data[label_column] == "None"].index)
    
    # initialize 10 folds based on set of unique items
    fold2items = tenfoldsplits2instances(list(set(data[splitby])))
    tenfold_accuracies, test_data, test_predictions = [], [], []
    for i, (fold_no, split2items) in tqdm(enumerate(fold2items.items())):
        logs.append(f"\nFold {fold_no}")
        
        # get data based on sets of items (type specified in splitby)
        train_X, train_y, dev_X, dev_y, test_X, test_y,\
          train_fold_data, dev_fold_data, test_fold_data = get_split_data(data, id2embeddings, split2items, splitby, id_column)
        logs.append(f"Train size: {len(train_y)} / Dev size: {len(dev_y)} / Test size: {len(test_y)}")
        test_data.extend([dict(row, **{'test_fold_no': fold_no}) for row in test_fold_data])

        # train and test classification model
        if clf == 'mlp':
            if not params:
                best_params = MLP_GridSearch(dev_X, dev_y)
                predictions, accuracy = train_test_MLP(train_X, train_y, test_X, test_y, best_params)
                logs.append(f"Grid Search Result - Best Hyperparameters: {best_params}")
            else:
                predictions, accuracy = train_test_MLP(train_X, train_y, test_X, test_y, params)
        elif clf == 'dimproj':
            if not params:
                best_params = DimProj_SimThresSearch(train_X, train_y, dev_X, dev_y)
                predictions, accuracy = train_test_DimProj(train_fold_data, train_X, train_y, test_X, test_y, best_params) 
                logs.append(f"Best Similarity Threshold (for embedding pair selection): {best_params}")
            else:
                predictions, accuracy, n_pairs = train_test_DimProj(train_fold_data, train_X, train_y, test_X, test_y, params) 
                logs.append(f"{n_pairs} pairs included for dimension creation")
            
        logs.append(f"Accuracy: {accuracy}")
        tenfold_accuracies.append(accuracy)
        test_predictions.extend(predictions)
    
    # save preds
    output_df = pd.DataFrame(test_data)
    output_df['prediction'] = test_predictions
    output_df.to_csv(output_path, index=False)

    logs.append(f'\nAverage of accuracies over {len(tenfold_accuracies)} folds: {sum(tenfold_accuracies)/len(tenfold_accuracies)}')
    logs.append(f"Overall accuracy of predictions for all test folds: {accuracy_score(output_df['encoded_label'], output_df['prediction'])}")
    logs.append(classification_report(output_df['encoded_label'], output_df['prediction'])) 

    return logs