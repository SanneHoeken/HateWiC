import pickle, more_itertools, random
from tqdm import tqdm
import pandas as pd
from classification import train_test_MLP, train_test_DimProj
from sklearn.metrics import accuracy_score, classification_report

def fivefoldsplits2instances(instances):
    random.shuffle(instances)
    fivefolds = [set(fold) for fold in more_itertools.divide(5, instances)]
    fold2instances = {i: {'test': fivefolds[i-2],
                    'train': set().union(*[f for j, f in enumerate(fivefolds) if (j) != (4 if i == 1 else i-2)])
                    } for i in range(1, 6)}
    return fold2instances


def get_split_data(data, id2embeddings, split2ids):

    train_X, train_y, test_X, test_y = [], [], [], []
    train_data, test_data = [], []
    for _, row in data.iterrows():
        if row['id'] in split2ids['train']:
            train_X.append(id2embeddings[row['id']])
            train_y.append(row['encoded_label'])
            train_data.append(row)
        elif row['id'] in split2ids['test']:
            test_X.append(id2embeddings[row['id']])
            test_y.append(row['encoded_label'])
            test_data.append(row)

    return train_X, train_y, test_X, test_y,\
          train_data, test_data


def dinu_evaluate(data_path, label_column, embedding_path, output_path, clf, params, random_split_seed):

    logs = []
    random.seed(random_split_seed)

    # load sense representations of model
    with open(embedding_path, 'rb') as infile:
        id2embeddings = pickle.load(infile)
    
    # load and shuffle data and encode labels
    data = pd.read_csv(data_path).sample(frac=1, random_state=12, ignore_index=True) 
    data['encoded_label'] = data[label_column]
    # exclude data for which no model representations exist
    data = data[data['id'].isin(id2embeddings)]
    
    # initialize 5 folds based on ids 
    fold2ids = fivefoldsplits2instances(list(set(data['id'])))
    fivefold_accuracies, test_data, test_predictions = [], [], []
    for i, (fold_no, split2ids) in tqdm(enumerate(fold2ids.items())):
        assert len(set.intersection(*list(split2ids.values()))) == 0
        logs.append(f"\nFold {fold_no}")
        
        # get data based on sets of ids
        train_X, train_y, test_X, test_y,\
          train_fold_data, test_fold_data = get_split_data(data, id2embeddings, split2ids)
        logs.append(f"Train size: {len(train_y)} / Test size: {len(test_y)}")
        test_data.extend([dict(row, **{'test_fold_no': fold_no}) for row in test_fold_data])

        # train and test classification model
        if clf == 'mlp':
            predictions, accuracy = train_test_MLP(train_X, train_y, test_X, test_y, params)
        elif clf == 'dimproj':
            predictions, accuracy = train_test_DimProj(train_fold_data, train_X, train_y, test_X, test_y, params) 
            
        logs.append(f"Accuracy: {accuracy}")
        fivefold_accuracies.append(accuracy)
        test_predictions.extend(predictions)
    
    # save preds
    output_df = pd.DataFrame(test_data)
    output_df['prediction'] = test_predictions
    output_df.to_csv(output_path, index=False)

    logs.append(f'\nAverage of accuracies over {len(fivefold_accuracies)} folds: {sum(fivefold_accuracies)/len(fivefold_accuracies)}')
    logs.append(f"Overall accuracy of predictions for all test folds: {accuracy_score(output_df['encoded_label'], output_df['prediction'])}")
    logs.append(classification_report(output_df['encoded_label'], output_df['prediction'])) 
    logs.append(f"\nAccuracy of predictions for each term:")
    term_accuracies = []
    for term in set(output_df['term']):
        preds = [p for p, t in zip(output_df['prediction'], output_df['term']) if t == term]
        gold = [g for g, t in zip(output_df['encoded_label'], output_df['term']) if t == term]
        acc = accuracy_score(gold, preds)
        term_accuracies.append(acc)
        logs.append(f"{term}: {acc}")
    logs.append(f'\nAverage of accuracies over {len(term_accuracies)} terms: {sum(term_accuracies)/len(term_accuracies)}')

    return logs