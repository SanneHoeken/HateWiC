import json
from embeddings import get_embedding_file
from tenfold_eval import evaluate
from dinu_eval import dinu_evaluate

def run(data_path, id_column, label_column, label_encoder,
                   embedding_dir, predictions_dir, logs_path,
                   models, model_layers, clf, embedding_types,
                   splitby, params=dict(), random_split_seed=12):
    
    logs = []
    for model in models:
        model_name = model.rsplit('/',1)[1] if '/' in model else model
        model_name = model_name.rsplit('.',1)[0] if '.' in model_name else model_name
        for embedding_type in embedding_types:
            experiment_description = f'\n{clf} / {model_name} / {embedding_type} embeddings / {model_layers} layer(s) / split by {splitby}\n'.upper()
            logs.append(experiment_description)
            logs.append(f"Hyperparameters: {params}")
            print(experiment_description)
            # default embeddings are token embeddings of example usages
            embedding_path = get_embedding_file(data_path, id_column, embedding_dir, embedding_type, model, model_layers)
            predictions_path = predictions_dir + f'{clf}-{model_name}-{embedding_type}-{model_layers}-splitby{splitby}.csv'
            if 'dinu' in data_path:
                experiment_logs = dinu_evaluate(data_path, label_column, embedding_path, predictions_path, clf, params, random_split_seed)
            else:
                experiment_logs = evaluate(data_path, id_column, label_column, label_encoder, embedding_path, 
                                            predictions_path, clf, params, random_split_seed, splitby) 
            logs.extend(experiment_logs)
            #print(experiment_logs)
    
    with open(logs_path, 'w') as outfile:
        for string in logs:
            outfile.write(string+'\n')