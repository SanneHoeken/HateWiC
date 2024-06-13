import torch, pickle, re
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from difflib import get_close_matches
from os import path

class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name):
        super(ContextEncoder, self).__init__()
        self.context_encoder = AutoModel.from_pretrained(encoder_name, output_hidden_states=True)

    def forward(self, input_ids):
        context_output = self.context_encoder(input_ids)
        return context_output

class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name):
        super(BiEncoderModel, self).__init__()
        self.context_encoder = ContextEncoder(encoder_name)

    def context_forward(self, context_input):
        return self.context_encoder.forward(context_input)
    

def find_target_indices(tknzr, example, term):
            
    # encode example and target term
    example_encoding = tknzr.encode(example, truncation=True)
    term_encoding = tknzr.encode(term, add_special_tokens=False)
    
    # find indices for target term
    term_indices = None
    for i in range(len(example_encoding)):
        if example_encoding[i:i+len(term_encoding)] == term_encoding:
            term_indices = (i, i+len(term_encoding))
    
    if not term_indices:
        new_term = None
        new_example = None
        
        # try plural (simple rules)
        if term + 's' in example:
            new_term = term + 's'
        elif term.replace('y', 'ies') in example:
            new_term = term.replace('y', 'ies')
        elif term.replace('man', 'men') in example:
            new_term = term.replace('man', 'men')
        else:
            # try to find the most similar word in the example
            potential_target = get_close_matches(term, example.split(), n=1, cutoff=0.6)
            if len(potential_target) == 1:
                most_similar = re.sub(r'[^\w\s-]','', potential_target[0])
                # replace the most similar word (for which we assume misspelling) with the target term
                new_example = example.replace(most_similar, term)
        
        if new_term or new_example:
            # encode new term or example
            if new_term:
                term_encoding = tknzr.encode(new_term, add_special_tokens=False)
            elif new_example:
                example_encoding = tknzr.encode(new_example, truncation=True)
            # try finding indices again
            for i in range(len(example_encoding)):
                if example_encoding[i:i+len(term_encoding)] == term_encoding:
                    term_indices = (i, i+len(term_encoding))
    
    return term_indices


def extract_biencoder_embedding(model, example_encoding, term_indices, layers):

    # feed example encodings to the model    
    input_ids = torch.tensor([example_encoding])
    encoded_layers = model.context_forward(input_ids)[-1]
    
    # extract selection of hidden layer(s)
    if layers == 'last':
        layers = -1
        vecs = encoded_layers[layers].squeeze(0)
    elif layers == 'lastfour':
        layers = [-4, -3, -2, -1]
        selected_encoded_layers = [encoded_layers[x] for x in layers]
        vecs = torch.mean(torch.stack(selected_encoded_layers), 0).squeeze(0)
    elif layers == 'all':
        vecs = torch.mean(torch.stack(encoded_layers), 0).squeeze(0)
    
    # target word selection 
    vecs = vecs.detach()
    start_idx, end_idx = term_indices
    vecs = vecs[start_idx:end_idx]
    
    # aggregate sub-word embeddings (by averaging)
    vector = torch.mean(vecs, 0)
    
    return vector


def extract_embedding(model, example_encoding, term_indices, layers):

    # feed example encodings to the model    
    input_ids = torch.tensor([example_encoding])
    encoded_layers = model(input_ids)[-1]
    
    # extract selection of hidden layer(s)
    if layers == 'last':
        layers = -1
        vecs = encoded_layers[layers].squeeze(0)
    elif layers == 'lastfour':
        layers = [-4, -3, -2, -1]
        selected_encoded_layers = [encoded_layers[x] for x in layers]
        vecs = torch.mean(torch.stack(selected_encoded_layers), 0).squeeze(0)
    elif layers == 'all':
        vecs = torch.mean(torch.stack(encoded_layers), 0).squeeze(0)
    
    # target word selection 
    vecs = vecs.detach()
    start_idx, end_idx = term_indices
    vecs = vecs[start_idx:end_idx]
    
    # aggregate sub-word embeddings (by averaging)
    vector = torch.mean(vecs, 0)
    
    return vector


def dataid2biencoderembeddings(input_path, example_column, id_column, output_path, encoder_model_name, wsd_biencoder_path, layers, type='token'):

    data = pd.read_csv(input_path)
    tknzr = AutoTokenizer.from_pretrained(encoder_model_name)
    model = BiEncoderModel(encoder_model_name)
    model.load_state_dict(torch.load(wsd_biencoder_path, map_location=torch.device('cpu')), strict=False)
    model.eval()

    embeddings = dict()
    for _, row in tqdm(data.iterrows()):
        example = row[example_column].lower() 
        example_encoding = tknzr.encode(example, truncation=True)
        if type == 'token':
            term_indices = find_target_indices(tknzr, example, row['term'].lower())     
        elif type == 'sentence':
            term_indices = (0, len(example_encoding))
        if term_indices:
            # extract embedding
            vector = extract_biencoder_embedding(model, example_encoding, term_indices, layers=layers)
            embeddings[row[id_column]] = vector
    
    with open(output_path, 'wb') as outfile:
        pickle.dump(embeddings, outfile)


def dataid2embeddings(input_path, example_column, id_column, output_path, model_name, layers, type='token'):

    data = pd.read_csv(input_path)
    tknzr = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    embeddings = dict()
    for _, row in tqdm(data.iterrows()):
        example = row[example_column].lower()
        example_encoding = tknzr.encode(example, truncation=True)
        if type == 'token':
            term_indices = find_target_indices(tknzr, example, row['term'].lower())
        elif type == 'sentence':
            term_indices = (0, len(example_encoding))
        if term_indices:
            # extract embedding
            vector = extract_embedding(model, example_encoding, term_indices, layers=layers)
            embeddings[row[id_column]] = vector
    
    with open(output_path, 'wb') as outfile:
        pickle.dump(embeddings, outfile)


def concatenate_embeddings(embedding_path1, embedding_path2, output_path):
    
    embeddings = dict()

    with open(embedding_path1, 'rb') as infile:
        id2embeddings1 = pickle.load(infile)
    
    with open(embedding_path2, 'rb') as infile:
        id2embeddings2 = pickle.load(infile)

    for id, e1 in id2embeddings1.items():
        e2 = id2embeddings2[id]
        embeddings[id] = torch.cat((e1, e2))

    with open(output_path, 'wb') as outfile:
        pickle.dump(embeddings, outfile)


def get_embedding_file(data_path, id_column, embedding_dir, embedding_type, model, layers):

    model_name = model.rsplit('/',1)[1] if '/' in model else model
    model_name = model_name.rsplit('.',1)[0] if '.' in model_name else model_name

    if 'example' in embedding_type:
        embedding_path = embedding_dir + f'{model_name}-{layers}-examples'
        if not path.exists(embedding_path):
            if 'biencoder' in model_name:
                dataid2biencoderembeddings(data_path, 'example', id_column, embedding_path, 'bert-base-uncased', model, layers)
            else:
                dataid2embeddings(data_path, 'example', id_column, embedding_path, model, layers, type='token')

    if 'generated_definition' in embedding_type:
        definition_path = embedding_dir + f'{model_name}-{layers}-generated_definitions'
        if not path.exists(definition_path):
            if 'biencoder' in model_name:
                dataid2biencoderembeddings(data_path, 'generated_definition', id_column, definition_path, 'bert-base-uncased', model, layers, type='sentence')
            else:
                dataid2embeddings(data_path, 'generated_definition', id_column, definition_path, model, layers, type='sentence')
        if not 'example' in embedding_type:
            embedding_path = definition_path
        else:
            combined_path = embedding_path+'-generated_definitions'
            if not path.exists(combined_path):
                concatenate_embeddings(embedding_path, definition_path, combined_path)
            embedding_path = combined_path
    elif 'definition' in embedding_type:
        definition_path = embedding_dir + f'{model_name}-{layers}-definitions'
        if not path.exists(definition_path):
            if 'biencoder' in model_name:
                dataid2biencoderembeddings(data_path, 'definition', id_column, definition_path, 'bert-base-uncased', model, layers, type='sentence')
            else:
                dataid2embeddings(data_path, 'definition', id_column, definition_path, model, layers, type='sentence')
        if not 'example' in embedding_type:
            embedding_path = definition_path
        else:
            combined_path = embedding_path+'-definitions'
            if not path.exists(combined_path):
                concatenate_embeddings(embedding_path, definition_path, combined_path)
            embedding_path = combined_path
    
    if 'target' in embedding_type:
        target_path = embedding_dir + f'{model_name}-{layers}-targets'
        if not path.exists(target_path):
            if 'biencoder' in model_name:
                dataid2biencoderembeddings(data_path, 'profile_description', id_column, target_path, 'bert-base-uncased', model, layers, type='sentence')
            else:
                dataid2embeddings(data_path, 'profile_description', id_column, target_path, model, layers, type='sentence')
        combined_path = embedding_path+'-targets'
        if not path.exists(combined_path):
            concatenate_embeddings(embedding_path, target_path, combined_path)
        embedding_path = combined_path
    
    return embedding_path