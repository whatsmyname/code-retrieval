import argparse
import json
import logging
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
import pickle
import tensorflow_hub as hub
import spacy

nlp = spacy.blank("en")
encoding="utf-8"
def word_tokenize(sent, verbose=None):
    """
    :param sent: document {text, phrase, word}
    :param verbose: None, just tokenize on sent
           1, lowercase and tokenize on sent
           2, lemmatize(lowercased) and tokenize on sent
           3, tokenize and remove stopwords on sent
           4, lowercase, tokenize and remove stopwords on sent
           5, lemmatize(lowercased), tokenize and remove stopwords on sent
    :return:
    """
    doc = nlp(sent)
    new_doc = []
    for token in doc:
        if (verbose is None) or (verbose == 3 and not token.is_stop):
            new_doc.append(token.orth_)
        elif (verbose == 1) or (verbose == 4 and not token.is_stop):
            new_doc.append(token.lower_)
        elif (verbose == 2) or (verbose == 5 and not token.is_stop):
            new_doc.append(token.lemma_)
        else:
            pass
    return new_doc

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def get_file_contents(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as f:
        content = f.read()
    return content

def load_word_embeddings(path, vocab, embedding_size):
    embeddings = {}
    vocab_size = len(vocab)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            w = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embeddings[w] = vectors

    embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
    num_loaded = 0
    for w, i in vocab.items():
        v = embeddings.get(w)
        if v is not None and i < vocab_size:
            embedding_matrix[i] = v
            num_loaded += 1
        else:
            print('Token {} with {} id is not found'.format(w,v))
    print('Successfully loaded pretrained embeddings for {}/{} words'.format(num_loaded, vocab_size))
    embedding_matrix = embedding_matrix.astype(np.float32)
    return embedding_matrix

def vocabulary_processor(tokenized_docs):
    indx_to_voc = {0: '<PAD>'} #, 1: '<S>', 2:'</S>',3:'<UNK>'
    voc_to_indx = {'<PAD>':0} # , '<S>':1, '</S>':2,'<UNK>':3
    offset = 1 # 4
    for tokens in tokenized_docs:
        for token in tokens:
            if token not in voc_to_indx:
                voc_to_indx[token] = offset
                indx_to_voc[offset] = token
                offset +=1
    print('{} vocs are created'.format(offset))
    return indx_to_voc, voc_to_indx

def fit_vocab_to_documents(tokenized_docs, voc_to_indx):
    document_lists = []
    for doc_indx, tokens in enumerate(tokenized_docs):
        token_list = []
        #token_list.append(voc_to_indx['<S>'])
        for token in tokens:
            try:
                token_list.append(voc_to_indx[token])
            except:
                print('{} token in Doc {} could not found'.format(token,doc_indx))
        #token_list.append(voc_to_indx['</S>'])
        document_lists.append(token_list)
    document_lists = np.array(document_lists)
    return document_lists

def reversedEnumerate(l):
    return zip(range(len(l)-1, -1, -1), l)

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """
    with tf.name_scope('l2_form') as scope:
        square_sum = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True, name='square_sum')
        norm = tf.sqrt(tf.maximum(square_sum, tf.keras.backend.epsilon()), name='norm')
    return norm

def pairwise_cosine_sim(A, B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions
    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point
    """
    with tf.name_scope('pairwise_cosine_sim') as scope:
        A_mag = l2_norm(A, axis=2)
        B_mag = l2_norm(B, axis=2)
        num = tf.keras.backend.batch_dot(A, tf.keras.backend.permute_dimensions(B, (0, 2, 1)))
        den = (A_mag * tf.keras.backend.permute_dimensions(B_mag, (0, 2, 1)))
        dist_mat = num / den
    return dist_mat

def pairwise_euclidean_distances(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
      A,    [m,d] matrix
      B,    [n,d] matrix
    Returns:
      D,    [m,n] matrix of pairwise distances
    """
    with tf.variable_scope('pairwise_euclidean_dist'):
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D

def evaluation_metrics(questions, paragraphs, labels, params, k=None, distance_type='cosine', functions=None):
    """
     question [n x d] tensor of n rows with d dimensions
     paragraphs [m x d] tensor of n rows with d dimensions
     params config
     """
    if distance_type == 'cosine':
        questions, labels, paragraphs, scores = pairwise_expanded_cosine_similarities(questions, labels, paragraphs)
    else:
        scores= pairwise_euclidean_distances(questions, paragraphs)
        scores = tf.negative(scores)
        #number_of_questions = tf.to_int64(tf.shape(questions)[0])
    formatted_labels = tf.reshape(tf.one_hot(labels, paragraphs.shape[0]), [-1,paragraphs.shape[0]])
    metrics = dict()
    for _function in [functions] if functions is not None else params.metrics["functions"]:
        metrics[_function] = dict()
        for _k in [k] if k is not None else params.metrics["top_k"]:
            with tf.name_scope('top_k_{}'.format(_k)) as k_scope:
                #founds, _, __ = calculate_top_k(scores, labels, _k, distance_type)
                # total_founds_in_k = tf.reduce_sum(founds, name='{}_reduce_sum'.format(_k))
                if _function.lower() == "recall":
                    labels = tf.cast(labels, dtype=tf.int64)
                    _, value = tf.metrics.recall_at_k(labels, scores, _k)
                    metrics[_function][_k] = value
                elif _function.lower() == "precision":
                    labels = tf.cast(labels, dtype=tf.int64)
                    _, value = tf.metrics.precision_at_k(labels, scores, _k)
                    metrics[_function][_k] = value
                elif _function.lower() == "map":
                    labels = tf.cast(labels, dtype=tf.int64)
                    _, value = tf.metrics.average_precision_at_k(labels, scores, _k)
                    metrics[_function][_k] = value
                elif _function.lower() == "dcg":
                    labels = tf.cast(labels, dtype=tf.float32)
                    _, value = tfr.metrics.discounted_cumulative_gain(formatted_labels, scores, topn=_k)
                    metrics[_function][_k] = value
                elif _function.lower() == "ndcg":
                    labels = tf.cast(labels, dtype=tf.float32)
                    _, value = tfr.metrics.normalized_discounted_cumulative_gain(formatted_labels, scores, topn=_k)
                    metrics[_function][_k] = value
                else:
                    labels = tf.cast(labels, dtype=tf.float32)
                    if 'all' in metrics[_function]:
                        continue
                    if _function.lower() == "arp":
                        _, value_all = tfr.metrics.average_relevance_position(formatted_labels, scores)
                    elif _function.lower() == "mrp":
                        _, value_all = tfr.metrics.mean_reciprocal_rank(formatted_labels, scores)
                    else:
                        raise ValueError("No evaluation function is found : {}".format(_function))
                    metrics[_function]['all'] = value_all
    return metrics  # recalls, rates #recalls / number_of_questions)

def pairwise_expanded_cosine_similarities(questions, labels, paragraphs):
    # in order to support batch_size feature, we expanded dims for 1
    paragraphs = tf.expand_dims(paragraphs, axis=0)
    labels = tf.expand_dims(labels, axis=0)
    questions = tf.expand_dims(questions, axis=0)
    # now question, paragraphs pairwise calculation
    cosine_similarities = pairwise_cosine_sim(questions, paragraphs)
    return questions, labels, paragraphs, cosine_similarities

def save_as_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_module(module_url = "https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False):
    if trainable:
        embed = hub.Module(module_url, trainable)
    else:
        embed = hub.Module(module_url)
    return embed

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')