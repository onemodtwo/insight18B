#!/usr/bin/env python
# coding: utf-8

"""The code in this file takes the labeled turns dataframes and produces a
dataframe with records representing the four text vectors for words spoken by
each advocate to the court and words spoken by justices to each advocate during
the course of oral arguments. Each record in the resulting dataframe is keyed
by the case citaion.
"""

from functools import reduce
import gensim
import os
import pandas as pd
import pickle


def read_data(fn):
    with open(fn, 'rb') as fp:
        return pickle.load(fp)


def write_data(output, fn):
    with open(fn, 'wb') as fp:
        pickle.dump(output, fp)


def prep_text(turns_labeled):
    """Takes in the labeled turns dataframe and returns a dataframe with text
    prepared to be vectorized.

    The labeled turn data is collapsed into 'documents', one for each advocate's
    remarks to the court and one for the court's remarks to each advocate. The
    raw text is then preprossessed by removal of stop words, creation of bigrams,
    and each 'document' is given a unique integer label for use in creating
    document vectors. The resulting dataframe has four records per case.
    """
    text_df = (turns_labeled.groupby(['citation', 'client', 'target']).
               agg({'text': lambda x: '{}'.format(' '.join(x))}).reset_index())
    text_df.columns = ['citation', 'speaker', 'target', 'text']

    STOPS = ['inaudible', 'voice overlap', 'the', 'may']
    data = text_df['text']
    for s in STOPS:
        data = [text.lower().replace(s, '') for text in data]
    data = [gensim.utils.simple_preprocess(text, min_len=3, deacc=True)
            for text in data]
    text_df['prepped_text'] = data

    phrases = gensim.models.phrases.Phrases(data)
    bigram = gensim.models.phrases.Phraser(phrases)

    text_df['prepped_bg'] = bigram[data]
    text_df['text_label'] = [[i] for i in range(len(text_df))]
    text_df['labeled_text'] = text_df.apply(lambda row: (gensim.models.doc2vec.
                                                         TaggedDocument(
                                                             row['prepped_bg'],
                                                             row['text_label'])),
                                            axis=1)

    text_df['label'] = text_df['text_label'].apply(lambda x: x[0])
    text_df = text_df[['citation', 'speaker', 'target',
                       'label', 'labeled_text']]
    return text_df


def make_model(input2vec):
    """Takes the labeled text created by prep_text, trains a doc2vec model,
    and returns the model.
    """
    model = gensim.models.Doc2Vec(vector_size=100, window=10, min_count=5,
                                  workers=4, train_lbls=False, alpha=0.025,
                                  min_alpha=0.025)  # use fixed learning rate
    model.build_vocab(input2vec)
    for epoch in range(10):
        model.train(input2vec, total_examples=model.corpus_count,
                    epochs=model.epochs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    return model


def make_vectors(model, text_df):
    """Takes in the doc2vec model and the text document dataframe and
    returns a dataframe with one record per case that has representations of
    all four text vectors (from each advocate to the court and to each advocate
    from the court.

    The function merges the text dataframe with the vectors using the labels
    created in prep_text. Each component of each vector is a column in the
    resulting dataframe, so that it ultimately has 401 columns (one for the case
    citation and 100 for each vector.)
    """
    vecs = []
    for label in range(len(model.docvecs)):
        vecs.append({'label': label, 'vec': model[label]})

    text_vecs = pd.merge(text_df, pd.DataFrame(vecs), on='label')
    text_vecs.drop(['label', 'labeled_text'], axis=1, inplace=True)

    v_len = len(text_vecs.loc[0, 'vec'])
    text_vecs[['tv_' + str(i) for i in range(1, v_len + 1)]] = \
        pd.DataFrame(text_vecs['vec'].values.tolist(), index=text_vecs.index)

    text_vecs.drop(['vec'], inplace=True, axis=1)
    cols = text_vecs.columns

    # We separate the vectors from each advocate and from the court to each
    # advocate into separate dataframes that we can then merge together to
    # create a dataframe that has all four of these vectors in one record per
    # case.
    vecs_fr_pet = (text_vecs[text_vecs['speaker'] == 'petitioner'].
                   drop(['speaker', 'target'], axis=1))
    vecs_fr_res = (text_vecs[text_vecs['speaker'] == 'respondent'].
                   drop(['speaker', 'target'], axis=1))
    vecs_to_pet = (text_vecs[text_vecs['target'] == 'petitioner'].
                   drop(['speaker', 'target'], axis=1))
    vecs_to_res = (text_vecs[text_vecs['target'] == 'respondent'].
                   drop(['speaker', 'target'], axis=1))

    vecs_to_pet.columns = ['citation'] + ['to_p_' + field for field in cols[3:]]
    vecs_to_res.columns = ['citation'] + ['to_r_' + field for field in cols[3:]]
    vecs_fr_pet.columns = ['citation'] + ['fr_p_' + field for field in cols[3:]]
    vecs_fr_res.columns = ['citation'] + ['fr_r_' + field for field in cols[3:]]

    # Merge the four dataframes and return the result.
    vec_dfs = [vecs_fr_pet, vecs_to_pet, vecs_fr_res, vecs_to_res]
    return reduce(lambda lt, rt: pd.merge(lt, rt, on='citation'), vec_dfs)


def run(infile, outfile, modelfile, base):
    base_path = os.path.expanduser(base)
    infile = os.join.path(base_path, infile)
    outfile = os.join.path(base_path, outfile)
    text_df = prep_text(read_data(infile))
    model = make_model(text_df['labeled_text'])
    model.save(os.path.join(base_path, modelfile))
    text_vectors = make_vectors(model, text_df)
    write_data(text_vectors, outfile)


if __name__ == '__main__':
    run(infile='turns_labeled.p', outfile='text_vecs.p',
        modelfile='doc2vec_model.p',
        base='~/projects/insight/__data__/SCOTUS')
