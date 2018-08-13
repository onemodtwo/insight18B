#!/usr/bin/env python
# coding: utf-8

"""The code in this file reads in the raw transcript data taken from the Oyez
project and returns a dataframe of speaker 'turns', each labeled by case, and
speaker or target (with each of the latter two being either 'petitioner',
'respondent' or 'scotus'--the justices are not individually identified.)
"""

import os
import pandas as pd
import pickle


def read_data(fn):
    with open(fn, 'rb') as fp:
        return pickle.load(fp)


def write_data(output, fn):
    with open(fn, 'wb') as fp:
        pickle.dump(output, fp)


def get_initial_labels(turns):
    """
    Each 'section' of the raw transcript record refers to a different
    individual addressing the court. We use this to make an initial labeling
    of speakers. Because it is not always clear which party a speaker is
    advocating for, we default to taking the first two sections as representing
    the petitioner and respondent, respectively (this is always the case). We
    then record the start time of each of these advocates to ensure that we
    label the target of remarks from the bench accurately in the get_target
    function.
    """
    num_advocates = []
    case_advocates = []

    # Pull out the advocates speaking in each case. Find the minimum start time
    # for each.
    advocates_by_case = {cite: case[case['role'] !='scotus_justice'][['speaker', 'section', 'start']].
                         groupby(['speaker', 'section'])['start'].min() for
                         cite, case in turns.groupby('citation')}

    # Count the number of advocates speaking per case and filter out those
    # that have only one.
    for cite in advocates_by_case:
        num_advocates.append({'citation': cite,
                              'num_advocates': len(advocates_by_case[cite])})
    n_advocates_df = pd.DataFrame(num_advocates)
    good_cites = n_advocates_df[n_advocates_df['num_advocates'] > 1]['citation']

    # Identify the first speaker as petitioner advocate and the second as
    # respondent advocate. Note the start time for each. Merge this with the
    # `turns` dataframe.
    for cite in good_cites:
        advocates_by_case[cite] = (advocates_by_case[cite].reset_index()
                                   .sort_values('start').reset_index(drop=True))
        case_advocates.append({'citation': cite, 'petitioner': advocates_by_case[cite].loc[0, 'speaker'],
                               'pet_start': advocates_by_case[cite].loc[0, 'start'],
                               'respondent': advocates_by_case[cite].loc[1, 'speaker'],
                               'res_start': advocates_by_case[cite].loc[1, 'start']})

    case_advocates_df = pd.DataFrame(case_advocates)
    turns_labeled = pd.merge(turns, case_advocates_df, on='citation')
    return turns_labeled


def get_client(row):
    """
    Label each row with a client. The transcripts label the role of
    each speaker. If this is `scotus_justice` we put scotus as the client. The
    identfications made in the `case_advocates` dataframe allow us to label the
    clients for the advocates.
    """
    if row['role'] == 'scotus_justice':
        return 'scotus'
    elif row['speaker'] == row['petitioner']:
        return 'petitioner'
    elif row['speaker'] == row['respondent']:
        return 'respondent'
    else:
        return None


def get_target(row):
    """
    row['client'] == 'scotus' means a justice is speaking. We know the
    comment is addressed to the petitioner if the start time of the remark is
    before the start time of the respondent and vice versa. If the client is
    not scotus, we know that an advocate is speaking to the court and the
    target is labeled as scotus.
    """
    if row['client'] == 'scotus':
        if row['start'] < row['res_start']:
            return 'petitioner'
        else:
            return 'respondent'
    else:
        return 'scotus'


def get_final_labels(turns_labeled):
    """Wrappper to fill 'client' and 'target' fields in turns dataframe."""
    turns_labeled['client'] = turns_labeled.apply(get_client, axis=1)
    turns_labeled = turns_labeled[turns_labeled['client'].notnull()]
    turns_labeled['target'] = turns_labeled.apply(get_target, axis=1)
    turns_labeled.drop(['petitioner', 'respondent'], axis=1, inplace=True)
    return turns_labeled


def filter_cases(turns_labeled):
    """Filters cases that have remarks from and to each party"""
    ll = []
    for case, _ in turns_labeled.groupby(['citation', 'client', 'target']):
        ll.append({'citation': case[0], 'from': case[1], 'to': case[2]})
    ll_df = pd.DataFrame(ll)

    # We want only those cases that have 4 parts: remarks from each advocate
    # to the court and comments or questions to each advocate from the court.
    good_cases = []
    for cite, df in ll_df.groupby('citation'):
        if len(df) == 4:
            good_cases.append(cite)

    turns_labeled = turns_labeled[turns_labeled['citation'].isin(good_cases)]
    return turns_labeled


def run(infile, outfile, base):
    base_path = os.path.expanduser(base)
    infile = os.join.path(base_path, infile)
    outfile = os.join.path(base_path, outfile)
    turns_labeled = get_initial_labels(read_data(infile))
    turns_labeled = get_final_labels(turns_labeled)
    turns_labeled = filter_cases(turns_labeled)
    write_data(turns_labeled, outfile)


if __name__ == '__main__':
    run(infile='transcripts_df.p', outfile='turns_labeled.p',
        base='~/projects/insight/__data__/SCOTUS')
