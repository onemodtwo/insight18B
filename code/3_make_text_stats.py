#!/usr/bin/env python
# coding: utf-8

"""The code in tnis file reads in the labeled turns data and outputs a dataframe
with one record per case that includes metrics for the number of times an
advocate was interrupted (cut off) while addressing the court and the proportion
of the advocate's time that he or she--rather than one of the justices--was
speaking.
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


def fix_stop_time(row):
    """Utility function called by get_initial_stats to fix the stop
    time in those records that have missing or faulty stop times.
    """
    if row['stop'] >= row['start']:
        return row['stop']
    elif row['citation'] == row['cite_lag']:
        return row['start_lag']
    else:
        return row['start']


def get_cutoffs(row):
    """Takes in a transcript 'turn' and returns 1 if the speaker was cut off
    and 0 otherwise.

    When speakers are interrupted, the transcript turn ends in a hyphen.
    We can use this to label a turn as being cut off or not and then count
    how many times this occurs."""
    if row['role'] == 'advocate':
        if row['text'].strip()[-1] == '-':
            return 1
    return 0


def get_initial_stats(turns_labeled):
    """Takes in the labeled turns dataframe and returns a dataframe that
    has records with fields for the case citation, the party (petitioner or
    respondent) and values for a normed--by the amount of time the advocate
    spoke--and then standardized value for the number of times the advocate was
    cut off, and the duration as a percentage of the total time allotted and
    used by the advocate that the advocate (as opposed to the justices) spoke.
    """
    turns_labeled['start_lag'] = turns_labeled['start'].shift(-1)
    turns_labeled['cite_lag'] = turns_labeled['citation'].shift(-1)
    turns_labeled['new_stop'] = turns_labeled.apply(fix_stop_time, axis=1)
    turns_labeled['duration'] = (turns_labeled['new_stop'] -
                                 turns_labeled['start'])
    turns_labeled['cut_off'] = turns_labeled.apply(get_cutoffs, axis=1)
    turns_labeled.drop(['start_lag', 'cite_lag'], axis=1)

    turns_gb_adv = (turns_labeled[turns_labeled['client'] != 'scotus'].
                    groupby(['citation', 'client']).
                    agg({'duration': 'sum', 'cut_off': 'sum'}).reset_index())
    turns_gb_adv.columns = ['citation', 'party', 'duration', 'cut_off']

    turns_gb_jus = (turns_labeled[turns_labeled['client'] == 'scotus'].
                    groupby(['citation', 'target']).agg({'duration': 'sum'})
                    .reset_index())
    turns_gb_jus.columns = ['citation', 'target', 'j_duration']

    turns_gb_adv['normed_cut_off'] = (turns_gb_adv['cut_off'] /
                                      turns_gb_adv['duration'])

    turns_gb_adv['scaled_cutoff'] = ((turns_gb_adv['normed_cut_off'] -
                                      turns_gb_adv['normed_cut_off'].mean()) /
                                     turns_gb_adv['normed_cut_off'].std())

    text_stats = pd.merge(turns_gb_adv, turns_gb_jus,
                          left_on=['citation', 'party'],
                          right_on=['citation', 'target'])

    text_stats['duration_pct'] = (text_stats['duration'] /
                                  (text_stats['duration'] +
                                   text_stats['j_duration']))

    return text_stats[['citation', 'party', 'duration_pct', 'scaled_cutoff']]


def get_stats_by_case(text_stats):
    """Flattens the stats records to have one record per case including stats
    for both the petitioner and respondent.
    """
    cols = list(text_stats.columns)

    text_stats_pet = text_stats[text_stats['party'] == 'petitioner']
    text_stats_res = text_stats[text_stats['party'] == 'respondent']

    text_stats_pet.columns = ['citation'] + ['p_' + field for field in cols[1:]]
    text_stats_res.columns = ['citation'] + ['r_' + field for field in cols[1:]]

    stat_feats = pd.merge(text_stats_pet, text_stats_res, on='citation')
    return stat_feats.drop(['p_party', 'r_party'], axis=1)


def run(infile, outfile, base):
    base_path = os.path.expanduser(base)
    infile = os.join.path(base_path, infile)
    outfile = os.join.path(base_path, outfile)
    text_stats = get_initial_stats(read_data(infile))
    stat_features = get_stats_by_case(text_stats)
    write_data(stat_features, outfile)


if __name__ == '__main__':
    run(infile='turns_labeled.p', outfile='text_stats.p',
        base='~/projects/insight/__data__/SCOTUS')
