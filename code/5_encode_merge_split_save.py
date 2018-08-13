#!/usr/bin/env python
# coding: utf-8

"""The code in this file collapses and target encodes the structured variables
used in the project, merges it with the text stats and text vectors, and
produces training and testing data for modeling each justice.
"""

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from utils import TargetEncoder


def read_data(fn):
    with open(fn, 'rb') as fp:
        return pickle.load(fp)


def write_data(output, fn):
    with open(fn, 'wb') as fp:
        pickle.dump(output, fp)


def prep_structured(struct_fn):
    """This function reads in the downloaded CSV file containing the structured
    data from the Supreme Court Database Project, does some preliminary
    manual collapsing of categories for certain of the indicators used from this
    data, and returns a dataframe keyed by case citation.
    """
    def collapse_parties(row, party):
        state = party + 'State'
        if row[party] in party_type_to_state:
            return (row[state] + 602) if row[state] else row[party]
        elif row[party] in party_type_map:
            return party_type_map[row[party]]
        else:
            return row[party]

    def collapse_disposition(val):
        if val in disposition_map:
            return disposition_map[val]
        else:
            return val

    # Hand mapped values for party (petitioner or respondent) type, lower court
    # disposition and circuit court.
    party_type_to_state = [4, 5, 7, 17, 19, 20, 28]
    party_type_map = {3: 601, 18: 601, 8: 602, 13: 602}
    disposition_map = {3: 13, 4: 13, 5: 14, 8: 14, 6: 15, 7: 15}
    court_circuit_map = {1: 13, 2: 13, 3: 13, 4: 14, 5: 14, 6: 13, 7: 13, 8: 13,
                         9: 22, 10: 99, 12: 9, 13: 99, 14: 13, 15: 99, 16: 99,
                         17: 99, 18: 99, 19: 0, 20: 22, 21: 1, 22: 2, 23: 3,
                         24: 4, 25: 5, 26: 6, 27: 7, 28: 8, 29: 9, 30: 10,
                         31: 11, 32: 12, 41: 11, 42: 11, 43: 11, 44: 9, 45: 9,
                         46: 8, 47: 8, 48: 9, 49: 9, 50: 9, 51: 9, 52: 10, 53: 2,
                         54: 3, 55: 12, 56: 11, 57: 11, 58: 11, 59: 11, 60: 11,
                         61: 11, 62: 9, 63: 9, 64: 9, 65: 7, 66: 7, 67: 7, 68: 7,
                         69: 7, 70: 8, 71: 8, 72: 10, 73: 6, 74: 6, 75: 5, 76: 5,
                         77: 5, 78: 1, 79: 4, 80: 1, 81: 6, 82: 6, 83: 8, 84: 5,
                         85: 5, 86: 8, 87: 8, 88: 9, 89: 8, 90: 9, 91: 1, 92: 3,
                         93: 10, 94: 2, 95: 2, 96: 2, 97: 2, 98: 4, 99: 4, 100: 4,
                         101: 8, 102: 9, 103: 6, 104: 6, 105: 10, 106: 10, 107: 10,
                         108: 9, 109: 3, 110: 3, 111: 3, 112: 1, 113: 1, 114: 4,
                         115: 8, 116: 6, 117: 6, 118: 6, 119: 5, 120: 5, 121: 5,
                         122: 5, 123: 10, 124: 2, 125: 3, 126: 4, 127: 4, 128: 9,
                         129: 9, 130: 4, 131: 4, 132: 7, 133: 7, 134: 10, 150: 5,
                         151: 9, 152: 4, 153: 7, 155: 4, 160: 4, 162: 11, 163: 5,
                         164: 11, 165: 7, 166: 7, 167: 8, 168: 6, 169: 5, 170: 8,
                         171: 3, 172: 3, 173: 2, 174: 4, 175: 6, 176: 3, 177: 3,
                         178: 5, 179: 4, 180: 4, 181: 7, 182: 6, 183: 3, 184: 9,
                         185: 11, 186: 8, 187: 5, 300: 0, 301: 0, 302: 0, 400: 99,
                         401: 99, 402: 99, 403: 11, 404: 8, 405: 9, 406: 2,
                         407: 3, 408: 11, 409: 11, 410: 7, 411: 7, 412: 8,
                         413: 10, 414: 6, 415: 5, 416: 1, 417: 4, 418: 1, 419: 6,
                         420: 8, 421: 5, 422: 8, 423: 9, 424: 1, 425: 3, 426: 2,
                         427: 4, 428: 6, 429: 9, 430: 3, 431: 1, 432: 4, 433: 6,
                         434: 5, 435: 2, 436: 4, 437: 4, 438: 7,
                         439: 10, 440: 12, 441: 8, 442: 10, 443: 9}

    df_justices = pd.read_csv(struct_fn, encoding='latin-1')
    df_justices = df_justices[df_justices['partyWinning'] != 2]
    df_justices['vote'] = df_justices.apply(lambda row: ((row['majority'] -
                                                         row['partyWinning']) %
                                                         2), axis=1)
    df_justices = df_justices[df_justices['vote'].notnull()]

    for party in ['petitioner', 'respondent']:
        df_justices['adj_' + party] = df_justices.apply(collapse_parties,
                                                        axis=1, party=party)

    df_justices['adj_disposition'] = \
        df_justices['lcDisposition'].apply(collapse_disposition)

    df_justices['circuit'] = \
        df_justices['caseSource'].apply(lambda x: court_circuit_map.get(x, None))

    df_justices.fillna(value=999, downcast='infer', inplace=True)

    df_justices = df_justices[['usCite', 'justiceName', 'adj_petitioner',
                               'adj_respondent', 'adj_disposition', 'circuit',
                               'issueArea', 'certReason', 'vote']]

    return df_justices


def merge_split_encode_write(df_justices, stats_fn, vecs_fn, base_path):
    """This function take in the dataframe containing the strurctured data and
    reads in the dataframes containing the text stats and text vectors created
    by make_text_stats.py and make_text_vecs.py, respectively. These dataframes
    are merged, then in the central loop the data are segmented by justice and
    split into train and test sets. The structured variables are then target
    encoded and the data are saved into sources for modeling of each justice.
    """
    JUSTICES = ['WJBrennan', 'BRWhite', 'TMarshall', 'HABlackmun',
                'WHRehnquist', 'JPStevens', 'SDOConnor', 'AScalia', 'AMKennedy',
                'CThomas', 'RBGinsburg', 'SGBreyer']

    enc_cols = ['adj_petitioner', 'adj_respondent', 'adj_disposition',
                'circuit', 'issueArea', 'certReason']

    merged_df = df_justices.merge(pd.merge(read_data(stats_fn),
                                           read_data(vecs_fn), on='citation'),
                                  left_on='usCite', right_on='citation')

    target_encoder = TargetEncoder(smoothing=10, min_samples=50)

    for _justice in JUSTICES:
        features = (merged_df[merged_df['justiceName'] == _justice].
                    drop('vote', axis=1))
        targets = merged_df[merged_df['justiceName'] == _justice]['vote']
        X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                            test_size=0.2,
                                                            random_state=0)
        for col in enc_cols:
            X_train[col], X_test[col] = target_encoder.encode(X_train[col],
                                                              X_test[col],
                                                              y_train)
        write_data((X_train, X_test, y_train, y_test),
                   os.path.join(base_path, _justice + '_model_data.p'))
    return


def run(structured, stats, vecs, base):
    base_path = os.path.expanduser(base)
    struct_fn = os.join.path(base_path, structured)
    stats_fn = os.join.path(base_path, stats)
    vecs_fn = os.join.path(base_path, vecs)
    df_justices = prep_structured(struct_fn)
    merge_split_encode_write(df_justices, stats_fn, vecs_fn, base_path)


if __name__ == '__main__':
    run(structured='SCDB_2017_01_justiceCentered_Citation.csv',
        stats='text_stats.p', vecs='text_vecs.p',
        base='~/projects/insight/__data__/SCOTUS')
