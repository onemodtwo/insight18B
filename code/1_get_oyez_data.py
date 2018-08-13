#!/usr/bin/env python
# coding: utf-8

"""This code in this file contains code to pull down oral argument transcripts
from the Oyez Project directed by Cornell University Law school.
"""

import os
import pandas as pd
import pickle
import requests
from utils import get_logger


def get_summaries(url, start, end):
    """Get the case summaries for the specified terms."""
    summaries = requests.get(url).json()
    return [summary for summary in summaries if
            (len(summary['term']) == 4 and int(summary['term']) >= start and
             int(summary['term']) <= end)]


def get_argument_url_sections(case):
    """
    Get the url of the oral argument and the sections of the
    argument. The programming is highly defensive to guard
    against many bad entries in the data.
    """
    argument_url, sections = None, None
    if ((type(case.get('oral_argument_audio')) == list) and
       (len(case.get('oral_argument_audio')) > 0) and  # non-empty list
       (type(case.get('oral_argument_audio')[0]) == dict) and
       (case['oral_argument_audio'][0].get('href'))):  # non-empty dict entry
        argument_url = case['oral_argument_audio'][0]['href']
    if argument_url:
        transcript_json = requests.get(argument_url).json()
        transcript = transcript_json.get('transcript')
        if type(transcript) == dict:  # Check this field too is not empty.
            sections = transcript.get('sections')
    return {'argument_url': argument_url, 'sections': sections}


def get_records(url, start, end, logfile):
    """
    Main program execution Read json record for each case and extract
    required fields for case records and transcripts. The data in case records
    overlaps significantly with Supreme Court Database information, so it
    is not critical if it is missing here.
    """
    logger = get_logger(__name__, logfile)
    logger.info("Transcript retrieval started.")
    records = {'cases': [], 'transcripts': []}
    summaries = get_summaries(url, start, end)
    for case in summaries:
        case_json = requests.get(case['href']).json()
        argument_dict = get_argument_url_sections(case_json)
        argument_url = argument_dict['argument_url']
        sections = argument_dict['sections']
        if sections:
            try:  # Try to construct case record from meta-data.
                record = {}
                advocates = []
                check_advocates = 0
                citation = case_json.get('citation')
                if citation:
                    record['citation'] = (citation.get('volume') + ' U.S. ' +
                                          citation.get('page'))
                for field in ['docket_number', 'name', 'term', 'first_party',
                              'second_party', 'first_party_label',
                              'second_party_label']:
                    record[field] = case_json.get(field)
                record['num_arguments'] = len(case_json['oral_argument_audio'])
                decisions = case_json.get('decisions')
                if ((type(decisions) == list) and (len(decisions) > 0) and
                   (type(decisions[0]) == dict)):  # Is the decision recorded?
                    record['num_decisions'] = len(decisions)
                    for field in ['decision_type', 'winning_party',
                                  'majority_vote', 'minority_vote']:
                        record[field] = decisions[0].get(field)
                    votes = decisions[0].get('votes')
                    if votes:
                        case_votes = []
                        for vote in votes:
                            member = vote.get('member')
                            if member:
                                j_name = member.get('name')
                            j_vote = vote.get('vote')
                            opinion_type = vote.get('opinion_type')
                            joining = ([justice['name'] for justice
                                        in vote.get('joining')]
                                       if vote.get('joining') else 'none')
                            case_votes.append((j_name, j_vote, opinion_type,
                                               joining))
                        record['votes'] = case_votes
                advs = case_json.get('advocates')
                if type(advs) == list:  # Are the advocates listed?
                    for advocate in advs:
                        a_id, a_name, client = None, None, None
                        adv = advocate.get('advocate')
                        if adv:
                            a_id = adv.get('ID')
                            a_name = adv.get('name')
                        desc = advocate.get('advocate_description')
                        if desc:  # Are the advocates labeled as to party?
                            if ((desc.lower().find('petitioner') > 0) or
                               (desc.lower().find('appellant') > 0)):
                                client = 'petitioner'
                            elif ((desc.lower().find('respondent') > 0) or
                                  (desc.lower().find('appellee') > 0)):
                                client = 'respondent'
                            elif (desc.lower().
                                  find(record['first_party'].lower()) > 0):
                                client = record['first_party_label'].lower()
                            elif (desc.lower().
                                  find(record['second_party'].lower()) > 0):
                                client = record['second_party_label'].lower()
                            else:
                                client = advocate['advocate_description']
                                check_advocates = 1
                        else:
                            client = None
                            check_advocates = 1
                            # Flag that we don't know who is representing whom.
                        advocates.append((a_id, a_name, client))
                record['advocates'] = advocates
                record['argument_url'] = argument_url
                record['check_advocates'] = check_advocates
                records['cases'].append(record)
            except Exception:
                logger.exception('Error processing {}\n'.format(case['href']))
                continue
            logger.info('Case records built. Moving on to transcripts.')
            try:  # Now we want to pull the transcript data.
                for num, section in enumerate(sections, 1):
                    turns = section.get('turns')
                    if type(turns) == list:  # Speaker turns is not empty.
                        for turn in turns:
                            tr_record = {}
                            tr_record['section'] = num
                            for field in ['citation', 'name', 'term',
                                          'docket_number']:
                                tr_record[field] = record[field]
                            if type(turn) == dict:  # Turn is not empty.
                                speaker = turn.get('speaker')
                                if type(speaker) == dict:  # Info about speaker.
                                    tr_record['speaker'] = speaker.get('name')
                                    if speaker.get('roles'):
                                        tr_record['role'] = \
                                            speaker['roles'][0]['type']
                                    else:
                                        tr_record['role'] = 'advocate'
                                else:
                                    tr_record['speaker'] = None
                                    tr_record['role'] = None
                                text_blocks = turn.get('text_blocks')
                                if ((type(text_blocks) == list) and
                                   (len(text_blocks) > 0)):
                                    tr_record['start'] = text_blocks[0]['start']
                                    tr_record['stop'] = text_blocks[-1]['stop']
                                    tr_record['text'] = ' '.join([block['text']
                                                                  for block in
                                                                  turn['text_blocks']])
                            records['transcripts'].append(tr_record)
            except Exception:
                logger.exception('Error processing {}\n'.format(argument_url))
                continue
            logger.info('Transcript records built.')
            return records


def dump_data(records, cases_fn, transcripts_fn):
    cases_df = pd.DataFrame(records['cases'])
    transcripts_df = pd.DataFrame(records['transcripts'])
    with open(cases_fn, 'wb') as case_fp:
        pickle.dump(cases_df, case_fp)
    with open(transcripts_fn, 'wb') as trans_fp:
        pickle.dump(transcripts_df, trans_fp)
    return


def run(url, start, end, logfile, cases_out, trans_out, base):
    base_path = os.path.expanduser(base)
    records = get_records(url=url, start=start, end=end, logfile=logfile)
    dump_data(records, cases_fn=os.path.join(base_path, cases_out),
              transcripts_fn=os.path.join(base_path, trans_out))
    return


if __name__ == '__main__':
    run(records_url='https://api.oyez.org/cases?per_page=0', start=1956,
        end=2017, logfile='logs/oyez_error.log', cases_out='cases_df.p',
        trans_out='transcripts_df.p',
        base='~/projects/insight/__data__/SCOTUS')
