#!/usr/bin/env python

import json
import argparse
import pandas as pd

from utils import get_full_annotation, parse_data

from transformers import BertTokenizer

def main():
    parser = argparse.ArgumentParser(description='Data Download for Fine-Grained Argument Unit Recognition and Classification')
    parser.add_argument('-a', '--aurc_data', type=str, default='./data/AURC_DATA.tsv', help='Location of the AURC dataset.')
    parser.add_argument('-d', '--domain_split_data', type=str, default='./data/AURC_DOMAIN_SPLITS.tsv', help='Location of the AURC domain split data.')
    parser.add_argument('-s', '--sentence_level_data', type=str, default='./data/AURC_SENTENCE_LEVEL_STANCE.tsv', help='Location of the AURC sentence level data.')
    parser.add_argument('-o', '--output_file', type=str, default='./data/AURC_DATA_dict.json', help='Location of the prepared output file.')
    parser.add_argument('-n', '--num_labels', type=int, default=3, help='Using either 3 (pro, con, non) or 2 (arg, non) labels.')
    args = parser.parse_args()

    ############################################################################
    # Load Data
    print(args.aurc_data)
    AURC_DATA = pd.read_csv(args.aurc_data, sep='\t')
    print(len(AURC_DATA))

    for sentence in AURC_DATA[['sentence']].values.tolist():
        assert type(sentence)!=str

    AD = AURC_DATA[['sentence_hash','topic','sentence','merged_segments']]
    print(len(AD))
    
    topics = sorted(set(AURC_DATA.topic.values.tolist()))
    print(len(topics), topics)
    
    print(args.domain_split_data)
    AURC_DOMAIN_SPLITS = pd.read_csv(args.domain_split_data, sep='\t')
    print(len(AURC_DOMAIN_SPLITS))
    
    print(args.sentence_level_data)
    AURC_SENTENCE_LEVEL_STANCE = pd.read_csv(args.sentence_level_data, sep='\t')
    print(len(AURC_SENTENCE_LEVEL_STANCE))
    
    
    ############################################################################
    # Prepare Data
    AURC_DATA_dict = dict()

    ID2LABEL = dict()
    ID2LABEL['p'] = 'pro'
    ID2LABEL['c'] = 'con'
    ID2LABEL['n'] = 'non'
    ID2LABEL['a'] = 'arg'
    
    for count_t, topic in enumerate(topics):
        print(count_t, topic)
        AD = AURC_DATA.loc[AURC_DATA.topic==topic]
        print(len(AD))
        for count_s, (sH, sentence, ms) in enumerate(AD[['sentence_hash','sentence','merged_segments']].values.tolist()):
            assert type(sentence) == str
            AD_dict = dict()
            # ---
            processed_sentence, COLORED_SENTENCE, ts_spacy, ts_spacy_labels, ts_bert, ts_bert_labels = parse_data(
                sentence_hash=sH, sentence=sentence, merged_segments=ms, num_labels=args.num_labels)
            # ---
            assert len(ts_spacy)==len(ts_spacy_labels)
            assert len(ts_bert)==len(ts_bert_labels)
            # ---
            ADS = AURC_DOMAIN_SPLITS.loc[AURC_DOMAIN_SPLITS.sentence_hash==sH]
            InDomainSet = ADS[['In-Domain']].values.tolist()[0][0]
            CrDomainSet = ADS[['Cross-Domain']].values.tolist()[0][0]
            # ---
            sentence_level_stance = AURC_SENTENCE_LEVEL_STANCE.loc[AURC_SENTENCE_LEVEL_STANCE.sentence_hash==sH][['sentence_level_stance']].values.tolist()[0][0]
            # ---
            AD_dict['sentence_hash'] = sH
            AD_dict['sentence'] = sentence
            AD_dict['In-Domain'] = str(InDomainSet)
            AD_dict['Cross-Domain'] = str(CrDomainSet)
            AD_dict['sentence_level_stance'] = sentence_level_stance
            AD_dict['tokenized_sentence_spacy'] = " ".join([str(t) for t in ts_spacy])
            AD_dict['tokenized_sentence_spacy_labels'] = " ".join([ ID2LABEL[ str(list(set(l))[0]) ] for l in ts_spacy_labels])
            AD_dict['tokenized_sentence_bert'] = " ".join([str(t) for t in ts_bert])
            AD_dict['tokenized_sentence_bert_labels'] = " ".join([ ID2LABEL[ str(list(set(l))[0]) ] for l in ts_bert_labels])
            # ---
            try:
                AURC_DATA_dict[topic].append(AD_dict)
            except:
                AURC_DATA_dict[topic] = [AD_dict]
            # ---
        assert len(AD) == len(AURC_DATA_dict[topic])
        print()
    
    
    ############################################################################
    # Save Data
    print(args.output_file)
    with open(args.output_file,'w') as my_file:
        json.dump( AURC_DATA_dict, my_file, sort_keys=True, indent=4, separators=(',', ': ') )

if __name__ == "__main__":
    main()
