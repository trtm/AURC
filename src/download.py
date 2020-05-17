#!/usr/bin/env python

import io
import time
import justext
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator


def download_page(common_crawl_data):

    offset, length = int(common_crawl_data['offset']), int(common_crawl_data['length'])
    offset_end = offset + length - 1

    prefix = 'https://commoncrawl.s3.amazonaws.com/'

    resp = requests.get(prefix + common_crawl_data['filename'], headers={'Range': 'bytes={}-{}'.format(offset, offset_end)})
    raw_data = io.BytesIO(resp.content)

    uri = None
    page = None
    for record in ArchiveIterator(raw_data, arc2warc=True):
        uri = record.rec_headers.get_header('WARC-Target-URI')
        R = record.content_stream().read()
        try:
            page = R.strip().decode('utf-8')
        except:
            page = R.strip().decode('latin1')
    return uri, page


def main():
    parser = argparse.ArgumentParser(description='Data Download for Fine-Grained Argument Unit Recognition and Classification')
    parser.add_argument('-d', '--data', type=str, default='data/AURC_DATA.tsv', help='Location of the AURC dataset.')
    parser.add_argument('-t', '--topic', type=str, required=True, help='One of the eight topics to download.')
    parser.add_argument('-w', '--wait', type=float, default=0.0, help='Wait in seconds after every download.')
    args = parser.parse_args()

    # Load Data
    AURC_DATA = pd.read_csv(args.data, sep='\t')
    
    sentence_hashes = AURC_DATA.loc[AURC_DATA.topic==args.topic].sentence_hash.values.tolist()
    assert len(set(sentence_hashes)) == 1000

    for i in tqdm( range( len( sentence_hashes ) ) ):
        sH = sentence_hashes[i]
        AD = AURC_DATA.loc[AURC_DATA.sentence_hash==sH]

        if type( AD.sentence.values.tolist()[0] ) == float: # empty
    
            CommonCrawlData = dict()
            CommonCrawlData['url'] = AD['url'].values.tolist()[0]
            CommonCrawlData['offset'] = AD['doc_offset'].values.tolist()[0]
            CommonCrawlData['length'] = AD['doc_length'].values.tolist()[0]
            CommonCrawlData['filename'] = AD['warc_file'].values.tolist()[0]

            # Download Data
            uri, page = download_page(CommonCrawlData)
    
            paragraphs = justext.justext(page, justext.get_stoplist("English"))
            paragraph_text = []
            for paragraph in paragraphs:
                if not paragraph.is_boilerplate or sH=='1216640d6684f99ad46b2ecbdbfdcf34':
                    paragraph_text.append(paragraph.text)
    
            cleaned_text = " ".join(paragraph_text)
            cleaned_text = cleaned_text.replace('\n',' ')
            assert len(cleaned_text.splitlines()) == 1

            start_in_text = int(AD['start_in_text'].values.tolist()[0])
            end_in_text = start_in_text + int(AD['length_in_text'].values.tolist()[0])
            sentence = cleaned_text[start_in_text:end_in_text]
            AURC_DATA.loc[AURC_DATA.sentence_hash==sH, ['sentence']] = sentence
            assert len(AURC_DATA) == 8000

            # Save Data
            AURC_DATA.to_csv(args.data, sep='\t', index_label=False, index=False)

            time.sleep( args.wait )


if __name__ == "__main__":
    main()
