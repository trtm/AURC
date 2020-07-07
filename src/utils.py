#!/usr/bin/env python

from itertools import chain
from sty import fg, bg, ef, rs, RgbFg

import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.gold import align

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')


class InputFeatures(object):
    '''
        A single set of features of data.
    '''

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids, sentence_hash):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.sentence_hash = sentence_hash


def get_data_with_labels(all_input_tokens, all_y_true, all_y_pred):
    count_s = 0
    DATA = dict()
    for count_t, (tokens, y_true, y_pred)  in enumerate(zip(all_input_tokens, all_y_true, all_y_pred)):
        seq_len = [i for i,j in enumerate(tokens) if j=='[SEP]'][0]
        tokenized_sequence = " ".join(tokens[1:seq_len]).replace(" ##","")
        assert len(tokenized_sequence.split(" "))==len(y_true)==len(y_pred)
        TOKENS = []
        stance_per_token_pred = []
        stance_per_token_true = []
        for pos, (token, pred, true) in enumerate(zip(tokenized_sequence.split(" "), y_pred, y_true)):
            TOKENS.append(token)
            stance_pred = pred
            stance_true = true
            stance_per_token_pred.append((pos, stance_pred))
            stance_per_token_true.append((pos, stance_true))
        assert len(TOKENS)==len(stance_per_token_pred)==len(stance_per_token_true)
        sentence = " ".join(TOKENS)
        assert len(TOKENS)==len(y_true)==len(y_pred)
        DATA[count_t] = dict()
        DATA[count_t]['sentence'] = sentence
        DATA[count_t]['y_true'] = " ".join([str(y) for y in y_true])
        DATA[count_t]['y_pred'] = " ".join([str(y) for y in y_pred])
    assert len(DATA)==len(all_input_tokens)
    return DATA


def get_full_annotation(sentence, annotation_spans, stance_labels):
    '''
        Add stance/None info for the full sentence.
    '''
    tokenized_sentence = sentence #.split()
    
    AS = []
    if stance_labels and annotation_spans:
        for stance, pos in zip(stance_labels, annotation_spans):
            if stance and pos:
                begin, end = pos
                AS.append([begin, end, stance])
    
    all_annotation_spans_with_stance = []
    if AS:
        l = len(tokenized_sentence)
        next_pos = 0
        for b,e,s in AS:
            if next_pos<b:
                all_annotation_spans_with_stance.append([next_pos, b, 'non'])
            next_pos = e
            all_annotation_spans_with_stance.append([b, e, s])
        if e<l:
            all_annotation_spans_with_stance.append([e, l, 'non'])
    
    return all_annotation_spans_with_stance


def parse_data(sentence_hash, sentence, merged_segments, num_labels=3):
    """
        parse the input data for the token-level task with spacy and bert tokenization
    """
    ################################################################################
    # Correction
    ################################################################################
    # T1: abortion
    if sentence_hash=='03d3a8b95a9b83ab24b894df7dcc2736':
        sentence = sentence.replace("-It","- It")
        merged_segments = merged_segments.replace("(34,132)","(35,132)")
    if sentence_hash=='9ace278617e42a54838a8e3eac04ab9a':
        sentence = sentence.replace("itâs","it's")
        merged_segments = merged_segments.replace(";(75,73);",";(75,71);")
    if sentence_hash=='8d7bcb76a975ff78484593c7d0a9dd3f':
        sentence = sentence.replace("womens","women's")
        merged_segments = merged_segments.replace("(86,132);","(87,132);")
    if sentence_hash=='cc1b29b9f740813f8ac0dec4c0b79196':
        sentence = sentence.replace("-It","- It")
        merged_segments = merged_segments.replace("(86,132);","(87,132);")
    if sentence_hash=='d0c643ed9021857c17ebf87e72ff1c78':
        sentence = sentence.replace("™","")
    if sentence_hash=='e485953c94eaacc19c759feeedf8aff7':
        sentence = sentence.replace(" at www.AbortionRisks.org","")
    if sentence_hash=='82932254624fe4995c4319fb3e019e9a':
        sentence = sentence.replace(" www.realpharmacyrx.co","")
    ################################################################################
    # T2: cloning
    if sentence_hash=='3abb67003f9796c079256939b0bc9121':
        sentence = sentence.replace("Ð²Ð‚™","'")
        merged_segments = merged_segments.replace("(30,142);","(26,138);")
    if sentence_hash=='a43aac556e806a819d3b1e5014439f01':
        sentence = sentence.replace("­Opponents","Opponents")
        merged_segments = merged_segments.replace("(37,131);","(36,131);")
    if sentence_hash=='421357ae30839b4a51f2231f7b8483dd':
        sentence = sentence.replace(" ­the"," the")
        merged_segments = merged_segments.replace("(4,122);","(4,121);")
    if sentence_hash=='a8f2d9d5f9e3087a907b713f94e0be22':
        sentence = sentence.replace("Ð²Ð‚Ñš","")
        sentence = sentence.replace("Ð²Ð‚Ñœ","")
        merged_segments = merged_segments.replace("(25,85);","(25,73);")
    if sentence_hash=='be98156eb94fae06f709e8adf9a8bce5':
        sentence = sentence.replace(" http://support.sitecore.net","")
    if sentence_hash=='d9230caff5bd80281860d6a1ea6c4f75':
        sentence = sentence.replace("reproductive \xadequation?","reproductive equation?")
    if sentence_hash=='59cbacbfe71d66f55c26dd9b10491b06':
        sentence = sentence.replace("\xadIf","If")
        sentence = sentence.replace("ce\xadll","cell")
    if sentence_hash=='4cf24cbd1c1bce40ce2db1feb511711a':
        sentence = sentence.replace(" (http://www.answers.com)","")
    if sentence_hash=='664a340852d067af0a010e4ad5e56361':
        sentence = sentence.replace(" (www.ornl.gov/sci/)","")
    if sentence_hash=='c0881e201c1aeff2d82697d3938690b6':
        sentence = sentence.replace("machine  though","machine - though")
    if sentence_hash=='1c36ec01887f09cb8c585433c037e932':
        sentence = sentence.replace("Bio​ means life and ​ technology​ means","Bio means life and technology means")
    if sentence_hash=='16b7f74c491bdb283cdbbed9961b5509':
        sentence = sentence.replace("having hybrid vigor which",'having "hybrid vigor" which')
    ################################################################################
    # T3: death penalty
    if sentence_hash=='2caf9797cbc50a6a223f9c3b680d7cd4':
        merged_segments = merged_segments.replace("(18,1);(27,1);(29,1);(33,67);","(30,70);")
    if sentence_hash=='72369eefd00ddf2c49ecc4812ac0b892':
        sentence = sentence.replace(" the"," the")
        merged_segments = merged_segments.replace("(28,183);","(27,183);")
    if sentence_hash=='e1a7f4bff1e6bc743b90c9d53a6413b1':
        sentence = sentence.replace("  82 "," 82 ")
        merged_segments = merged_segments.replace("(0,200);","(0,198);")
    if sentence_hash=='ab928248c0f0bfa7d59c5caabd69beca':
        sentence = sentence.replace('claims-"because','claims - " because')
        sentence = sentence.replace("immoral-","immoral -")
        merged_segments = merged_segments.replace("(17,97);(124,79);","(17,97);(127,79);")
    if sentence_hash=='de8e008583d64bb5a95d30387c457cac':
        sentence = sentence.replace(", since",", since")
        sentence = sentence.replace(", and",", and")
        merged_segments = merged_segments.replace("(56,90);(153,44);","(55,90);(151,44);")
    if sentence_hash=='347b0b5b285d84ac902db48b9a53e328':
        sentence = sentence.replace(" http://homicidesurvivors.com/2006/03/20/the-death-penalty-not-a-human-rights-violation.aspx","")
    
    ################################################################################
    # T4: gun control
    if sentence_hash=='8c5ed312c08ef2b1a7b725e1d49fb8ef':
        sentence = sentence.replace("dont","don't")
    if sentence_hash=='d6ba242a063c197a08102ed213f500be':
        sentence = sentence.replace(" gun control ",' "gun control" ')
        merged_segments = merged_segments.replace("(22,83);","(21,84);")
    if sentence_hash=='209e9632c11b377246e1cf056e935f14':
        sentence = sentence.replace("  "," ")
    if sentence_hash=='7bfb40319ff4e1bba424ab28e64152a6':
        merged_segments = merged_segments.replace("(42,116);","(42,107);")
    if sentence_hash=='67bfbbe90cfef3877c3ff06f46a93f01':
        sentence = sentence.replace("peoples","people's")
    if sentence_hash=='e6b4b38f07fac0c847e141fcdf8ed020':
        sentence = sentence.replace("States,[27]","States, [27]")
        merged_segments = merged_segments.replace("(37,68);(115,75);","(37,68);(116,75);")
    if sentence_hash=='ca51685e334c6712fb4281074c792432':
        sentence = sentence.replace("Myth #8:“","Myth #8 : “")
        merged_segments = merged_segments.replace("(9,39);","(11,39);")
    if sentence_hash=='100395aa0288117ba8461dec7239536c':
    #if sentence.startswith("But leftists want gun control because fewer guns mean fewer murders"):
        sentence = "But leftists want gun control because fewer guns mean fewer murders; Gun control is a denial or limitation by [[government]]s of the right to armed [[self-defense]],"
        merged_segments = merged_segments.replace("(38,29);(135,93);","(38,29);(69,93);")
    if sentence_hash=='e4da9a2bfb2094a0af64e2fecb5c4400':
        sentence = sentence.replace("Myth #4:“","Myth #4 : “")
        merged_segments = merged_segments.replace("(9,57);","(11,57);")
    if sentence_hash=='0c1eeca8493fc5abd0ff3f9512728f79':
        sentence = sentence.replace("Myth #2:“","Myth #2 : “")
        merged_segments = merged_segments.replace("(9,74);","(11,74);")
    if sentence_hash=='e502a847af4c6ebca2d675e978cbd00e':
        sentence = sentence.replace("States,[26]","States, [26]")
        merged_segments = merged_segments.replace("(37,68);(111,79);","(37,68);(112,79);")
    if sentence_hash=='1b25099e21ee144434ba7d83a6e57173':
        sentence = sentence.replace("useand","use and")
    if sentence_hash=='04da621ee6a29ef41ebf78419e21f040':
        sentence = sentence.replace("weapons:[26]","weapons: [26]")
    if sentence_hash=='b3766163f132942b0f6043a17f8dc8d1':
        sentence = sentence.replace("advocates opposition","advocates opposition")
        merged_segments = merged_segments.replace("(148,68);","(147,68);")
    if sentence_hash=='b45cbc97940c05293ab6a558e2e0c453':
        sentence = sentence.replace("New Yorks","New York's")
        sentence = sentence.replace("industrythe","industry the")
    if sentence_hash=='adc0c3e202600f827911d423609c813a':
        sentence = sentence.replace("adultsin","adults in")
        merged_segments = merged_segments.replace("(0,98);(103,58);(162,38);","(0,98);(103,97);")
    if sentence_hash=='7f551f6d3671b249fa4cd3e25a993699':
        sentence = sentence.replace("a cop killer bullet, Vizzard","a 'cop killer' bullet,\" Vizzard")
    if sentence_hash=='b47f3b8c42c801ac1802ce039fd721e3':
        sentence = sentence.replace("opponents hold","opponents hold")
    if sentence_hash=='08a4ca4b0a19a93a1585c958a072de9f':
        sentence = sentence.replace("say gun control in",'say "gun control" in')
    if sentence_hash=='ff276dc16302e1965abcdc70d5939dce':
        sentence = sentence.replace("youll","you'll")
    if sentence_hash=='592098f4e068bc955f04944da47cfee5':
        sentence = sentence.replace("doesnt","doesn't")
    if sentence_hash=='42acde7251183d233b145536e0625a49':
        sentence = sentence.replace("</ref> + +","")
        sentence = sentence.replace('<ref>http://www.reason.com/news/show/28582.html :"'," ")
        merged_segments = merged_segments.replace("(68,159);","(58,100);")
    if sentence_hash=='fcfca2b311fa1eee5f5688919bc41781':
        sentence = sentence.replace('http://www.timesonline.co.uk/',"")
        sentence = sentence.replace('tol/news/politics/article3168607.ece',"")
        sentence = sentence.replace('&lt;/',"")
        sentence = sentence.replace('</ref> + ',"")
        merged_segments = merged_segments.replace("(77,142);","(3,142);")
    if sentence_hash=='2753cc6235787630cade1f56bee038eb':
        sentence = sentence.replace("http://www.timesonline.co.uk/tol/news/politics/article3168607.ece","")
        sentence = sentence.replace("&lt;/","")
        sentence = sentence.replace("</ref> + ","")
        merged_segments = merged_segments.replace("(77,129);","(3,129);")
    ################################################################################
    # T5: marijuana legalization
    if sentence_hash=='babb4d8e63f57f3004c3d16945f24e80':
        merged_segments = merged_segments.replace("(0,130);","(0,132);")
    if sentence_hash=='3698ac93d636e458e5445f6d99139164':
        sentence = sentence.replace("trust Â­ something","trust - something")
    ################################################################################
    # T6: minimum wage
    if sentence_hash=='ec511908827ee6a360b9d5d838088f38':
        merged_segments = merged_segments.replace("(64,88);","(64,86);")
    if sentence_hash=='751bbc373fc5e5c63c3704d826f51225':
        sentence = sentence.replace("Floridas","Florida's")
        merged_segments = merged_segments.replace("(40,84);(125,46);","(40,131);")
    if sentence_hash=='cf277c3c13a7939b7b24defc749cc2e3':
        merged_segments = merged_segments.replace("(24,87);","(25,107);")
    if sentence_hash=='cea5adcf5e9e38a7095132a4ea1792aa':
        sentence = sentence.replace("howeverprimarily","however primarily")
    if sentence_hash=='260aa636f2b6cd219fd92482a389287d':
        merged_segments = merged_segments.replace("(38,47);","(35,50);")
    if sentence_hash=='3a7e2feebace7f17c126759462f1d80f':
        sentence = sentence.replace("couldnt","couldn't")
    if sentence_hash=='a965e8d72419ff348a6ea3f88bd46abb':
        sentence = sentence.replace("workers—9.8","workers — 9.8")
    if sentence_hash=='0c02782b3490b94fa8dc55c2309d9f2b':
        merged_segments = merged_segments.replace("(80,71);","(80,61);")
    if sentence_hash=='fd5b967dd713ba69495d671d47ab0179':
        sentence = sentence.replace("variedNew Hampshire","varied New Hampshire")
        merged_segments = merged_segments.replace("(0,66);(67,148);","(0,215);")
    if sentence_hash=='36c46080cbf916a9c2025480275cfa2d':
        sentence = sentence.replace("sectorsincluding","sectors including")
        sentence = sentence.replace("workersexperienced","workers experienced")
    if sentence_hash=='c32d22ddc1b91b22afc2a4c23bb8b6d3':
        sentence = sentence.replace("revenuenot","revenue not")
    if sentence_hash=='9fa66659c729a1930068198f6d8a29ff':
        sentence = sentence.replace("moraleand","morale and")
        merged_segments = merged_segments.replace("(0,117);(118,22);","(0,140);")
    if sentence_hash=='f901d36be3187af8859d208155461bc4':
        sentence = sentence.replace("benefitfor","benefit for")
    if sentence_hash=='7bf7b1688b9782f348fca7bfad433702':
        sentence = sentence.replace("lawsbut","laws but")
    if sentence_hash=='d5666ce9dda6de44333314bc119a2066':
        sentence = sentence.replace("businessesfor","businesses for")
        sentence = sentence.replace("storesmay","stores may")
    if sentence_hash=='caa2bd036db27e1dd1349e43ca64c403':
        sentence = sentence.replace("minimuman","minimum an")
    if sentence_hash=='0475b14631d77fa01ccb13680d95c9d0':
        sentence = sentence.replace("months","month's")
    if sentence_hash=='076aa468273ad5aff362bfc769b0f91b':
        sentence = sentence.replace("frame16½ yearsthan","frame - 16½ years - than")
    if sentence_hash=='ed6825d1b9203236e342308e952e9d3d':
        sentence = sentence.replace("workers33","workers - 33")
    if sentence_hash=='8ebf563d3513135ff57c10c6c17bb0e7':
        sentence = sentence.replace("individuals","individual's")
        sentence = sentence.replace("ones","one's")
    if sentence_hash=='dbabe98d459803fcd49436cde72fff00':
        sentence = sentence.replace("incentives  incentives","incentives - incentives")
    if sentence_hash=='3cf3b4d12a6f4c6e0f1d4cb0218e9ff0':
        sentence = sentence.replace("hour  before","hour - before")
    ################################################################################
    # T7: nuclear energy
    if sentence_hash=='3e1fd713bb6d4c01732ce78eba2865db':
        sentence = sentence.replace("affordable  and","affordable and")
        merged_segments = merged_segments.replace("(9,58);(69,49);","(9,107);")
    if sentence_hash=='1d437bf5adbcdd0718e4b3766d0c38e7':
        sentence = sentence.replace(" â"," ")
        merged_segments = merged_segments.replace("(46,139);","(43,139);")
    if sentence_hash=='037c013e0835a932a614a710da17b50e':
        sentence = sentence.replace("nuclearâs","nuclear's")
        merged_segments = merged_segments.replace("(124,65);","(124,63);")
    if sentence_hash=='77799d98d61cd1263700a1eb30c2f53a':
        sentence = sentence.replace("nuclear energyâ€™s life-cycle","nuclear energy's life-cycle")
        merged_segments = merged_segments.replace("(34,164);","(34,182);")
    if sentence_hash=='10cb37a20707c81df08a0f4d0bd5fd90':
        sentence = sentence.replace("Its","It's")
    if sentence_hash=='14f3a33a593f0baee643ba324994484f':
        sentence = sentence.replace("isnât","isn't")
        merged_segments = merged_segments.replace("(0,61);(149,70);","(0,61);(149,68);")
    if sentence_hash=='027f452cee1c012aad99aa81ddbada02':
        sentence = sentence.replace("the human","the human")
        merged_segments = merged_segments.replace("(0,104);","(0,103);")
    if sentence_hash=='60feaab9ca37fde43f06f9c89f68abd1':
        sentence = sentence.replace('devastating"Nuclear’s',"""devastating "Nuclear's""")
    if sentence_hash=='2cc143e859220d0a51c926d4727f6290':
        sentence = sentence.replace("-if","- if")
        merged_segments = merged_segments.replace("(0,75);(86,128);","(0,75);(87,128);")
    if sentence_hash=='07e351c6f47d49bceafbf1f6e588c1cf':
        sentence = sentence.replace("bad”—doesn’t","bad\" — doesn't")
        merged_segments = merged_segments.replace("(48,73);(129,89);","(50,73);(131,89);")
    if sentence_hash=='7126aac6548a5d5c920d201a7fcb2e56':
        sentence = sentence.replace("Initiatives","Initiative's")
        sentence = sentence.replace("that  if implemented  would","that if implemented would")
    if sentence_hash=='845eef2bb28503a6c72265ba998e6c65':
        sentence = sentence.replace("nuclear energy â though","nuclear energy though")
    if sentence_hash=='69aee36d5c0c75838a6f1cb2193ed317':
        sentence = sentence.replace("the human factor is",'the "human factor" is')
    if sentence_hash=='77c5bbb33a88c6975384527a9594da66':
        sentence = sentence.replace("Oeko-Institut‬ (","Oeko-Institut (")
    if sentence_hash=='352c1f6a16ff71d6c520f84c2c264fe6':
        sentence = sentence.replace("lets","let's")
    if sentence_hash=='b2d43a9fd838a83f74cfe246b0f99554':
        sentence = sentence.replace("of Energy security and","of 'Energy security' and")
    if sentence_hash=='3996594171812d4896d0e1007ae32197':
        sentence = sentence.replace("nations","nation's")
    if sentence_hash=='b24663d260ffca1e3824d413a0f61dcb':
        sentence = sentence.replace("be â€œon the tableâ€ while",'be "on the table" while')
    ################################################################################
    # T8: school uniforms
    if sentence_hash=='7abedaa2c18cb3d1eaf03624a746fdf0':
        sentence = sentence.replace("adults inner","adults inner")
        merged_segments = merged_segments.replace("(107,90);","(106,90);")
    if sentence_hash=='eba899b6e30e734d1258fbe391230cfd':
        sentence = sentence.replace("arent","aren't")
    if sentence_hash=='a913700fda87cdbebeb70d541549e4f8':
        sentence = sentence.replace("a wholesome look",'a "wholesome" look')
        sentence = sentence.replace("identify anyone",'identify "anyone')
    if sentence_hash=='6c3da39e5ba8ba942ea1eea79f50b02f':
        sentence = sentence.replace("everyoneâs","everyone's")
        merged_segments = merged_segments.replace("(0,110);","(0,108);")
    if sentence_hash=='a53798413cc9b04563357a9993ec9328':
        sentence = sentence.replace("wont","won't")
    if sentence_hash=='84d9477582db41c2c5ceee82ea191859':
        sentence = sentence.replace("couldnt","couldn't")
    if sentence_hash=='a81836552c75096c4fa2ea0e4ed266f7':
        sentence = sentence.replace("students self-perception","students' self-perception")
    if sentence_hash=='bc6571a675e86454811e3152a161b2bd':
        sentence = sentence.replace("systems","system's")
    if sentence_hash=='0d3f8d45ed53d4e39ff7e55ad64f6f71':
        sentence = sentence.replace("studentsâ self-image","students' self-image")
        merged_segments = merged_segments.replace("(0,132);(134,11);","(0,143);")
    if sentence_hash=='f20de2d2aa765dd8689f7b8a928ee1ef':
        sentence = sentence.replace("childrens","children's")
    if sentence_hash=='bb0f464286e4c44bece2be276c69214f':
        sentence = sentence.replace("3)There","3) There")
        merged_segments = merged_segments.replace("(2,141);","(3,141);")
    if sentence_hash=='6ad397bf63b9f71586e6f87a59cbb72a':
        sentence = sentence.replace('"extreme"looks','"extreme" looks')
        merged_segments = merged_segments.replace("(0,75);","(0,76);")
    if sentence_hash=='b33fb6e732873de6cc0ad3fca6fb6adc':
        sentence = sentence.replace(" (www.libertarian-logic.com)","")
    if sentence_hash=='9bfe14924e33c8429bc8270f8cfcc44d':
        sentence = sentence.replace(" (www.libertarian-logic.com)","")
    if sentence_hash=='6caf1527ffe09873f642d213a2e12cc8':
        sentence = sentence.replace("http://www.jennysuemakeup.com ","")
        merged_segments = merged_segments.replace("(135,60);","(105,60);")
    if sentence_hash=='052db445247f03d52351bd461e36443d':
        sentence = sentence.replace("todays","today's")
    if sentence_hash=='17b8d2e3cd23ebf63fc82241721bde90':
        sentence = sentence.replace("Logged A real","Logged A real")
    if sentence_hash=='2ed2d47bcd7aa8dcb4e80cd410024ef6':
        sentence = sentence.replace("™","")
    if sentence_hash=='830ddb7bbc3ccf026ecd84a09dd8b7c8':
        sentence = sentence.replace("a no vote, she","a 'no' vote,\" she")
    if sentence_hash=='3f51c8e7527bb75cf15638e98c150d98':
        sentence = sentence.replace("customer​s​ a","customers a")
    ################################################################################
    
    if 'false' in merged_segments:
        annotated_or_not, arg_spans, arg_stances = merged_segments[2:-2].split("', '")
        arg_spans = [s.strip() for s in arg_spans.split(';') if s.strip()]
        arg_spans = [s[1:-1].split(',') for s in arg_spans]
        arg_spans = [(int(s[0]),int(s[1])) for s in arg_spans]
        arg_stances = [s.strip() for s in arg_stances.split(';') if s.strip()]
        correct_arg_spans = []
        for arg_span, arg_stance in zip(arg_spans, arg_stances):
            arg_span_start, arg_span_length = arg_span
            arg_span_end = arg_span_start + arg_span_length
            correct_arg_spans.append([arg_span_start,arg_span_end])    
        all_annotation_spans_with_stance = get_full_annotation(sentence, correct_arg_spans, arg_stances)
    
    if 'true' in merged_segments:
        all_annotation_spans_with_stance = []
        all_annotation_spans_with_stance.append([0, len(sentence),'non'])
    
    ################################################################################
    
    COLORED_SENTENCE = ''
    CHAR_LEVEL_STANCE = ''
    for SEGMENT_START, SEGMENT_END, SEGMENT_STANCE in all_annotation_spans_with_stance:
        #####
        if num_labels == 2:
            if SEGMENT_STANCE in ['pro','con']:
                SEGMENT_STANCE = 'arg'
        #####
        SEGMENT = sentence[SEGMENT_START:SEGMENT_END]

        if SEGMENT_STANCE=='pro':
            CHAR_LEVEL_STANCE+="".join(['p' for _ in range(len(SEGMENT))])
            SEGMENT = bg.green + fg(255, 255, 255) + SEGMENT + fg.rs + bg.rs

        elif SEGMENT_STANCE=='con':
            CHAR_LEVEL_STANCE+="".join(['c' for _ in range(len(SEGMENT))])
            SEGMENT = bg.red + fg(255, 255, 255) + SEGMENT + fg.rs + bg.rs

        elif SEGMENT_STANCE=='arg': # just argumentative: PRO & CON
            CHAR_LEVEL_STANCE+="".join(['a' for _ in range(len(SEGMENT))])
            SEGMENT = bg.blue + fg(255, 255, 255) + SEGMENT + fg.rs + bg.rs

        else:
            #CHAR_LEVEL_STANCE+="".join(['n' for _ in range(len(SEGMENT))])
            CHAR_LEVEL_STANCE+="".join(['n' for _ in range(len(SEGMENT))])
            SEGMENT = bg(200, 200, 200) + fg(0, 0, 0) + SEGMENT + fg.rs + bg.rs

        COLORED_SENTENCE+=SEGMENT
    assert len(sentence) == SEGMENT_END # last position
    
    ################################################################################
    
    doc = nlp(sentence)
    tokenized_sentence_spacy = [[token.text for token in s] for s in doc.sents]
    tokenized_sentence_spacy = list(chain.from_iterable(tokenized_sentence_spacy)) # flatten nested list if applicable
    tokenized_sentence_bert = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=False))
    
    ################################################################################
    
    CL = "".join([c for c,s in zip(CHAR_LEVEL_STANCE,sentence) if s!=" "])
    SP = "".join(tokenized_sentence_spacy)
    BE = "".join(tokenized_sentence_bert).replace("##","")
    assert len(SP)==len(BE)
    
    ################################################################################
    
    tokenized_sentence_spacy_labels = []
    pos = 0
    for t in tokenized_sentence_spacy:
        l = CL[pos:pos+len(t)]
        L = l
        assert len(list(set(l)))==1 # only one stance
        tokenized_sentence_spacy_labels.append(L)
        pos+=len(t)
    assert len(tokenized_sentence_spacy_labels)==len(tokenized_sentence_spacy)
    
    ################################################################################
    
    tokenized_sentence_bert_labels = []
    pos = 0
    for t in tokenized_sentence_bert:
        t = t.replace("##","")
        l = CL[pos:pos+len(t)]
        L = l
        assert len(list(set(l)))==1 # only one stance
        tokenized_sentence_bert_labels.append(L)
        pos+=len(t)
    assert len(tokenized_sentence_bert_labels)==len(tokenized_sentence_bert)
    
    ################################################################################
    return sentence, COLORED_SENTENCE, tokenized_sentence_spacy, tokenized_sentence_spacy_labels, tokenized_sentence_bert, tokenized_sentence_bert_labels


