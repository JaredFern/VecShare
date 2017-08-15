import numpy as np
import pandas as pd
import csv, os, random, sys

def _eval_all(emb_simset):
    inp_emb = {}
    for wordvec in emb_simset.iterrows():
        word, vec = wordvec[1][0], wordvec[1][1:].tolist()
        vec = np.fromiter(map(float, vec[1:]), dtype = np.float32)
        norm = np.linalg.norm(vec)
        inp_emb[word] = vec/norm if (norm != 0) else [vec]

    score_dict = {}
    score_dict['score'] = 0
    for root,dirs,files in os.walk('Test_Input'):
        files = [testfile for testfile in files if testfile[0]!='.']
        for testfile in files:
            f_path = 'Test_Input/'+testfile
            score_dict[testfile[:-4].strip().lower().replace(" ", "_").replace("-", "_")] = _eval_sim(f_path, inp_emb)
            if  testfile != 'mc-30.csv':
                score_dict['score'] += _eval_sim(f_path, inp_emb)/(len(files)-1)
    return score_dict

def _eval_sim(testfile,inp_emb):
    test, emb = np.empty(0), np.empty(0)
    testdrop = np.empty(0)
    spearman_corr = 0

    with open(testfile, 'rU') as comp_test:
        tests_csv = csv.reader(comp_test)
        for line in tests_csv:
            word1, word2 = line[0], line[1]
            if (word1 in inp_emb) and (word2 in inp_emb):
                wordvec_1, wordvec_2 = inp_emb[word1], inp_emb[word2]
                test = np.append(test, float(line[2]))
                if np.any(wordvec_1) and np.any(wordvec_2):
                    emb = np.append(emb, np.dot(wordvec_1, wordvec_2))
                else:
                    emb = np.append(emb, 0)
            else:
                testdrop = np.append(testdrop, float(line[2]))

    for i in range (0,5):
        embdrop = np.empty(0)
        for j in range (0, len(testdrop)):
            temp_test, temp_emb = np.empty(0), np.empty(0)
            randvec1 = random.choice(inp_emb.values())
            randvec2 = random.choice(inp_emb.values())
            if np.any(randvec1) and np.any(randvec2):
                embdrop = np.append(embdrop, np.dot(randvec1, randvec2))
            else:
                embdrop = np.append(embdrop, 0)

        temp_test = np.append(test, testdrop)
        temp_emb  = np.append(emb, embdrop)

        spearman_corr += (np.corrcoef(temp_test, temp_emb)[0, 1])/5
    return spearman_corr
