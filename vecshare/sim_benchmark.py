import numpy as np
import pandas as pd
import csv, os

def _eval_all(emb_simset):
    inp_emb = {}
    for wordvec in emb_simset.iterrows():
        word, vec = wordvec[1][0], wordvec[1][1:].tolist()
        vec = np.fromiter(map(float, vec[1:]), dtype = np.float32)
        norm = np.linalg.norm(vec)
        inp_emb[word] = vec/norm if (norm != 0) else [vec]
    files = ['Test_Input/'+ f_name for f_name in next(os.walk('Test_Input'))[2]]
    sim_score = 0
    for testfile in files:
        sim_score += _eval_sim(testfile, inp_emb)/len(files)
    return sim_score

def _eval_sim(testfile,inp_emb):
    sim_x, sim_y = np.empty(0), np.empty(0)
    with open(testfile, 'rU') as comp_test:
        tests_csv = csv.reader(comp_test)
        for line in tests_csv:
            word1, word2 = line[0], line[1]
            if (word1 in inp_emb) and (word2 in inp_emb):
                wordvec_1, wordvec_2 = inp_emb[word1], inp_emb[word2]
                sim_x = np.append(sim_x, float(line[2]))
                if np.any(wordvec_1) and np.any(wordvec_2):
                    sim_y = np.append(sim_y, np.dot(wordvec_1, wordvec_2))
                else:
                    sim_y = np.append(sim_y, 0)
    spearman_corr = np.corrcoef(sim_x, sim_y)[0, 1]
    return spearman_corr
