from abnumber import Chain
import pickle
import numpy as np
import os
from train_classifier import class_metrics

def put_anarci_obj(dataset,out = "datasets/dataset_ab.pkl"):
    dataset["light_anarci_abnumber"] = [Chain(d["light_anarci"].replace("-",""), scheme="chothia") for d in dataset.iloc()]
    dataset["heavy_anarci_abnumber"] = [Chain(d["heavy_anarci"].replace("-",""), scheme="chothia") for d in dataset.iloc()]
    pickle.dump(dataset, open(out,'wb'))


def similarity(stest,strain, CDR_only = True):#True):
    l_ids_cdr = list(range(24,35))+list(range(50,57))+list(range(89,98))
    h_ids_cdr = list(range(31,36))+list(range(50,66))+list(range(95,103))

    alignment = stest.align(strain)

    n = 0
    nt = 0
    for pos, (aa, bb) in alignment:
        i = str(pos)[1:]
        if not i[-1].isdigit():
            i = i[:-1]
        i = int(i)
        if CDR_only and str(pos)[0]=="L" and i not in l_ids_cdr:
            continue
        if CDR_only and str(pos)[0]=="H" and i not in h_ids_cdr:
            continue
        if aa == bb:
            n += 1
        nt += 1

    return n/nt


def compare_seqs_anarci(dataset, target_seq_l, target_seq_h):
    if isinstance(target_seq_l, str):
        l = Chain(target_seq_l.replace("-",""), scheme="chothia")
        h = Chain(target_seq_h.replace("-",""), scheme="chothia")
    else:
        l = target_seq_l
        h = target_seq_h

    dataset_ = dataset[dataset["good"].isin([0,1])]
    scores = []
    poss = []
    seqs = []
    hits_ = []

    for j,d_train in enumerate(dataset_.iloc()):
        strain = d_train["light_anarci_abnumber"]
        s1 = similarity(l, strain)
        strain = d_train["heavy_anarci_abnumber"]
        s2 = similarity(h, strain)
        scores.append(0.5*(s1+s2))
        hits_.append(d_train)
        poss.append(d_train["POS_class"])

    hits = {i:None for i in range(12)}
    vals = {i:0 for i in range(12)}
    for i,[s,p] in enumerate(zip(scores,poss)):
        if vals[p]<s:
            vals[p] = s
            hits[p] = hits_[i]

    return [vals[ii] for ii in range(12)], [hits[ii] for ii in range(12)]

def compare_seqs(dataset):
    dataset_ = dataset[dataset["good"].isin([0,1])]
    cv_id = 0
    q1 = [True if d else False for d in dataset_[f"cross_val_test_{cv_id}"].iloc()]

    dataset_test = dataset_[q1]#.iloc():#[:5]#.iloc()#[:5]
    dataset_train = dataset_[[not q_ for q_ in q1]]#.iloc()#[:5]

    pred = {}
    pred_s = {}
    for i,d_test in enumerate(dataset_test.iloc()):
        scores = []
        poss = []
        for j,d_train in enumerate(dataset_train.iloc()):
            #scores.append(0)
            #continue
            stest = d_test["light_anarci_abnumber"]
            strain = d_train["light_anarci_abnumber"]
            s1 = similarity(stest, strain)
            stest = d_test["heavy_anarci_abnumber"]
            strain = d_train["heavy_anarci_abnumber"]
            s2 = similarity(stest, strain)
            scores.append(s1+s2)
            poss.append(d_train["POS_class"])
        vals = {i:0 for i in range(12)}
        for s,p in zip(scores,poss):
            if vals[p]<s:
                vals[p] = s
        #print(ii, np.max(scores))

        pred[d_test["index"]] = dataset_train.iloc()[np.argmax(scores)]["POS_class"]
        pred_s[d_test["index"]] = np.array([vals[ii] for ii in range(12)])
        print(pred_s[d_test["index"]])
        #print(np.array([vals[ii] for ii in range(12)]))
        #exit(0)

    dataset["levenstein_predictions"] = None
    dataset["levenstein_predictions"] = dataset.apply(lambda row: pred[row['index']]
                                                 if row['index'] in pred else row["levenstein_predictions"], axis=1)

    dataset["levenstein_predictions_score"] = None
    dataset["levenstein_predictions_score"] = dataset.apply(lambda row: pred_s[row['index']]
                                                 if row['index'] in pred_s else row["levenstein_predictions_score"], axis=1)

    r = ~dataset["levenstein_predictions"].isnull()
    x = dataset["levenstein_predictions"][r]
    y = dataset["POS_class"][r]
    m = class_metrics(x,y)
    #print(m)

    pickle.dump(dataset, open("./datasets/dataset_ab_levenstein_cv_0.pkl",'wb'))

    #print(dataset["levenstein_predictions"])

def main():
    if not os.path.exists("./datasets/dataset_ab.pkl"):
        dataset = pickle.load(open("./datasets/dataset.pkl",'rb'))
        put_anarci_obj(dataset, "./datasets/dataset_ab.pkl")

    dataset = pickle.load(open("./datasets/dataset_ab.pkl",'rb'))
    #put_anarci_obj(dataset, "./datasets/dataset_ab.pkl")
    #print(list(dataset))
    compare_seqs(dataset)
    #compare_seqs
    pass

if __name__ == "__main__":
    main()