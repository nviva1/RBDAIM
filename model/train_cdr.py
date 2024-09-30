import json
import pandas as pd
import random
from sklearn.model_selection import KFold
import antiberty
from antiberty import AntiBERTyRunner
import numpy as np
from pathlib import Path
import subprocess
import sys
import argparse
import os
import ast
from antiberty import AntiBERTyRunner
import numpy as np
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from ast import literal_eval
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import torch
import transformers
from tqdm import tqdm
import antiberty
from antiberty import AntiBERTy
from antiberty.utils.general import exists
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, ModelOutput
from antiberty.utils.general import exists
#from epitope_model_CDR_esm import EpitopesPairDatasetCDR, AntiBERTyFAB_CDR_Pair
from epitope_model import EpitopesPairDatasetCDR, AntiBERTyFAB_CDR_Pair, AntiBERTyFAB_CLS_Pair
#VOCAB_FILE = os.path.join(trained_models_dir, 'vocab.txt')
from test import calculate_metrics_by_model 

def train_and_evaluate_model(model,
                             train_dataloader,
                             test_dataloader,
                             config,
                             device,
                             weights=None):

    model.to(device)
    num_epochs = config["epochs"]#3
    model.train()
                    
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(num_epochs):
        losses = []
        i = 0
        for batch in tqdm(train_dataloader, desc='Epoch ' + str(epoch+1), position=0, leave=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("antibody_id")
            outputs = model(**batch)
            losses.append(outputs["loss"].item())
            i+=1
            val = np.average(losses[:-25])       
            if i%100==0:
                print(val)
            loss = outputs["loss"]
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=0.1
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Evaluate the model on a separate validation dataset
    model.eval()  # Set the model in evaluation mode    
    test_predictions = {"predictions":[],
                        "labels":[],
                        "antibody_ids":[],
                        "embeddings":[]}
    
    with torch.no_grad():
        i = 0
        for batch in test_dataloader:
            i+=1
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels").detach().cpu().numpy()
            batch["labels"] = None
            fab_ids = batch.pop("antibody_id").detach().cpu().numpy()
            outputs = model(**batch)
            pred = outputs["logits"].detach().cpu().numpy()
            test_predictions["predictions"].append(pred)
            test_predictions["labels"].append(labels)
            test_predictions["antibody_ids"].append(fab_ids)
            test_predictions["embeddings"].append(outputs["embeddings"].detach().cpu().numpy())
    return test_predictions

def run_experiment(dataset,
                   config = None, #device = "mps",
                   device = "mps",
                   experiment_name = "AntiBERTyFAB_Pair"):

    cv_predictions = None
    dataset["our_test"] = [True if d else False for d in dataset["our_test"].iloc()]
    print(dataset["our_test"])
    pred_per_cv = {}
    for cv_id in range(10):
        q1 = [True if d else False for d in dataset[f"cross_val_test_{cv_id}"].iloc()]
        q2 = [not q for q in q1]
        train_df = dataset[q2]
        test_df  = dataset[q1]
        if experiment_name.endswith("full"):
            train_df = dataset[~dataset["our_test"]]
            test_df = dataset[dataset["our_test"]]

        ds_train = EpitopesPairDatasetCDR(train_df, config["num_classes"])
        ds_test = EpitopesPairDatasetCDR(test_df, config["num_classes"])
        batch_size=True
        
        train_dataloader = DataLoader(ds_train, 
                                      batch_size=config["batch_size"],
                                      shuffle=True)

        test_dataloader = DataLoader(ds_test, 
                                      batch_size=config["batch_size"],
                                      shuffle=False)
        print(ds_train.class_weights)

        if config["model_type"]=="CLS":
            model = AntiBERTyFAB_CLS_Pair(config, weights=ds_train.class_weights, loss_weights=ds_train.loss_weights)
        else:
            model = AntiBERTyFAB_CDR_Pair(config, weights=ds_train.class_weights, loss_weights=ds_train.loss_weights)
            

        pred = train_and_evaluate_model(model,
                                        train_dataloader, 
                                        test_dataloader,
                                        config=config,
                                        device=device)
        pred_per_cv[cv_id] = pred
        

        if experiment_name.endswith("full"):
            cv_predictions = pred#pred_per_cv[cv_id]
            cv_predictions = {k:np.concatenate(v,axis=0) for k,v in cv_predictions.items()}
            cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["predictions"])}
            dataset[experiment_name+f"_predictions"] = None
            dataset[experiment_name+f"_predictions"] = dataset.apply(lambda row: cv[row['index']]
                                                         if row['index'] in cv else row[experiment_name+"_predictions"],
                                                                    axis=1)

            cv_predictions = pred#pred_per_cv[cv_id]
            cv_predictions = {k:np.concatenate(v,axis=0) for k,v in cv_predictions.items()}
            
            cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["predictions"])}
            ev = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["embeddings"])}

            #dataset[experiment_name+f"_predictions"] = None
            dataset[experiment_name+f"_embeddings"] = None
            dataset[experiment_name+f"_embeddings"] = dataset.apply(lambda row: cv[row['index']]
                                                         if row['index'] in cv else row[experiment_name+"_embeddings"],
                                                                    axis=1)
 

            return dataset


        if cv_predictions is None:
            cv_predictions = pred
        else:
            for k,v in pred.items():
                cv_predictions[k].extend(v)
        del model

    cv_predictions = {k:np.concatenate(v,axis=0) for k,v in cv_predictions.items()}
    cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["predictions"])}
    dataset[experiment_name+"_predictions"] = None
    dataset[experiment_name+"_predictions"] = dataset.apply(lambda row: cv[row['index']] 
                                                 if row['index'] in cv else row[experiment_name+"_predictions"], 
                                                            axis=1)

    cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["embeddings"])}
    dataset[experiment_name+"_embeddings"] = None
    dataset[experiment_name+"_embeddings"] = dataset.apply(lambda row: cv[row['index']] 
                                                 if row['index'] in cv else row[experiment_name+"_embeddings"], 
                                                            axis=1)

    for cv_id in pred_per_cv:
        cv_predictions = pred_per_cv[cv_id]
        cv_predictions = {k:np.concatenate(v,axis=0) for k,v in cv_predictions.items()}
        cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["predictions"])}
        dataset[experiment_name+f"_predictions_cv_{cv_id}"] = None
        dataset[experiment_name+f"_predictions_cv_{cv_id}"] = dataset.apply(lambda row: cv[row['index']]
                                                     if row['index'] in cv else row[experiment_name+"_predictions"],
                                                                axis=1)

    return dataset
    
def train_final_model(train_df,
                      test_df,
                      experiment_name = "AntiBERTyFAB_Pair",
                      output = "0"):

    train_df = train_df
    ds_train = EpitopesPairDatasetCDR(train_df)
    ds_test = EpitopesPairDatasetCDR(test_df)
    batch_size=True
    train_dataloader = DataLoader(ds_train,
                                      batch_size=1, 
                                      shuffle=True)
    test_dataloader = DataLoader(ds_test, 
                                batch_size=1, 
                                shuffle=False)
    model = get_model(experiment_name)
    pred = train_and_evaluate_model(model,
                                    train_dataloader,
                                    test_dataloader)    
    torch.save(model.state_dict(), f"./models/antibery_pair_full_{output}.pth")
    Path("./models/").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "./models/antibery_pair_full.pth")
    del model
    return pred


def prepare_cv_tasks():
    for i in range(10):
        fo = open(f"run_cv_experiments_{i}.sh",'w')
        if i%2 == 0:
            device = "1"
        else:
            device = "0"
        jj = [str(i*10+j) for j in range(10)]
        jj = ",".join(jj)
        fo.write(f"export CUDA_VISIBLE_DEVICES={device}; python train.py {jj} 0\n")

    fo.close()

def cv_experiments(config, device, job_ids):
    Path(config["output_path"]).mkdir(exist_ok=True)
    dataset = pickle.load(open(config["dataset_path"],'rb'))
    for job_id in job_ids:
        if int(job_id)<0:
            job_name = "AntiBERTyFAB_Pair_"+str(job_id)+"_full"
        else:
            job_name = "AntiBERTyFAB_Pair_"+str(job_id)
            
        dataset = pickle.load(open(config["dataset_path"],'rb'))

        if config["only_good"]:
            dataset = dataset[dataset["good"]!=-1]

        output_name = os.path.join(config["output_path"], "dataset_model_"+job_name+".pkl" )
        if os.path.exists(output_name):
            print(output_name, "exists")
            continue
        
        dataset = run_experiment(dataset,
                             experiment_name=job_name,
                             config=config,
                             device = device)

        pickle.dump(dataset, open(output_name,'wb'))#, dataset)


        
def train_full_model(config=None,
                     model_id = 0):
    output_name = os.path.join(config["output_full_model_path"], f"antibery_pair_full_model_id_{model_id}.pth")
    dataset_path = config["dataset_path"]
    
    if os.path.exists(output_name):
        print("Model exists", output_name)
        return

    device = config["device"]
    dataset = pickle.load(open(dataset_path,'rb'))
    train_df = dataset[~dataset["our_test"]]#.iloc()#[:10]
    test_df = dataset[dataset["our_test"]]#.iloc()#[:10]
    
    ds_train = EpitopesPairDatasetCDR(train_df, n_epitopes = config["num_classes"])
    ds_test = EpitopesPairDatasetCDR(test_df, n_epitopes = config["num_classes"])
    batch_size=True
    
    train_dataloader = DataLoader(ds_train,
                                  batch_size=config["batch_size"], 
                                  shuffle=True)

    test_dataloader = DataLoader(ds_test, 
                                 batch_size=config["batch_size"], 
                                 shuffle=False)
    
    model = AntiBERTyFAB_CDR_Pair(config)#num_epitopes = config["num_classes"])

    pred = train_and_evaluate_model(model,
                                    train_dataloader,
                                    test_dataloader,
                                    config=config,
                                    device=device)

    Path(config["output_full_model_path"]).mkdir(exist_ok=True)
    torch.save(model.state_dict(),
               output_name)
    experiment_name = "AntiBERTyFAB_Pair"
    cv_predictions = {k:np.concatenate(v,axis=0) for k,v in pred.items()}
    cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["predictions"])}
    dataset[experiment_name+"_predictions"] = None
    dataset[experiment_name+"_predictions"] = dataset.apply(lambda row: cv[row['index']]
                                                 if row['index'] in cv else row[experiment_name+"_predictions"],
                                                            axis=1)

    cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["embeddings"])}
    dataset[experiment_name+"_embeddings"] = None
    dataset[experiment_name+"_embeddings"] = dataset.apply(lambda row: cv[row['index']]
                                                 if row['index'] in cv else row[experiment_name+"_embeddings"],
                                                            axis=1)

    output_wth = os.path.join(config["output_full_model_path"])
    Path(output_wth).mkdir(exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_wth, f"classifier_{model_id}.pth"))

    #pickle.dump(dataset, open(output_name[:-4]+".pkl",'wb'))
    
    del model
    
def train_random_split_model(device="cuda:0",
                      config=None):
    dataset_path = config["dataset_path"]
    dataset = pickle.load(open(dataset_path,'rb'))
    #print(dataset.iloc()[0])

    #exit(0)
    dataset = dataset[~dataset["our_test"]]
    dataset_our = dataset[dataset["our_test"]]
    
    #print(dataset)
    #exit(0)
    
    ids = list(range(len(dataset)))
    random.Random(4).shuffle(ids)
    n = int(len(ids)/10)
    ids_test = ids[-n:]
    ids_train = ids[:-n]
    train_df = dataset.iloc()[ids_train]
    test_df = dataset.iloc()[ids_test]
    ds_train = EpitopesPairDatasetCDR(train_df,config["num_classes"])
    class_weight = ds_train.class_weights
    ds_test = EpitopesPairDatasetCDR(test_df, config["num_classes"])

    batch_size=True
    train_dataloader = DataLoader(ds_train,
                                  batch_size=config["batch_size"], 
                                  shuffle=True)
    test_dataloader = DataLoader(ds_test, 
                                 batch_size=config["batch_size"], 
                                 shuffle=False) 
    model = AntiBERTyFAB_CDR_Pair(config, weights=class_weight)#num_epitopes = config["num_classes"])
    pred = train_and_evaluate_model(model,
                                    train_dataloader,
                                    test_dataloader,
                                    config=config,
                                    device=device)

    pred = np.concatenate(pred["predictions"],axis=0)    
    test_df["model_predictions"]  = list(pred)
    del model
    return calculate_metrics_by_model(test_df, "model_predictions") 
        
def grid_search(config, device):
    all_metrics = []
    lr0 = 1e-5

    fo = open("predictions/grid_search.log",'w')
    for batch_size in [1]:#5,4,3,2,1]:#1,2,3,4,5]:
        for epoch in [2,3,4]:
            for lr in [lr0/2.0, lr0,lr0*2,lr0*3,lr0*4,lr0*5,lr0*10]:#*2]:#, lr0*4]:#lr0/2, lr0, lr0*2, lr0*4, lr0*8, lr0*10]:
                config["batch_size"] = batch_size
                config["learning_rate"]= lr
                config["epochs"]=epoch
                df = train_random_split_model(config=config, device=device)
                df["batch_size"] = batch_size
                df["learning_rate"] = lr
                df["epochs"] = epoch
                all_metrics.append(df)
                fo.write(f"{df}\n")
                fo.flush()
                print(df)
    fo.close()

    all_metrics = pd.DataFrame(all_metrics)#concat(all_metrics)
    all_metrics.to_csv("predictions/grid_search.csv")
    print(all_metrics)

def grid_me():
    config = {"num_classes": 12,
    #"dataset_path": "./experiments_wd/dataset.pkl",
    "dataset_path": "./experiments_wd/dataset.pkl"}
    grid_search(config, "cuda:0")

def main():

    #config = json.load(open("./experiments_cdr/exp3/config.json"))
    #grid_search(config, "cuda:0")
    #exit(0)

    parser = argparse.ArgumentParser(description='train_NN_model')
    parser.add_argument('--cv_ensemble_ids', default=None, type=lambda s: [int(item) for item in s.split(',')], help='A unique id for CV evaluation of the model')
    parser.add_argument('--device', type=str, help='device', default="cuda:0")
    parser.add_argument('--model_type', type=str, help='model', default="CLS")
    parser.add_argument('--only_good', type=bool, help="remove not good ds", default=False)
    parser.add_argument('--loss_weight', type=bool, help="extra weight at loss", default=False)
    parser.add_argument('--freeze', type=bool, help="freeze parameters", default=False)
    parser.add_argument('--epochs', type=int, help="number parameters", default=-1)
    parser.add_argument('--model_config', type=str, help='model config', default="./configs/model.json")
    parser.add_argument('--train_cv', type=bool, help='train CV model', default=True)
    parser.add_argument('--train_full', type=bool, help='train full model', default=False)
    parser.add_argument('--hyp_search', type=bool, help='hyperparameters search mode', default=False)   
    parser.add_argument('--output_predictions_path', type=str, help='folder to store model predictions', default="./output/")
    parser.add_argument('--output_models_path', type=str, help='folder to store model weights', default="./weights/")
    parser.add_argument('--loss', type=str, help='BCE or CE', default="BCE")

    args = parser.parse_args()

    cv_ids = args.cv_ensemble_ids
    device = args.device

    #print(config)
    #exit(0)
    
    train_cv = args.train_cv
    train_full = args.train_full
    hsearch = args.hyp_search

    config = {"num_classes": 12,
                   "dataset_path": "../datasets/dataset.pkl",
                   "learning_rate": 1e-05,
                   "batch_size": 1,
                   "model_type": "CDR",
                   "epochs": 4}
    config["device"] = "mps"

    if "output_path" not in config:
        config["output_path"] = args.output_predictions_path
    if "output_full_model_path" not in config:
        config["output_full_model_path"] = args.output_models_path
    if "only_good" not in config:
        config["only_good"] = args.only_good
    if "epochs" not in config:
        if args.epochs!=-1:
            config["epochs"] = args.epochs
    if "freeze" not in config:
        config["freeze"] = args.freeze
    if "loss" not in config:
        config["loss"] = args.loss

    for i in range(4):
        train_full_model(config=config,                         
                         model_id=i)
                     
    #grid_me()
    exit(0)
    
        
    if hsearch:
        grid_search(config, device)
        exit(0)

    #if train_full:
    #    exit(0)
    if "model_type" not in config: 
        config["model_type"] = args.model_type


    #just_put_embeddings()
    #exit(0)
    if train_cv:
        cv_experiments(config, device, cv_ids)#[i for i in cv_ids if i !=-1])
        #for i in [i_ for i_ in cv_ids if i_<0]:# in range(10):
        #    train_full_model(config=config,
        #                     device=device,
        #                     model_id=i)

    exit(0)
    """
    train for inference on our test set
    """
    dataset = pickle.load(open("data/dataset_2.pkl",'rb'))

def just_put_embeddings(path = "dataset.pkl",
                   path_out = None,
                   config = None, #device = "mps",
                   device = "mps"):

    data = pickle.load(open(path,'rb'))

    ds = EpitopesPairDatasetCDR(data, config["num_classes"])
    dataloader = DataLoader(ds, 
                                      batch_size=1, 
                                      shuffle=False)
    if config["model_type"]=="CLS":
        model = AntiBERTyFAB_CLS_Pair(config, weights=ds.class_weights, loss_weights=ds.loss_weights)#num_epitopes = config["num_classes"])
    else:
        model = AntiBERTyFAB_CDR_Pair(config, weights=ds.class_weights, loss_weights=ds.loss_weights)#num_epitopes = config["num_classes"])
            
    model.eval()  # Set the model in evaluation mode    
    test_predictions = {"predictions":[],
                        "labels":[],
                        "antibody_ids":[],
                        "embeddings":[]}
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}  # Send input data to the device (GPU or CPU)
            labels = batch.pop("labels").detach().cpu().numpy()
            batch["labels"] = None
            fab_ids = batch.pop("antibody_id").detach().cpu().numpy()
            outputs = model(**batch)
            pred = outputs["logits"].detach().cpu().numpy()
            test_predictions["labels"].append(labels)
            test_predictions["antibody_ids"].append(fab_ids)
            test_predictions["embeddings"].append(outputs["embeddings"].detach().cpu().numpy())

    data["embeddings"]= test_predictions["embeddings"]

    print(data)
    exit(0)
if __name__ == "__main__":
    main()
    exit(0)
    data = pickle.load(open("./data/our_ab_seq.pkl",'rb'))
    #print(data)
    
    for d_ in data.iloc():
        if d_["light_anarci"].startswith("-"):
            print(d_["Name"], d_["light_anarci"])#["light_anarci"])
        #print(d_)
        
