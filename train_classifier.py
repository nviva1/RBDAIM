import pickle
from transformers import AdamW
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rbdaim.epitope_model import AntiBERTyFAB_CDR_Pair, AntiBERTyFAB_CLS_Pair, EpitopesPairDatasetCDR
import os
import pickle

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from rbdaim.epitope_model import AntiBERTyFAB_CDR_Pair, AntiBERTyFAB_CLS_Pair, EpitopesPairDatasetCDR


#VOCAB_FILE = os.path.join(trained_models_dir, 'vocab.txt')
#from test import calculate_metrics_by_model

def train_and_evaluate_model(model,
                             train_dataloader,
                             test_dataloader):
    config = get_model_config()
    device = config["device"]
    model.to(device)
    num_epochs = config["epochs"]
    model.train()
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(num_epochs):
        losses = []
        i = 0
        for batch in tqdm(train_dataloader, desc='Epoch ' + str(epoch+1), position=0, leave=True):
            batch = {k: v.to(device) for k, v in batch.items()}  # Send input data to the device (GPU or CPU)
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

    # Evaluate the rbdaim on a separate validation dataset
    model.eval()  # Set the rbdaim in evaluation mode
    test_predictions = {"predictions":[],
                        "labels":[],
                        "antibody_ids":[],
                        "embeddings":[]}

    with torch.no_grad():
        # Validation loop
        i = 0
        for batch in test_dataloader:
            i+=1
            batch = {k: v.to(device) for k, v in batch.items()}  # Send input data to the device (GPU or CPU)
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

def train_models(dataset,
                   config = None,
                   device = "mps",
                   model_type = "CLS",
                   cv_id = 0,
                   ensemble_id = 0,
                   full_model=False,
                   rewrite=False,
                   experiment_name = "AntiBERTyFAB_Pair"):

    model_name = f'weights/NN_cv_{cv_id}_{model_type}_ens_{ensemble_id}.pth'

    if not rewrite and os.path.exists(model_name):
        return

    if config is None:
        config = get_model_config()

    cv_predictions = None
    dataset["our_test"] = [True if d else False for d in dataset["our_test"].iloc()]
    pred_per_cv = {}

    model = None

    q1 = [True if d else False for d in dataset[f"cross_val_test_{cv_id}"].iloc()]
    q2 = [not q for q in q1]

    train_df = dataset[q2]
    test_df  = dataset[q1]

    if full_model:
        train_df = dataset[~dataset["our_test"]]

    ds_train = EpitopesPairDatasetCDR(train_df, 12)#config["num_classes"])
    ds_test = EpitopesPairDatasetCDR(test_df, 12)#config["num_classes"])

    train_dataloader = DataLoader(ds_train,
                                      batch_size=1,
                                      shuffle=True)

    test_dataloader = DataLoader(ds_test,
                                      batch_size=1,
                                      shuffle=False)

    if model_type == "CLS":
        model = AntiBERTyFAB_CLS_Pair(config)
    else:
        model = AntiBERTyFAB_CDR_Pair(config)

    pred = train_and_evaluate_model(model,
                                        train_dataloader,
                                        test_dataloader)

    pred_per_cv[cv_id] = pred
    if cv_predictions is None:
        cv_predictions = pred
    else:
        for k,v in pred.items():
            cv_predictions[k].extend(v)

    cv_predictions = {k:np.concatenate(v,axis=0) for k,v in cv_predictions.items()}
    cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["predictions"])}
    dataset[experiment_name+"_predictions"] = None
    dataset[experiment_name+"_predictions"] = dataset.apply(lambda row: cv[row['index']]
                                                 if row['index'] in cv else row[experiment_name+"_predictions"],
                                                            axis=1)

    for cv_id in pred_per_cv:
        cv_predictions = pred_per_cv[cv_id]
        cv_predictions = {k:np.concatenate(v,axis=0) for k,v in cv_predictions.items()}
        cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["predictions"])}
        dataset[experiment_name+f"_predictions_cv_{cv_id}"] = None
        dataset[experiment_name+f"_predictions_cv_{cv_id}"] = dataset.apply(lambda row: cv[row['index']]
                                                     if row['index'] in cv else row[experiment_name+"_predictions"],
                                                                axis=1)

    model.eval()
    model.to("cpu")

    if full_model:
        model_name = f'weights/NN_cv_{cv_id}_{model_type}_full.pth'
    else:
        model_name = f'weights/NN_cv_{cv_id}_{model_type}_ens_{ensemble_id}.pth'

    torch.save(model.state_dict(), model_name)
    return dataset


def class_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    metrics = {"accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1": f1,
               "MCC": mcc}
    return metrics

def calculate_metrics_by_model(df, model_name):
    sel = ~df[model_name].isnull()
    y_test = df["POS_class"][sel].to_numpy()
    y_pred = np.array(list(df[model_name][sel]))
    y_test_ = np.eye(y_pred.shape[1])[y_test]
    metrics = {}
    metrics["model_name"] = model_name.rstrip("_predictions")
    metrics.update(class_metrics(y_test,
                                 np.argmax(y_pred,axis=1)))
    return metrics

def get_model_config():
    config = {"num_classes": 12,
              "model_type": "CDR",
              "epochs": 3,
              "lr":2e-5,
              "device":"mps"
              }
    return config

def main():
    ###
    dataset = pickle.load(open("datasets/dataset.pkl",'rb')).reset_index(drop=True)
    for i in range(10):
        for ens_id in range(5):
            train_models(dataset,
                           cv_id=i,
                           model_type="CLS",
                           ensemble_id=ens_id,
                           full_model=False)
            train_models(dataset,
                           cv_id=i,
                           model_type="CDR",
                           ensemble_id=ens_id,
                           full_model=False)

    pass

if __name__ == "__main__":
    main()