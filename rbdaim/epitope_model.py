import numpy as np
from torch.utils.data import Dataset
import os
import transformers
import antiberty
from antiberty import AntiBERTy
import torch
from torch import nn
import os

import antiberty
import numpy as np
import torch
import transformers
from antiberty import AntiBERTy
from torch import nn
from torch.utils.data import Dataset

project_path = os.path.dirname(os.path.realpath(antiberty.__file__))
trained_models_dir = os.path.join(project_path, 'trained_models')
CHECKPOINT_PATH = os.path.join(trained_models_dir, 'AntiBERTy_md_smooth')

class EpitopesPairDatasetCDR(Dataset):
    """
    Dataset of labeled Fab pairs
    and corresponding epitope binding class
    """
    def __init__(self, df, n_epitopes):
        """
        :param df: input PandasDataframe
        :param n_epitopes: number of epitope classes
        """
        self.df    = df
        self.n_epitopes = n_epitopes

        VOCAB_FILE = os.path.join(trained_models_dir, 'vocab.txt')

        self.tokenizer = transformers.BertTokenizer(vocab_file=VOCAB_FILE,
                                                    do_lower_case=False)

        self.indices = self.df["index"].to_list()

        self.cdr_ids_l = self.df["light_ids_cdr"].to_list()
        self.cdr_ids_h = self.df["heavy_ids_cdr"].to_list()

        for i in range(len(self.cdr_ids_l)):
            self.cdr_ids_l[i] = np.array(self.cdr_ids_l[i])
            self.cdr_ids_l[i]-=1

        for i in range(len(self.cdr_ids_h)):
            self.cdr_ids_h[i] = np.array(self.cdr_ids_h[i])
            self.cdr_ids_h[i]-=1

        self.sequences_l = self.df["light_anarci"].to_list()
        self.sequences_h = self.df["heavy_anarci"].to_list()

        self.class_labels = np.eye(self.n_epitopes)[self.df['POS_class'].to_numpy()]
        self.max_length = 150

    def __len__(self):
        return len(self.sequences_l)
    
    def __getitem__(self, idx):
        sequence_l = " ".join(list(self.sequences_l[idx]))
        sequence_h = " ".join(list(self.sequences_h[idx]))

        cdr_ids_l = self.cdr_ids_l[idx]
        cdr_ids_h = self.cdr_ids_h[idx]

        class_label = self.class_labels[idx]
        
        encoding_l = self.tokenizer.encode_plus(
            sequence_l,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        encoding_h = self.tokenizer.encode_plus(
            sequence_h,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )        
        # Return the tokenized sequence and corresponding class label
        return {
            'antibody_id': self.indices[idx], 
            'input_ids_light': encoding_l['input_ids'].flatten(),
            'attention_mask_light': encoding_l['attention_mask'].flatten(),
            'input_ids_heavy': encoding_h['input_ids'].flatten(),
            'attention_mask_heavy': encoding_h['attention_mask'].flatten(),
            'cdr_ids_light': torch.tensor(cdr_ids_l, dtype=torch.long),
            'cdr_ids_heavy': torch.tensor(cdr_ids_h, dtype=torch.long),
            'labels': torch.tensor(class_label, dtype=torch.long)
        }

class AntiBERTyFAB_CLS_Pair(nn.Module):
    """
    Model predicting the epitope class based on concatenated CLS embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_epitopes = config["num_classes"]
        self.model_l = AntiBERTy.from_pretrained(CHECKPOINT_PATH)
        self.model_h = AntiBERTy.from_pretrained(CHECKPOINT_PATH)
        self.epitopes = nn.Linear(self.model_l.config.hidden_size*2, self.num_epitopes)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self,
                input_ids_light, 
                attention_mask_light, 
                cdr_ids_light,
                input_ids_heavy, 
                attention_mask_heavy, 
                cdr_ids_heavy,
                labels=None,
                return_embeddings = False):
        
        outputs_l = self.model_l.bert(
                input_ids=input_ids_light,
                attention_mask=attention_mask_light,
                output_hidden_states=True,
                output_attentions=True)

        outputs_h = self.model_h.bert(
                input_ids=input_ids_heavy,
                attention_mask=attention_mask_heavy,
                output_hidden_states=True,
                output_attentions=True)

        _,sequence_output_l = outputs_l[:2]
        _,sequence_output_h = outputs_h[:2]

        embeddings = torch.concat([sequence_output_l, sequence_output_h], axis=-1)
        pred = self.epitopes(embeddings)
        if labels is not None:
            loss = self.loss(pred.view(-1), labels.view(-1).float())
        else:
            loss = None

        return {"logits":pred,
                "loss": loss, 
                "labels":labels,
                "embeddings":embeddings}

class AntiBERTyFAB_CDR_Pair(nn.Module):
    """
    Model predicting the epitope class based on concatenated average CDS embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_epitopes = config["num_classes"]
        self.model_l = AntiBERTy.from_pretrained(CHECKPOINT_PATH)
        self.model_h = AntiBERTy.from_pretrained(CHECKPOINT_PATH)
        self.epitopes = nn.Linear(self.model_l.config.hidden_size*2, self.num_epitopes)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, 
                input_ids_light, 
                attention_mask_light, 
                cdr_ids_light,
                input_ids_heavy, 
                attention_mask_heavy, 
                cdr_ids_heavy,
                labels=None,
                return_embeddings = False):

        outputs_l = self.model_l.bert(
                input_ids=input_ids_light,
                attention_mask=attention_mask_light,
                output_hidden_states=True,
                output_attentions=True)

        outputs_h = self.model_h.bert(
                input_ids=input_ids_heavy,
                attention_mask=attention_mask_heavy,
                output_hidden_states=True,
                output_attentions=True)

        cdr_l = torch.index_select(outputs_l[0], dim=1, index=cdr_ids_light[0])
        cdr_h = torch.index_select(outputs_h[0], dim=1, index=cdr_ids_heavy[0])

        cdr_l = torch.mean(cdr_l, dim=1)
        cdr_h = torch.mean(cdr_h, dim=1)

        embeddings = torch.concat([cdr_l, cdr_h],axis=-1)
        pred = self.epitopes(embeddings)        

        if labels is not None:
            loss = self.loss(pred.view(-1), labels.view(-1).float())
        else:
            loss = None

        return {"logits":pred, 
                "loss": loss, 
                "labels":labels,
                "embeddings":embeddings}

