import itertools
import json
import seaborn as sns
import matplotlib.pyplot as plt
import random
import re
import shutil
import argparse
from matplotlib.patches import Rectangle
import pandas as pd
import os
import pickle
import subprocess
import numpy as np
from utils.openfold_wrapper import OpenFoldWraper
from pathlib import Path
from rbdaim.epitope_model import AntiBERTyFAB_CDR_Pair, AntiBERTyFAB_CLS_Pair, EpitopesPairDatasetCDR
import torch
from torch import nn
from abnumber import Chain
from fab_similarity import compare_seqs_anarci
from train_classifier import class_metrics, calculate_metrics_by_model

from utils.protein_utils import (
    load_protein,
    save_pdb,
    get_sequence,
    mutate_protein,
    make_dummy_protein,
    aa_3_to_1,
    aa_1_to_3
)

def fab_align(seq1,seq2):
    """
    align antibody sequences
    :param seq1:
    :param seq2:
    :return:
    """
    alignment = seq1.align(seq2)
    n = 0
    nt = 0
    al_seq = ["",""]
    for pos, (aa, bb) in alignment:
        al_seq[0]+=aa
        al_seq[1]+=bb
    return al_seq


def kalign(
    seq1="MVLTIYPDELVQIVSDKIASNKDKPFWYILAESTLQKEVYFLLAH",
    seq2="MVLTIYPDELVQDKPFWYILAESTLQKEVYFLLAH",
):
    """
    align antigen sequences
    :param seq1:
    :param seq2:
    :return:
    """
    fasta_sequences = f""">protein1
{seq1}
>protein2
{seq2}
"""
    command = ["kalign"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(input=fasta_sequences)
    al_seq = []
    for r in stdout.split("\n"):
        if len(r) == 0:
            continue
        if r[0] == ">":
            al_seq.append([])
            continue
        if len(al_seq) == 0:
            continue
        al_seq[-1].append(r.rstrip())
    al_seq[0] = "".join(al_seq[0])
    al_seq[1] = "".join(al_seq[1])

    return al_seq
    

def align_points(A, B):
    """
    A,B - input xyz coordiantes
    output - rotation matrix and translation vector
    """
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B - np.dot(R, centroid_A)
    return R, t


def align_complex(target_protein,
                  ref_rbd_path = "./datasets/ref.pdb"):
    """
    target_path: path to protein to rotate, chain_id of RBD must be '1'
    ref_pdb_path: path to reference RBD pdb
    """

    ref_rbd = load_protein(ref_rbd_path)

    target_rbd = target_protein[target_protein["chain_id_original"]=='1']
    # perform operations only on CA atoms
    target_rbd = target_rbd[target_rbd["atom_name"]=="CA"].reset_index(drop=True)
    ref_rbd = ref_rbd[ref_rbd["atom_name"] =="CA"].reset_index(drop=True)

    target_seq = get_sequence(target_rbd)
    ref_seq = get_sequence(ref_rbd)

    
    target_seq_al, ref_seq_al = kalign(target_seq, ref_seq) # get aligned sequences
    target_i = -1
    ref_i = -1
    match = []
    for s1,s2 in zip(target_seq_al, ref_seq_al):
        if s1!='-':
            target_i+=1
        if s2!='-':
            ref_i+=1
        if s1=='-' or s2=='-':
            continue
        match.append((target_i,ref_i))
    ids_t = [r[0] for r in match]
    ids_r = [r[1] for r in match]
    
    target_rbd = target_rbd.iloc()[ids_t]
    ref_rbd = ref_rbd.iloc()[ids_r]
    
    target_xyz = target_rbd[["x_coord", "y_coord", "z_coord"]].to_numpy()
    ref_xyz = ref_rbd[["x_coord", "y_coord", "z_coord"]].to_numpy()

    # aligned sequences for matched aa pairs
    R,t = align_points(target_xyz, ref_xyz)
    target_xyz =  target_protein[["x_coord", "y_coord", "z_coord"]].to_numpy()
    transformed_points = np.dot(target_xyz, R.T) + t
    
    # update coordinates
    target_protein["x_coord"] = transformed_points[:,0]
    target_protein["y_coord"] = transformed_points[:,1]
    target_protein["z_coord"] = transformed_points[:,2]
    return target_protein



def get_model_config():
    config = {"num_classes": 12, "model_type": "CDR", "epochs": 4}
    config["device"] = "mps"
    return config


def load_classifier_models(postfix = "_full",
                           ensemble = False):

    config = get_model_config()
    models = []
    for i in range(5):
        for m in ["CDR", "CLS"]:
            print(i, "rbdaim")

            if postfix=="_full":
                model_name = f"./weights/NN_cv_{i}_{m}{postfix}.pth"
            else:
                model_name = f"./weights/NN_cv_0_{m}_{i}{postfix}.pth"
            if m == "CDR":
                model = AntiBERTyFAB_CDR_Pair(get_model_config())
            if m == "CLS":
                model = AntiBERTyFAB_CLS_Pair(get_model_config())

            state_dict = torch.load(model_name)
            model.load_state_dict(state_dict)
            model.to(config["device"])
            model.eval()
            models.append(model)
        if not ensemble:
            break
    return models

def predict_classifier_probs(data, models):
    probs = []
    for d_ in data.iloc():
        ps = []
        for model in models:
            p = predict_epitopes_probabilities(sequence_heavy=d_["heavy_upd"],
                                           sequence_light=d_["light_upd"],
                                           df=d_,
                                           model=model)["scores"]
            ps.append(p)
        ps = np.array(ps)
        ps = np.average(ps,axis=0)
        probs.append(ps.flatten())
    return probs

def predict_levinstein_scores(data, dataset):
    scores = []
    fabs = []
    for d_ in data.iloc():
        ps, hits = compare_seqs_anarci(dataset=dataset,
                                target_seq_h=d_["heavy_upd"],
                                target_seq_l=d_["light_upd"])
        ps = np.average([ps],axis=0)
        scores.append(ps.flatten())
        fabs.append(hits)
    return scores, fabs


def preprocess_dataset():
    d = pd.read_json("./datasets/pdb_classes_3.json")
    pdb_dict = {}
    pdb_dict_r = {}
    for d_ in d.iloc():
        pdb_name = d_['pdbid']+"_"+d_["chids"][0]+"_"+d_["chids"][1]
        pdb_dict[pdb_name] = d_["Name"]
        pdb_dict_r[d_["Name"]] = pdb_name
    dataset = pickle.load(open("./datasets/dataset_ab.pkl",'rb'))
    pdb_ids = []
    for d in dataset.iloc():
        pdb_id = None
        if d["good"]==1:
            pdb_id = pdb_dict_r[d["Name"]]
        pdb_ids.append(pdb_id)
    dataset["pdb_ids"] = pdb_ids
    pickle.dump(dataset, open("./datasets/dataset_production.pkl",'wb'))


def predict_epitopes_probabilities(sequence_light,
              sequence_heavy,
                                   df = None,
                                   model = None):
    """
    In this example we return random score
    But it must be replaced with AntiBerty-based mdoel
    """
    if isinstance(sequence_light, str):
        chain_l = Chain(sequence_light, scheme='chothia')
        chain_h = Chain(sequence_heavy, scheme='chothia')

    else:
        chain_l = sequence_light
        chain_h = sequence_heavy


    l_ids_cdr = list(range(24,35))+list(range(50,57))+list(range(89,98))
    h_ids_cdr = list(range(31,36))+list(range(50,66))+list(range(95,103))

    n_light_cdr = []
    n_heavy_cdr = []

    seq_light_anarci = []
    seq_heavy_anarci = []

    n = 0
    for pos, aa in chain_l:
        i = str(pos)[1:]
        if not i[-1].isdigit():
            i = i[:-1]
        i = int(i)
        n+=1
        if i in l_ids_cdr:
            n_light_cdr.append(n)#i-1)
        seq_light_anarci.append(aa)

    n = 0
    for pos, aa in chain_h:
        i = str(pos)[1:]
        if not i[-1].isdigit():
            i = i[:-1]
        i = int(i)
        n+=1
        if i in h_ids_cdr:
            n_heavy_cdr.append(n)#i-1)
        seq_heavy_anarci.append(aa)

    df = pd.DataFrame({"light_ids_cdr":[np.array(n_light_cdr)],
                       "heavy_ids_cdr":[np.array(n_heavy_cdr)],
                       "light_anarci":["".join(seq_light_anarci)],
                       "heavy_anarci":["".join(seq_heavy_anarci)],
                       "POS_class":[-1],
                       "index":[0]})

    config = get_model_config()
    dataset = EpitopesPairDatasetCDR(df, config["num_classes"])

    with torch.no_grad():
        d= dataset[0]
        d.pop("antibody_id")
        d = {k:v[None,:] for k,v in d.items()}
        d = {k:torch.tensor(v,device=config["device"]) for k,v in d.items()}
        res = model(**d)["logits"].detach().cpu().numpy()

    scores = res
    epitopes = ["A", "B", "C", "D1", "D2", "E1", "E21", "E22", "E3", "F1", "F2", "F3"]
    return {"scores":scores,
            "epitopes":epitopes}


def process_anarci_sequences(chain_l, chain_h):
    """
    function to derive Chothia sequence and numbering
    """
    l_ids_cdr = list(range(24, 35)) + list(range(50, 57)) + list(range(89, 98))
    h_ids_cdr = list(range(31, 36)) + list(range(50, 66)) + list(range(95, 103))
    n_light_cdr = []
    n_heavy_cdr = []
    seq_light_anarci = []
    seq_heavy_anarci = []
    n = 0
    for pos, aa in chain_l:
        i = str(pos)[1:]
        if not i[-1].isdigit():
            i = i[:-1]
        i = int(i)
        n += 1
        if i in l_ids_cdr:
            n_light_cdr.append(n)  # i-1)
        seq_light_anarci.append(aa)

    n = 0
    for pos, aa in chain_h:
        i = str(pos)[1:]
        if not i[-1].isdigit():
            i = i[:-1]
        i = int(i)
        n += 1
        if i in h_ids_cdr:
            n_heavy_cdr.append(n)  # i-1)
        seq_heavy_anarci.append(aa)

    df = pd.DataFrame({"light_ids_cdr": [np.array(n_light_cdr)],
                       "heavy_ids_cdr": [np.array(n_heavy_cdr)],
                       "light_anarci": ["".join(seq_light_anarci)],
                       "heavy_anarci": ["".join(seq_heavy_anarci)],
                       "POS_class": [-1],
                       "index": [0]})
    return df

def model_rbd(template = "test/7Z0X_H_L.pdb",
              input = None, 
              chains = [],
              ofr = None):
    
    rbd = load_protein("datasets/ref.pdb")
    rbd["chain_id_original"]="1"
    template = load_protein(template)
    template["residue_number"]+=rbd.iloc()[-1]["residue_number"]+25
    protein = pd.concat([rbd, template], axis=0).reset_index(drop=True)
    protein["line_idx"] = protein.index
    protein["atom_number"] = protein.index
    metrics, model = homology_model_multimer(input,
                                    chains,
                                    protein,
                                    ofr)
    return metrics, model


def homology_model(input_sequence, protein):
    p_ca = protein[protein["atom_name"] == "CA"].reset_index(drop=True)
    seq = [aa_3_to_1[r] for r in list(p_ca["residue_name"])]
    seq = "".join(seq)

    if isinstance(input_sequence,Chain):
        seq_target, seq_template = fab_align(input_sequence, Chain(seq, scheme="chothia"))
    else:
        seq_target, seq_template = kalign(input_sequence, seq)

    dummy_protein = make_dummy_protein(str(input_sequence))

    n_t = 0
    seq_len = len(seq_target.replace("-", ""))
    protein_model = [None for _ in range(seq_len)]
    n_target, n_template = -1, -1

    for i in range(len(seq_target)):
        resi = None
        if seq_template[i] != "-":
            n_template += 1
            resi = p_ca.iloc()[n_template]["residue_number"]
            residue = protein[protein["residue_number"] == resi]
        if seq_target[i] != "-":
            n_target += 1
        if seq_target[i] == "-":
            continue
        if resi is None:
            residue = dummy_protein[dummy_protein["residue_number"] == n_target+1]
        if seq_target[i]!=aa_3_to_1[residue["residue_name"].iloc()[0]]:
            residue["residue_name"] = aa_1_to_3[seq_target[i]]
            residue = residue[residue["atom_name"].isin(["N","C","CA","O"])]
        residue["residue_number"] = n_target
        protein_model[n_target] = residue

    protein_model = pd.concat(protein_model, axis=0).reset_index(drop=True)
    pa = protein_model[protein_model["atom_name"]=="CA"]
    protein_model["line_idx"] = protein_model.index
    protein_model["residue_number_original"] = protein_model["residue_number"]
    protein_model["atom_number"] = protein_model.index
    return protein_model


def homology_model_multimer(input_sequence,
                            chains,
                            template_protein,
                            ofr):
    protein_model = []
    n_residue = 1
    for chain in chains:
        seq = input_sequence[chain]
        template_protein_ = template_protein[template_protein["chain_id_original"]==chain]
        if len(protein_model)!=0:
            n_residue = protein_model[-1].iloc()[-1]["residue_number"]+25
        else:
            n_residue = 1
        protein_model_ = homology_model(seq, template_protein_)
        protein_model_["residue_number"]+=n_residue
        protein_model_["chain_id_original"] = chain
        protein_model.append(protein_model_)
        
    protein_model = pd.concat(protein_model).reset_index(drop=True)
    protein_model["line_idx"] = protein_model.index
    protein_model["atom_number"] =protein_model.index

    of_output_0,  pdb_df_pred_0 = ofr.inference_monomer(
        protein_model, n_recycle=2
    )
    of_output_0 = {o:of_output_0[o] for o in ["plddt",
                                              "predicted_aligned_error"]}

    return of_output_0, pdb_df_pred_0


def find_homologues(sequence_light,
                    sequence_heavy,
                    epitope_scores,
                    device="cuda:0"):
    """
    In this example we picke random sequence
    But it must be replaced with AntiBerty-based mdoel
    """
    data = pd.read_json("./datasets/pdb_classes_3.json")
    
    hit_ids = [0,1,2]

    ofr = OpenFoldWraper(device=device,
                         weights_path="./weights/params_model_2_ptm.npz")

    plddt_scores = []
    models = []
    names = []
    for i in hit_ids:
        d_hit = data.iloc()[i]
        template_id = d_hit["pdbid"]+"_"+d_hit["chids"][0]+"_"+d_hit["chids"][1]

        template_path = "./datasets/pdb_clean_ab_only/"+template_id+".pdb"
        names.append([d_hit["Name"], template_id])

        input = {d_hit["chids"][0]:sequence_heavy,#[:15],
                 d_hit["chids"][1]:sequence_light,#[:15],
                 "1":"NLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGP"}
        input["1"] = input["1"]#[:15]
        
        metrics, model = model_rbd(template=template_path,
                                   chains=[d_hit["chids"][0], d_hit["chids"][1], "1"],
                                   input=input,
                                   ofr=ofr)
        
        plddt_scores.append(np.average(metrics["plddt"]))
        models.append(model)
        
    n_max = np.argmax(plddt_scores[i])
    
    return models[n_max], plddt_scores[n_max], names[n_max]
        

def plot_scores(scores, output):             
    fig,ax = plt.subplots(figsize=(10,1))
    ax = sns.heatmap(scores["scores"],
                     xticklabels=scores["epitopes"],
                     yticklabels=["P"],
                     annot=True,
                     cmap='viridis')
    plt.tight_layout()
    plt.savefig(output)

def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def get_weights(weights_dir="./weights/"):
    OF_WEIGHTS = "params_model_2_ptm.npz"
    weights_path = os.path.join(weights_dir, OF_WEIGHTS)
    
    if not os.path.exists(weights_path):
        SOURCE_OF_WEIGHTS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-12-06.tar"
        tmp_path = f"{weights_dir}/af_colab.tar"
        if not os.path.exists(tmp_path):
            runcmd(f"wget -O {tmp_path} {SOURCE_OF_WEIGHTS_URL}")
        runcmd(f"""tar --extract  --file={tmp_path} --directory={weights_dir}""")
                   
        for name in ["af_colab.tar",
                     "params_model_1.npz",
                     "params_model_1_multimer_v3.npz",
                     "params_model_2.npz",
                     "params_model_2_multimer_v3.npz",
                     "params_model_3.npz",
                     "params_model_3_multimer_v3.npz",
                     "params_model_4.npz",
                     "params_model_4_multimer_v3.npz",
                     "params_model_5.npz",
                     "params_model_5_multimer_v3.npz"]:
            runcmd(f"rm {weights_dir}/{name}")      


def load_epitopes_df():
    """
    load dataframe with per-residue labeling of RBD domain if it belongs to the epitope class
    """
    epitope_ids = ["A", "B", "C", "D1", "D2", "E1", "E21", "E22", "E3", "F1", "F2", "F3"]
    epitope_df = pd.read_csv('./datasets/rbd_classes.csv',
                             names=["residue_number"]+epitope_ids,
                             skiprows=1,
                             delimiter=";")
    return epitope_df


def get_putative_escape_mutations(epitope_class = "E1",
                                  user_seq = None):
    sequences = json.load(open("./datasets/rbd_strains.json"))
    if user_seq is None:
        user_seq = sequences["Wuhan"]
    sequences["user"] = user_seq
    epitope_df = load_epitopes_df()

    mutations = {}
    for name,seq in sequences.items():
        muts, df_ = get_mutations_list(seq, epitope_df)
        mutations[name] = muts[epitope_class]
        if name == 'user':
            df_align = df_
    # mutations = {name:get_mutations_list(seq, epitope_df)[epitope_class] for name,seq in sequences.items()}
    strains_list = ['user', 'Wuhan', '24A_(JN.1)', '21C_(Epsilon)', '20J_(Gamma)', '23I_(BA.2.86)', '23H_(HK.3)', '21A_(Delta)', '21L_(BA.2)', '22B_(BA.5)', '23F_(EG.5.1)', '23E_(XBB.2.3)', '22D_(BA.2.75)', '21K_(BA.1)', '23B_(XBB.1.16)', '20H_(Beta)', '21G_(Lambda)', '21D_(Eta)', '21F_(Iota)', '23A_(XBB.1.5)', '23D_(XBB.1.9.1)', '20I_(Alpha)', '22E_(BQ.1.1)', '21H_(Mu)', '21B_(Kappa)', '23G_(XBB.1.5.70)', '22F(XBB.1)', '24B_(JN.1.11.1)', '22C_(BA.2.12.1)', '22A_(BA.4)']

    escape_matrix = np.zeros((len(strains_list), len(strains_list)))
    for i,j in itertools.combinations(range(len(strains_list)),2):
        reference_strain = mutations[strains_list[i]]
        new_strain = mutations[strains_list[j]]
        difference = [r for r in new_strain if r not in reference_strain]
        #print(strains_list[i], strains_list[j], len(difference))
        escape_matrix[i,j] = len(difference)
        escape_matrix[j,i] = len(difference)

    df = pd.DataFrame(escape_matrix, index=strains_list, columns=strains_list)

    return df, df_align

def get_mutations_list(target_seq,
                       epitope_df):

    """
    find mutations in the input sequeunce compare to reference Wuhan sequence
    sort mutations based on the epitopes class
    """

    wuhan_seq = "NLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGP"#sequences["Wuhan"]
    alignment = []

    aligned_ref_seq, aligned_tar_seq = kalign(wuhan_seq, target_seq)
    for i, [a1,a2] in enumerate(zip(list(aligned_ref_seq),
                                    list(aligned_tar_seq))):
        if a1!="-":
            alignment.append([])
        if len(alignment) != 0:
            alignment[-1].append(a2)

    alignment = ["".join(a) for a in alignment]
    mutations = []
    for i,a in enumerate(alignment):
        if a!=aligned_ref_seq[i]:
            mutations.append(aligned_ref_seq[i]+str(i+334)+a)

    epitope_ids = list(epitope_df)[1:]
    epitope_mutations = {epitope_id:[] for epitope_id in epitope_ids}

    for mutant in mutations:
        pos = int(re.findall(r'\d+', mutant)[0])
        ec = epitope_df[epitope_df["residue_number"]==pos].to_numpy()[0,1:]
        for i in np.where(ec>0)[0]:
            epitope_mutations[epitope_ids[i]].append(mutant)

    alignment_df = pd.DataFrame({'ref_rbd': [a for a in aligned_ref_seq],
                                 'user_rbd': [a for a in aligned_tar_seq]})

    return epitope_mutations, alignment_df
    
def main():

    parser = argparse.ArgumentParser(description='Prediction of RBD epitopes')
    parser.add_argument('--fab_light_sequence', default="ELVMTQTPLSLPVSLGDQASISCRSSQNGNTYLEWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPRTFGGGTKLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGSIVHS", help='light antibody sequence')
    parser.add_argument('--fab_heavy_sequence', default="QVQLVESGGGLVQPGGSLRLSCATSGFTFTDYYMSWVRQPPGKALEWLGFIRNGYTTEYSASVKGRFTISRDNSQSILYLQMRAEDSATYYCARDGSYAMDYWGQGTSVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPNKANTL", help='heavy antibody sequence')
    parser.add_argument('--output_directory', default="./output/", help='output_directory')
    parser.add_argument('--device', default="cuda:0", help='device')

    args = parser.parse_args()
    epitope_probs = predict_epitopes_probabilities(args.fab_light_sequence,
                                                   args.fab_heavy_sequence)

    model, score, hit_name = find_homologues(args.fab_light_sequence,
                                             args.fab_heavy_sequence,
                                             epitope_probs,
                                             device=args.device)

    output = args.output_directory
    Path(output).mkdir(exist_ok=True)
    save_pdb(model, os.path.join(output, "rbdaim.pdb"))
    predicted_epitope = epitope_probs["epitopes"][np.argmax(epitope_probs["scores"])]

    json.dump({"score":str(score),
               "epitope_probabilities":epitope_probs,
               "predicted_epitope":predicted_epitope,
               "model_path":os.path.join(output, "rbdaim.pdb"),
               "closest_template":hit_name},
               open(os.path.join(output, "results.json"),'w'))

    plot_scores(epitope_probs, output+"/heatmap.png")

    print("Results:")
    print("Score of predicted protein", str(score))
    print("Eptiope probabilities:", epitope_probs)
    print("Predicted epitope:", predicted_epitope)
    print("Path of PDB rbdaim:", os.path.join(output, "rbdaim.pdb"))
    print("Closest Fab template:", hit_name)
    
if __name__ == "__main__":
    main()
    
