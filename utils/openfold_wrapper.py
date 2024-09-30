from pathlib import Path
import sys

# sys.path.append(str(Path(__file__).resolve().parent))
# print(str(Path(__file__).resolve().parent))
import utils.openfold.np.protein as protein
import utils.openfold.np.residue_constants as residue_constants
from utils.openfold.config import model_config
from utils.openfold.data import feature_pipeline
from utils.openfold.utils.script_utils import load_models_from_command_line, prep_output
from utils.openfold.utils.tensor_utils import dict_multimap, tensor_tree_map
from utils.openfold.np import residue_constants
import numpy as np
from utils.protein_utils import (
    load_protein,
    pdb_str_to_dataframe,
    save_pdb,
    proteinTask,
)
from functools import partial
import torch
from scipy.spatial import distance
import json


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


class OFSVMModel:
    def __init__(self, features_list):
        self.features_list = features_list
        self.scaler = StandardScaler()
        self.svr = None
        pass

    def train(self, features_dict):
        X, Y = [], []
        self.svr = SVR()
        for name, feats in features_dict.items():
            X_ = feats.get_features(self.features_list)[0]
            Y_ = feats.target
            X.append(X_)
            Y.append(Y_)
        X = np.array(X)
        X = self.scaler.fit_transform(X)
        self.svr.fit(X, Y)

    def test(self, features_dict):
        X, Y = [], []
        for name, feats in features_dict.items():
            X_ = feats.get_features(self.features_list)[0]
            Y_ = feats.target
            X.append(X_)
            Y.append(Y_)

        X = np.array(X)
        X = self.scaler.transform(X)
        Y_pred = self.svr.predict(X)

        if len([y_ is not None for y_ in Y])==len(Y):
            correlation_coefficient, p_value = pearsonr(Y, Y_pred)
        print(correlation_coefficient, p_value)
        print(Y, Y_pred)


class OFFeatures:
    def __init__(self, task_wt, task_mt, target=None, store_protein=False):
        self.protein_wt = None
        self.protein_mt = None
        if store_protein:
            self.protein_wt = task_wt.protein_job["protein_mt_of"]
            self.protein_mt = task_mt.protein_job["protein_mt_of"]
        self.feature_wt = task_wt.protein_job["embeddings"]
        self.feature_mt = task_mt.protein_job["embeddings"]
        self.target = target

    def get_features(self, features_list=["logits"]):
        all_feats = []
        for name in self.feature_wt:
            wt_feats = [self.feature_wt[name][f_name] for f_name in features_list]
            mt_feats = [self.feature_mt[name][f_name] for f_name in features_list]
            all_feats.append(np.concatenate(wt_feats + mt_feats, axis=0))
        return all_feats


def get_embeddings_by_residues(cycle, residues, pdb_df):
    """
    :param cycle: output of openfold
    :param residues: list of residue in format [(residue_number, chain_id) ... ]
    :param pdb_df: biopandas protein dataframe
    :return: of embeddings and metrics that corresponds to residues list
    """
    # ['sm', 'msa', 'pair', 'single', 'final_atom_positions', 'final_atom_mask', 'final_affine_tensor', 'lddt_logits',
    # 'plddt', 'distogram_logits', 'masked_msa_logits', 'experimentally_resolved_logits', 'tm_logits',
    # 'predicted_tm_score', 'aligned_confidence_probs', 'predicted_aligned_error', 'max_predicted_aligned_error']

    pdb_df_ca = pdb_df[pdb_df["atom_name"] == "CA"].reset_index(drop=True)
    embeddings = {}
    for residue in residues:
        resi = residue["resi"]
        chain_id = residue["chain_id"]
        is_deletion = residue["deletion"]
        p_resi = pdb_df_ca[
            (pdb_df_ca["residue_number_original"] == resi)
            & (pdb_df_ca["chain_id_original"] == chain_id)
        ]

        if p_resi.shape[0] == 0 and is_deletion:
            """
            if it is a deletion we want to take embeddings of previous residue
            """
            p_resi = pdb_df_ca[
                (pdb_df_ca["residue_number_original"] == resi - 1)
                & (pdb_df_ca["chain_id_original"] == chain_id)
            ]

        if p_resi.shape[0] == 0 and is_deletion:
            """
            if it is a C-terminus mutation
            """
            p_resi = pdb_df_ca[
                (pdb_df_ca["residue_number_original"] == resi + 1)
                & (pdb_df_ca["chain_id_original"] == chain_id)
            ]
        resi = p_resi.index[0]
        embeddings[(resi, chain_id)] = {}
        embeddings[(resi, chain_id)]["msa"] = cycle["msa"][0, resi, :]
        embeddings[(resi, chain_id)]["pair"] = cycle["pair"][resi, resi, :]
        embeddings[(resi, chain_id)]["single"] = cycle["single"][resi, :]
        embeddings[(resi, chain_id)]["lddt_logits"] = cycle["lddt_logits"][resi, :]
        embeddings[(resi, chain_id)]["plddt"] = np.array([cycle["plddt"][resi]])[
            None, :
        ]
        embeddings[(resi, chain_id)]["distogram_logits"] = cycle["distogram_logits"][
            resi, :
        ]
        embeddings[(resi, chain_id)]["tm_logits"] = cycle["tm_logits"][resi, resi, :]
        embeddings[(resi, chain_id)]["aligned_confidence_probs"] = cycle[
            "aligned_confidence_probs"
        ][resi, resi, :]
        embeddings[(resi, chain_id)]["predicted_aligned_error"] = np.array(
            [cycle["predicted_aligned_error"][resi, resi]]
        )
        embeddings[(resi, chain_id)]["single_sm"] = cycle["sm"]["single"][resi, :]
        embeddings[(resi, chain_id)]["max_predicted_aligned_error"] = np.array(
            [cycle["max_predicted_aligned_error"]]
        )
        embeddings[(resi, chain_id)]["predicted_tm_score"] = np.array(
            [cycle["predicted_tm_score"]]
        )
    return embeddings


class OpenFoldWraper:
    def __init__(
        self,
        device="mps",
        weights_path="/Users/ivanisenko/projects/ProteinAIDesign/CGMover/alphafold/params_model_2_ptm.npz",
    ):
        self.device = device
        self.init_alphafold(weights_path)

    def prepare_features(self, target_protein):
        """prepare protein features for AlphaFold calculation"""
        sequence = residue_constants.aatype_to_str_sequence(target_protein.aatype)
        features = {
            "template_all_atom_positions": target_protein.atom_positions[None, ...],
            "template_all_atom_mask": target_protein.atom_mask[None, ...],
            "template_sequence": [sequence],
            "template_aatype": target_protein.aatype[None, ...],
            "template_domain_names": [None],  # f''.encode()]
        }
        num_templates = features["template_aatype"].shape[0]
        """ look for more elegant way to calculate sequence features """
        sequence_features = {}
        num_res = len(sequence)
        sequence_features["aatype"] = residue_constants.sequence_to_onehot(
            sequence=sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )
        sequence_features["between_segment_residues"] = np.zeros(
            (num_res,), dtype=np.int32
        )
        sequence_features["domain_name"] = np.array(
            ["input".encode("utf-8")], dtype=np.object_
        )
        sequence_features["residue_index"] = np.array(
            target_protein.residue_index, dtype=np.int32
        )
        sequence_features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
        sequence_features["sequence"] = np.array(
            [sequence.encode("utf-8")], dtype=np.object_
        )
        deletion_matrix = np.zeros(num_res)
        sequence_features["deletion_matrix_int"] = np.array(
            deletion_matrix, dtype=np.int32
        )[None, ...]
        int_msa = [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
        sequence_features["msa"] = np.array(int_msa, dtype=np.int32)[None, ...]
        sequence_features["num_alignments"] = np.array(np.ones(num_res), dtype=np.int32)
        sequence_features["msa_species_identifiers"] = np.array(["".encode()])
        feature_dict = {**sequence_features, **features}
        return feature_dict

    def init_alphafold(
        self,
        weights_path="/Users/ivanisenko/projects/ProteinAIDesign/CGMover/alphafold/params_model_2_ptm.npz",
    ):
        """get config, prepare feature_processor and load model"""
        self.config = model_config("model_2_ptm", low_prec=(True))
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        model_generator = load_models_from_command_line(
            self.config, self.device, None, weights_path, None
        )
        for model, output_directory in model_generator:
            self.model = model
            output_directory = output_directory
        self.model.to(self.device)

    def get_batch_features(self, af_output_batch, af_input_features, batch_id):
        """
        Input: output of alphafold
               features for alphafold inference
               batch_id
        Output:
               output and features for batch_id
        """
        out = {"sm": {}}
        for k, out_ in af_output_batch.items():
            if k == "sm":
                for name in out_:
                    if out_[name].shape[0] == 1:
                        out["sm"][name] = out_[name][batch_id, ...]
                    if out_[name].shape[0] == 8:
                        out["sm"][name] = out_[name][:, batch_id, ...]
                continue
            if len(out_.shape) == 0:
                out[k] = out_
                continue
            out[k] = out_[batch_id, ...]
        features = tensor_tree_map(
            lambda x: np.array(x[batch_id, ..., -1].cpu()), af_input_features
        )

        return out, features

    def extract_residue_embeddings(self, output, id):
        output_upd = {}
        # print(output["msa"].shape, id)
        output_upd["msa"] = output["msa"][0, id, :]
        output_upd["pair"] = output["pair"][id, id, :]
        output_upd["lddt_logits"] = output["lddt_logits"][id, :]
        output_upd["distogram_logits"] = output["distogram_logits"][id, id, :]
        output_upd["aligned_confidence_probs"] = output["aligned_confidence_probs"][
            id, id, :
        ]
        output_upd["predicted_aligned_error"] = output["predicted_aligned_error"][
            id, id
        ].reshape(-1)
        output_upd["plddt"] = output["plddt"][id].reshape(-1)
        output_upd["single"] = output["single"][id, :]
        output_upd["tm_logits"] = output["tm_logits"][id, id, :]
        return output_upd

    def run_task(self, protein_task):
        pdb_df = protein_task.protein_job["protein_mt"]
        of_output, pdb_df_pred = self.inference_monomer(pdb_df)
        pdb_df_pred_ca = pdb_df_pred[pdb_df_pred["atom_name"] == "CA"].reset_index(
            drop=True
        )
        assert pdb_df_pred_ca.shape[0] == of_output["single"].shape[0], print(
            "OF output and PDB size dimensions doesn't match"
        )

        protein_task.protein_job["embeddings"] = {}
        for name, id in protein_task.protein_job["obs_positions"].items():
            protein_task.protein_job["embeddings"][name] = (
                self.extract_residue_embeddings(of_output, id=id)
            )

        protein_task.protein_job["protein_mt_of"] = pdb_df_pred

        return

    def inference_monomer(self, pdb_df, n_recycle=2, return_all_cycles=False):
        """inference openfold with pdb dataframe input
         0 cycle with template features
         1 cycle masked template features
        ...
        n cycle masked template features

        return alphafold output and predicted pdb structure
        return_all_cycles - return embeddings for each cycle
        """
        pdb = protein.from_pdb_df(pdb_df)
        resi = pdb.residue_index
        features = self.prepare_features(pdb)
        pdb_df_ca = pdb_df[pdb_df["atom_name"] == "CA"].reset_index()
        features["template_all_atom_mask"][
            :, pdb_df_ca[pdb_df_ca["mask"]].index, :
        ] *= 0
        processed_feature_dict = self.feature_processor.process_features(
            features,
            mode="predict",
        )
        """ add recycling features with masked template features """
        processed_feature_dict_list = [
            processed_feature_dict
        ]  # , processed_feature_dict
        # ]
        for i in range(n_recycle):
            processed_feature_dict_list.append(
                {k: p.detach().clone() for k, p in processed_feature_dict.items()}
            )
            processed_feature_dict_list[-1]["template_mask"] *= 0

        cat_fn = partial(torch.cat, dim=-1)
        processed_feature_dict = dict_multimap(cat_fn, processed_feature_dict_list)

        for c, p in processed_feature_dict.items():
            if p.dtype == torch.float64:
                processed_feature_dict[c] = torch.as_tensor(
                    p, dtype=torch.float32, device=self.device
                )
            else:
                processed_feature_dict[c] = torch.as_tensor(p, device=self.device)
            processed_feature_dict[c] = processed_feature_dict[c][None, ...]

        """ load alphafold model """

        with torch.no_grad():
            out_batch, out_per_cycle = self.model(processed_feature_dict)

        for i in range(len(out_per_cycle)):
            out_per_cycle[i] = tensor_tree_map(
                lambda x: np.array(x[...].detach().cpu()), out_per_cycle[i]
            )
            out_per_cycle[i], _ = self.get_batch_features(out_per_cycle[i], {}, 0)
        out_batch = tensor_tree_map(
            lambda x: np.array(x[...].detach().cpu()), out_batch
        )
        out, ifd = self.get_batch_features(out_batch, processed_feature_dict, 0)
        unrelaxed_protein = prep_output(
            out, ifd, ifd, self.feature_processor, "model_2_ptm", 200, False
        )
        pdb_str = protein.to_pdb(unrelaxed_protein)
        pdb_df_pred = pdb_str_to_dataframe(pdb_str, pdb_df)

        if return_all_cycles:
            return out, out_per_cycle, pdb_df_pred

        return out, pdb_df_pred

    def get_structure_metrics(self, cycle):
        """
        :param cycle:
        :param pdb_df:
        :param residues:
        :return:
        """
        plddt = np.average(cycle["plddt"])
        pae = np.average(cycle["predicted_aligned_error"])
        return {"pae_average": pae.astype(float), "plddt_average": plddt.astype(float)}

    def get_structure_metrics_by_interface(self, cycle, pdb_df, interface_chain):
        """
        :param cycle: of output
        :param pdb_df: pdb dataframe
        :param residues:  residue in format [(resi1,chain1), ... (resin,chainn)]
        :param include_neighbors: include other residues within 5.0 A from residues
        :return:
        """
        pdb_df = pdb_df[pdb_df["atom_name"] == "CA"].reset_index()
        pdb_df_A = pdb_df[pdb_df["chain_id_original"] == interface_chain]
        pdb_df_B = pdb_df[pdb_df["chain_id_original"] != interface_chain]

        resi_index_A = list(pdb_df_A.index)
        resi_index_B = list(pdb_df_B.index)

        xyz1 = pdb_df_A[["x_coord", "y_coord", "z_coord"]].to_numpy()
        xyz2 = pdb_df_B[["x_coord", "y_coord", "z_coord"]].to_numpy()
        pae = cycle["predicted_aligned_error"]

        cd = distance.cdist(xyz1, xyz2)
        pae_scores = [
            pae[resi_index_A[i1], resi_index_B[i2]]
            for i1, i2 in zip(*np.where(cd < 8.0))
        ]
        pae_scores += [
            pae[resi_index_B[i2], resi_index_A[i1]]
            for i1, i2 in zip(*np.where(cd < 8.0))
        ]

        ids_1 = [resi_index_A[i] for i in sorted(list(set(np.where(cd < 8.0)[0])))]
        ids_2 = [resi_index_B[i] for i in sorted(list(set(np.where(cd < 8.0)[1])))]
        plddt = np.average(cycle["plddt"][ids_1 + ids_2])
        # pae_1 = np.average(pae[ids_1, :][:, ids_2])
        # pae_2 = np.average(pae[:, ids_1][ids_2, :])
        return {
            "pae_average": np.average(pae_scores).astype(float),
            "plddt_average": plddt.astype(float),
        }

    def get_structure_metrics_by_residues(
        self, cycle, pdb_df, residues, include_neighbors=False
    ):
        """
        :param cycle: of output
        :param pdb_df: pdb dataframe
        :param residues:  residue in format [(resi1,chain1), ... (resin,chainn)]
        :param include_neighbors: include other residues within 5.0 A from residues
        :return:
        """
        pdb_df = pdb_df[pdb_df["atom_name"] == "CA"].reset_index()
        resi_index = []
        for resi, chain in residues:
            residue_df = pdb_df[
                (pdb_df["residue_number_original"] == resi)
                & (pdb_df["chain_id_original"] == chain)
            ]
            resi_index += list(residue_df.index)

        if include_neighbors:
            xyz = pdb_df[["x_coord", "y_coord", "z_coord"]].to_numpy()
            xyz_resi = xyz[resi_index, :]
            cd = distance.cdist(xyz, xyz_resi)
            resi_and_neighbors = sorted(list(set(np.where(cd < 5.0)[0])))
        else:
            resi_and_neighbors = resi_index
        plddt = np.average(cycle["plddt"][resi_and_neighbors])
        pae = cycle["predicted_aligned_error"]
        pae_1 = np.average(pae[resi_and_neighbors, :])
        pae_2 = np.average(pae[:, resi_and_neighbors])
        return {"pae_average": pae_1 + pae_2, "plddt_average": plddt}

    def get_embeddings_by_residues(
        self, cycle, pdb_df, residues, include_neighbors=False
    ):
        for s in cycle:
            if s == "sm":
                break
            print(cycle[s].shape)

    def extract_svm_features(
        self, output_1, output_2, ids=None, feature_list=["lddt_logits", "plddt"]
    ):
        features_A = self.extract_svm_features_single(output_1, ids, feature_list)
        features_B = self.extract_svm_features_single(output_2, ids, feature_list)
        return np.concatenate([features_A, features_B], axis=1)

    def extract_svm_features_single(
        self, output, ids=None, feature_list=["lddt_logits", "plddt"]
    ):
        """
        function to extract and concatenate embeddings from the openfold output
        :param cycle:
        :param feature_list:
        :return:
        """
        features = []
        if ids is None:
            ids = output["lddt_logits"].shape[0]
        for f_name in feature_list:
            if len(output[f_name].shape) == 3:
                # print(output[f_name].shape)
                d = []
                for ii in ids:
                    d.append(output[f_name][ii, ii, :])
                d = np.array(d)
                # d = np.diagonal(output[f_name]).T#.shape)
                # print(output[f_name][ids, :,:].shape)#[ids, :].shape)
                # exit(0)
                # print(d.shape)
                # print(d[ids].shape)
                # exit(0)
                features.append(d)  # [ids])#np.diagonal(output[f_name])[ids])
                # features.append(output[f_name][ids, ...][ids,...])# ids, :])
            elif len(output[f_name].shape) == 1:
                features.append(output[f_name][ids, None])
            else:
                features.append(output[f_name][ids])

        return np.concatenate(features, axis=1)


if __name__ == "__main__":
    pdb_df = load_protein("../tests/test_dimer.pdb")
    ofr = OpenFoldWraper()
    of_output, pdb_df_pred = ofr.inference_monomer(pdb_df)
    print(ofr.get_structure_metrics(of_output))
    print(
        ofr.get_structure_metrics_by_interface(of_output, pdb_df, interface_chain="A")
    )
    save_pdb(pdb_df_pred, "../tests/test.pdb")
