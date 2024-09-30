# import pyrosettacolabsetup; pyrosettacolabsetup.install_pyrosetta()
import pyrosetta
import itertools
import numpy as np
pyrosetta.init("-ignore_unrecognized_res 1 -ex1 -ex2 -flip_HNQ")
# import pyrosetta
from pyrosetta.teaching import *
from pyrosetta import *
import os, sys, urllib
from rosetta.protocols.minimization_packing import *
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code

class RosettaWrapper:
    def __init__(self):
        self.scorefxn = pyrosetta.create_score_function("ref2015_cart")
        pass

    def load_pose(self, path):
        return pyrosetta.pose_from_pdb(path)




    def relax_pose(self, input_pdb_path, output_pdb_path, max_iter = 100):
        pose = pyrosetta.pose_from_pdb(input_path)
        movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
        movemap.set_bb(True)
        movemap.set_chi(True)
        relax = pyrosetta.rosetta.protocols.relax.FastRelax()
        relax.constrain_relax_to_native_coords(True)
        relax.cartesian(True)
        relax.set_scorefxn(self.scorefxn)
        relax.set_movemap(movemap)
        relax.max_iter(max_iter)
        relax.apply(pose)
        pose.dump_pdb(output_path)

    def repack_pose(self, input_pdb_path, output_pdb_path, max_iter = 100):
        pose = pyrosetta.pose_from_pdb(input_pdb_path)
        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())  # should be operation.PreventRepacking()?
        pack_task = tf.create_task_and_apply_taskoperations(pose)
        packer = PackRotamersMover(self.scorefxn, pack_task)
        packer.apply(pose)

        mm = MoveMap()
        mm.set_bb(True)
        mm.set_chi(True)

        min_mover = rosetta.protocols.minimization_packing.MinMover()
        min_mover.movemap(mm)
        min_mover.score_function(self.scorefxn)
        min_mover.min_type("lbfgs_armijo")
        min_mover.tolerance(1e-6)
        min_mover.apply(pose)

        pose.dump_pdb(output_pdb_path)
        return pose

    def mutate_residue(self, pose, mutant_code):
        """
        mutant_code -
        :param pose:
        :param mutant_code:  (aa, residue_position, chain) ...
        :return:
        """
        G_before = rr.scorefxn(pose)
        aa, resi, mutant_chain = mutant_code
        prevent_off = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
        prevent_in = pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT()
        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        #tf.push_back(pyrosetta.rosetta.core.pack.task.operation.PreventRepacking())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

        design_resi = f"{resi}{mutant_chain}"
        design_resi = rosetta.core.select.residue_selector.ResidueIndexSelector(design_resi)
        not_design_resi = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector( design_resi )
        sur_region = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(design_resi, 8.0, True)
        not_sur_region = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector( sur_region )
        not_resi = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(design_resi, 8.0, True)
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_in,
                                                                                       not_design_resi,#not_designable_interface,
                                                                                       False))
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_off,
                                                                                       not_sur_region,#not_designable_interface,
                                                                                       False))
        packer_task_design = tf.create_task_and_apply_taskoperations(pose)
        pdb2pose_resi = pyrosetta.rosetta.core.pose.get_pdb2pose_numbering_as_stdmap(pose)
        pose_resi = pdb2pose_resi[mutant_chain+str(resi)+"."]

        if aa != "*":
            mutant_aa = int(aa_from_oneletter_code(aa))
            aa_bool = pyrosetta.Vector1([aa_ == mutant_aa for aa_ in range(1, 21)])
            packer_task_design.nonconst_residue_task(pose_resi).restrict_absent_canonical_aas(aa_bool)

        packer = PackRotamersMover(self.scorefxn, packer_task_design)
        #G_before = self.scorefxn(pose)
        packer.apply(pose)
        G_after = self.scorefxn(pose)
        ddG  = G_after - G_before

        return {"ddG":     G_after-G_before,
                "G_before":G_before,
                "G_after": G_after}

    def mutate_task(self, pose, mutant_codes, chain_1="A", chain_2="B"):
        """
        function to introduce mutation on protein-protein interface
        :param pose:
        :param mutant_codes:
        :param chain_1:
        :param chain_2:
        :return:
        """
        prevent_in = pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT()
        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.PreventRepacking())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())
        chain_A = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(chain_1)
        chain_B = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(chain_2)
        interface_selector_1 = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(chain_A, 10, True)
        interface_selector_2 = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(chain_B, 10, True)
        ia = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(chain_A)
        ib = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(chain_B)
        ia = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(interface_selector_1, ia)
        ib = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(interface_selector_2, ib)
        interface_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(ia, ib)
        prevent_in = pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT()
        prevent_off = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
        not_interface = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(interface_selector)
        #packer_task_design = tf.create_task_and_apply_taskoperations(pose)
        design_resi = [f"{r}{chain}" for _,r,chain in mutant_codes]
        design_resi = rosetta.core.select.residue_selector.ResidueIndexSelector(",".join(design_resi))
        designable_interface = design_resi
        not_designable_interface = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(design_resi)
        not_designable_interface = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(interface_selector,
                                                                                                     not_designable_interface)
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_in,
                                                                                       not_designable_interface,
                                                                                       False))
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_off,
                                                                                       not_interface,
                                                                                       False))
        packer_task_design = tf.create_task_and_apply_taskoperations(pose)
        pdb2pose_resi = pyrosetta.rosetta.core.pose.get_pdb2pose_numbering_as_stdmap(pose)
        for (_,resi,chain), aa in mutant_codes.items():
            pose_resi = pdb2pose_resi[chain+str(resi)+"."]
            if aa == "*":
                continue
            aa = list(aa)
            mutant_aa = [int(aa_from_oneletter_code(aa_)) for aa_ in aa]
            #mutant_aa = int(aa_from_oneletter_code(aa))
            aa_bool = pyrosetta.Vector1([aa_ in mutant_aa for aa_ in range(1, 21)])
            packer_task_design.nonconst_residue_task(pose_resi).restrict_absent_canonical_aas(aa_bool)

        return packer_task_design


    def mutate_and_repack(self, pose, mutant_codes):
        m_task = self.mutate_task(pose, mutant_codes)
        packer = PackRotamersMover(self.scorefxn, m_task)
        packer.apply(pose)
        return self.calc_ddg(pose)

    def calc_ddg(self, pose):
        c = list(pyrosetta.rosetta.core.pose.chain_end_res(pose))
        p1 = pyrosetta.rosetta.protocols.grafting.return_region(pose, 1,c[0])
        p2 = pyrosetta.rosetta.protocols.grafting.return_region(pose, c[0]+1,c[-1])
        print(self.scorefxn(pose), self.scorefxn(p1), self.scorefxn(p2))
        ddg = self.scorefxn(pose) - self.scorefxn(p1) - self.scorefxn(p2)
        return ddg


    def calculate_energy_features(self, pose):
        """
        Calculate pairwise residue-residue energies and per residue energy values
        energy["fa_atr_pairwise"] - attractive pairwise energies
        energy["fa_rep_pairwise"] - repulsive  pairwise energies
        energy["fa_sol_pairwise"] - solubility pairwise energies
        energy[i,i] == 0
        energy["residue_energy"]  - total energy per residue
        """
        sfxn = self.scorefxn
        sfxn.score(pose)
        pdb2pose_resi = pyrosetta.rosetta.core.pose.get_pdb2pose_numbering_as_stdmap(pose)
        #print(pdb2pose_resi)
        pdb_residues = list(pdb2pose_resi)
        pos_residues = [pdb2pose_resi[r] for r in pdb_residues]
        #print(pos_residues)
        #print(residues)
        #exit(0)

        #residues = self.get_pdb_residues()


        seq_length = len(pdb_residues)
        energy = {}
        energy_terms = ['fa_atr',
                        'fa_rep',
                        'fa_sol',
                        #'fa_intra_rep',
                        #'fa_intra_sol_xover4',
                        'lk_ball_wtd',
                        'fa_elec']
                        #'pro_close',
                        #'hbond_sr_bb',
                        #'hbond_lr_bb',
                        #'hbond_bb_sc',
                        #'hbond_sc',
                        #'dslf_fa13',
                        #'omega',
                        #'fa_dun',
                        #'p_aa_pp',
                        #'yhh_planarity',
                        #'ref',
                        #'rama_prepro']

        energy_terms_dict = {k: i for i, k in enumerate(energy_terms)}
        energy_pairwise = np.zeros((seq_length, seq_length, len(energy_terms)))
        # energy_per_residue       = np.zeros((seq_length))
        energy["energy_single"] = np.zeros((seq_length))
        energy["energy_pair"] = energy_pairwise
        for i in range(seq_length):
            energy["energy_single"][i] = pose.energies().residue_total_energy(pos_residues[i])

        #print(energy)
        #exit(0)
        ids = list(range(seq_length))
        for i_1, i_2 in itertools.combinations(ids, 2):
            rosetta_resi_1 = pos_residues[i_1]
            rosetta_resi_2 = pos_residues[i_2]

            emap_ = pyrosetta.rosetta.core.scoring.EMapVector()
            sfxn.eval_ci_2b(pose.residue(rosetta_resi_1),
                            pose.residue(rosetta_resi_2),
                            pose,
                            emap_)

            values = self.decompose_emap(emap_)
            for energy_term, v in values.items():
                eid = energy_terms_dict[energy_term]
                energy["energy_pair"][i_1, i_2, eid] = v
                energy["energy_pair"][i_2, i_1, eid] = v

        e = energy["energy_pair"]
        residues_numbering = {"pdb_numbering":pdb_residues,
                              "rosetta_numbering":pos_residues}

        return residues_numbering, energy


    def decompose_emap(self, emap_):
        types = [pyrosetta.rosetta.core.scoring.ScoreType.fa_atr,
                 pyrosetta.rosetta.core.scoring.ScoreType.fa_rep,
                 pyrosetta.rosetta.core.scoring.ScoreType.fa_sol,
                 #pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_rep,
                 #pyrosetta.rosetta.core.scoring.ScoreType.fa_intra_sol_xover4,
                 pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_wtd,
                 pyrosetta.rosetta.core.scoring.ScoreType.fa_elec]
                 #pyrosetta.rosetta.core.scoring.ScoreType.pro_close,
                 #pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb,
                 #pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb,
                 #pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc,
                 #pyrosetta.rosetta.core.scoring.ScoreType.dslf_fa13,
                 #pyrosetta.rosetta.core.scoring.ScoreType.omega,
                 #pyrosetta.rosetta.core.scoring.ScoreType.fa_dun,
                 #pyrosetta.rosetta.core.scoring.ScoreType.p_aa_pp,
                 #pyrosetta.rosetta.core.scoring.ScoreType.yhh_planarity,
                 #pyrosetta.rosetta.core.scoring.ScoreType.ref,
                 #pyrosetta.rosetta.core.scoring.ScoreType.rama_prepro]

        names = ["fa_atr",
                 "fa_rep",
                 "fa_sol",
                 #"fa_intra_rep",
                 #"fa_intra_sol_xover4",
                 "lk_ball_wtd",
                 "fa_elec"]
                 #"pro_close",
                 #"hbond_sr_bb",
                 #"hbond_lr_bb",
                 #"hbond_bb_sc",
                 #"hbond_sc",
                 ##"dslf_fa13",
                 #"omega",
                 #"fa_dun",
                 #"p_aa_pp",
                 #"yhh_planarity",
                 #"ref",
                 #"rama_prepro"]

        values = {}
        for n, t in zip(names, types):
            values[n] = emap_[t]
        return values

    def calculate_emap(self, pose):

        return ddg

    def calculate_residue_position_score(self, pose):
        """
        This function conducts screening of all amino acid substitutions within protein
        and calculate if there are stabilizing/destabilizing mutants
        """
        pdb2pose_resi = pyrosetta.rosetta.core.pose.get_pdb2pose_numbering_as_stdmap(pose)
        print(pdb2pose_resi)
        #pdb_residues = list(pdb2pose_resi)
        #pos_residues = [pdb2pose_resi[r] for r in pdb_residues]


def screen_mutants_for_protein(pose, rr):
    import random
    pdb2pose_resi = pyrosetta.rosetta.core.pose.get_pdb2pose_numbering_as_stdmap(pose)
    pdb_residues = list(pdb2pose_resi)
    ddG_data = []
    random.shuffle(pdb_residues)
    for p in pdb_residues[:25]:
        #print(p)
        #print(p)
        chain = p[0]
        resi = p[1:-1]
        for aa in "QERTIPASDFGHKLCVNM":
            pose_ = pose.clone()
            aa3 = pose.residue(pdb2pose_resi[p]).name()[:3]
            d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                 'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
            ddG = rr.mutate_residue(pose_, (aa,int(resi),chain))
            ddG["resi"] = resi
            ddG["chain"] = chain
            ddG["aa_after"] = aa
            ddG["aa_before"] = d[aa3]
            ddG_data.append(ddG)
    #    break
    return ddG_data
    #len(ddG_data)
if __name__ == "__main__":
    rr = RosettaWrapper()
    #rr.calculate_residue_position_score(pose)
    #numbering, energy = rr.calculate_energy_features(pose)
    #print(energy)

    pose = rr.load_pose("../tests/test_dimer_repacked.pdb")
    mutants_data = screen_mutants_for_protein(pose,
                            rr)
    print(len(mutants_data))

    exit(0)
    pose = rr.repack_pose("../tests/test_dimer.pdb",
                   "../tests/test_dimer_repacked.pdb")
    ddg  = rr.calc_ddg(pose)

    seq_hard = "SYFSIATKWW"
    residues_hard = {("S",35,"B"):"*",
                     ("Y",37,"B"):"*",
                     ("F",47,"B"):"*",
                     ("S",49,"B"):"*",
                     ("I",50,"B"):"*",
                     ("A",58,"B"):"*",
                     ("T",60,"B"):"*",
                     ("K",96,"B"):"*",
                     ("W",98,"B"):"*",
                     ("W",111,"B"):"*"}

    residues_easy = {("S",35,"B"):"*",
                     ("Y",37,"B"):"*",
                     ("K",96,"B"):"*",
                     ("W",98,"B"):"*",
                     ("W",111,"B"):"*"}

    ddg_easy = rr.mutate_and_repack(pose, residues_easy)
    print(ddg_easy)

    #ddg_easy = rr.mutate_and_repack(pose, residues_easy)
    #dG_hard = optimizer.mutate(seq_hard, residues_hard.split(","))
    #print(dG_hard)

    seq_light = "SYKWW"
    residues_light = "B35,B37,B96,B98,B111"
    dG_light = optimizer.mutate(seq_hard, residues_light.split(","))
    print(dG_light)


#   print(ddg)
# def mutate_rbd(pose, scorefxn):
#     resi = [35,37,96,98,111]
#     #aa   = "TRS"
#     aa = "SYKWW"
#     scorefxn(pose)
#     c = list(pyrosetta.rosetta.core.pose.chain_end_res(pose))
#     mutant_codes = {("Y",  37,  'B'): "R",
#                     ("K",  96,  'B'): "D",
#                     ("W", 111,  'B'): "W",
#                     ("W",  98,  'B'): "K",
#                     ("S",  35,  'B'): "C",
#                     ("L",  452, 'A'): "R"}
#     ddg_1 = mutate_and_repack(pose,
#                         mutant_codes,
#                         fa_sfxn = scorefxn)
#
#     #pose_dimer.dump_pdb("mutant.pdb")
#     print(ddg_1)
#     return
#     aa   = "RLR"#]"R" for _ in resi]
#     ddg_2 = mutate_and_repack(pose,
#                         resi = resi,#[334,335,336],#346,452,490],#3,6,10,14,17,30,33,37,40,41],
#                         chain = 'A',
#                         aa = aa,#"TRS",#RRRRRRRRRR",
#                         fa_sfxn = scorefxn)
#
#     print(ddg_1)
#     print(ddg_2)

