pdb_antibody_preprocess.py - scripts for preprocessing PDB antigen/antibody structures for run of ESM-IF1 and AlphaFold.
Script takes multichain PDB structure as input and outputs single chain and renumbered from 1 PDB structure in the
dataframe format.

Input - Path to multichain antigen/fab structure & mask

Format of mask of residues to optimize:
[[residue_number_start,residue_number_end, chain_id], ... ]
Example:
[[100,110, "H"],
[50,60],  "L"]]

Output:
PDBID_preprocessed.pkl - biopandas.pdb dataframe file. Dataframe comprise following columns:
['record_name', 'atom_number', 'blank_1', 'atom_name', 'alt_loc', 'residue_name', 'blank_2', 'chain_id', 'residue_number', 'insertion', 'blank_3', 'x_coord', 'y_coord', 'z_coord', 'occupancy', 'b_factor', 'blank_4', 'segment_id', 'element_symbol', 'charge', 'line_idx', 'resi_key', 'residue_number_insertion_fix', 'mask']

residue_number -> renumbered sequence from 1 to N
resi_key -> contain tuples unique to residue with old originial PDB numbering and chain ids.
mask -> mask :)

PDBID_preprocessed.pdb - same but in PDB format. Can be used as input for ESM-IF1...

PDBID_sequence.fasta - amino acid sequence. X - denotes missing residues and gaps, in particular between chains.
