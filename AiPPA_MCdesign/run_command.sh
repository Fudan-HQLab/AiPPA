source activate IgFold
python runfile.py \
	--target 0_target/TL1A.pdb \
	--nb 0_target/vhh1.pdb \
	--seq 0_target/vhh1.fasta \
	--step 30000 \
	--terminal -20 \
	--T_factor 0.2 \
	--seed 16 \
	--mut_point 1 \
	--mut_fr false \
	--sol_thresh 0.45

# >&T02_seed16_mut_1.log&
