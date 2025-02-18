#!/bin/bash

FASTA_in=$1

cp $FASTA_in reformat.in
perl fasta_seq_reformat_export.pl > run_log

cp $FASTA_in $FASTA_in"_ORIGINAL"
mv reformat_out $FASTA_in 

cp $FASTA_in "composition.in";

perl seq_compositions_perc_pipeline_export.pl >> run_log

mv composition_all.out seq_composition.txt

perl server_prediction_seq_export.pl >> run_log

cp $FASTA_in seq_props.in

perl seq_props_ALL_export.pl >> run_log

mv seq_prediction.txt seq_prediction_OLD.txt
perl profiles_gather_export.pl > run_log

rm bins.txt reformat.in seq_props.in seq_props.out STYprops.out composition.in seq_prediction_OLD.txt

mv run_log run.log
