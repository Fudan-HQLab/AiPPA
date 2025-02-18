
October 2017

Sequence-based prediction code used in University of Manchester protein-sol server.
Available from download tab at www.protein-sol.manchester.ac.uk.
see Hebdtich et al (2017) Bioinformatics 33:3098-3100.

Jim Warwicker and Mex Hebditch, Manchester

*****************
   Copyright (C) 2017 Jim Warwicker and Max Hebdtich

    These programs are free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    These programs are distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

The code is available 'as is', we are planning further developments to the server
and code, and here is just a snapshot that should allow interested users to make
calculations with multiple fasta sequences (as opposed to the single sequence
operation of the web server).
*****************

CODE is in various perl scripts (.pl)

fasta_seq_reformat_export.pl
seq_compositions_perc_pipeline_export.pl
server_prediction_seq_export.pl
seq_props_ALL_export.pl
profiles_gather_export.pl

RUN is initiated with

'multiple_prediction_wrapper_export.sh sequence_input_file'

sequence_input_file has something like a fasta format, it will convert to
paired records of ID and sequence for each entry:

>blah11
MVKVYAPASSANMSVGFDVLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDKLPSEPRENIVYQCWERFCQE

>blah22
MVKVYAPASSANMSVGFDVLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDKLPSEPRENIVYQCWERFCQE

but regular fasta should be OK (fasta_seq_reformat_export to convert).

OTHER INPUT (DATA) FILES:

ss_propensities.txt		(sec struc propensities)
seq_reference_data_NIWA.txt	(fitting to experimental solubility data)

RUNNING, as set up, will occur with all files in the local directort/

OUTPUT

seq_prediction.txt (CSV) contains the data relating to that provided on the server:

LEGEND records - brief information on features used for the predictions

HEADERS records - matched with keywords for the data output SEQUENCE records:

HEADERS PREDICTIONS with SEQUENCE PREDICTIONS
percent-sol, scaled-sol, population-sol, pI

HEADERS FEATURES ORIGINAL gives definitions of the features
HEADERS FEATURES PLOT give short-hand names for the features
both of these HEADERS lines are matched columns with SEQUENCE WEIGHTS and SEQUENCE DEVIATIONS
lines, where:
SEQUENCE WEIGHTS - weights from the fit to experimental data, only 10 features are non-zero
SEQUENCE DEVIATIONS - these are the z-score (see publication) deviation for each feature

then follows PROFILE data across successive 21 amino acid windows for:
Kyte-Doolittle
Uversky fold value	- plotted on server
sequence entropy
windowed net charge	- plotted on server




