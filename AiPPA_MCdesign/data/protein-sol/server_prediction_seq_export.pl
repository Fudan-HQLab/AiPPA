#!/usr/bin/perl -w

$file_here		= 'blah.txt';
open (BLAH, ">>$file_here");			# file for comments used throughout server code

# read in the reference data for this dataset.

$file_here		= 'seq_reference_data.txt';;
open (REF, "<$file_here") or die "cannot open seq_reference_data.txt\n";
@ref	= <REF>;
close (REF);

foreach $line_spaces (@ref) {
  chomp $line_spaces;
  $line					= $line_spaces;
  $line					=~ s/\s//g;
  @words				= split (",",$line);
  $nwords				= @words;
  if (exists $words[0]) {			# non-null line
    if (substr ($line,0,1) ne "#") {		# not comment line

      if ($words[2] eq 'AVG') {

	if ($words[0] eq 'POP') {
	  $population_avg	= $words[3];
	}

	if ($nwords != 10) {			# check the number, but not format, of expected fields
	  printf BLAH "stopping *** number of fields in AVG line mismatch\n";
	  exit;
	} else {
	  $AVG_line{$words[0]}		= 'seen';
	  if ($words[0] eq 'TOP') { $top_prop_avg	= $words[3]; }
	  if ($words[0] eq 'LOW') { $low_prop_avg	= $words[3]; }
	}

      } elsif ($words[3] eq 'headers') {
	if ($nwords != 40) {			# check the number, but not format, of expected fields
	  printf BLAH "stopping *** number of fields in headers line mismatch\n";
	  exit;
	} else {
	  $headers_line{$words[0]}	= 'seen';
	}
	if ($words[0] eq 'POP') {
	  $headers_store			= $line;	# store the headers just once, for results write
	}

      } elsif ($words[3] eq 'average') {
	if ($nwords != 40) {			# check the number, but not format, of expected fields
	  printf BLAH "stopping *** number of fields in average line mismatch\n";
	  exit;
	} else {
	  $average_line{$words[0]}	= 'seen';
	}
        if ($words[0] eq 'TOP') {		# store top and low averages to derive the linear fits (per feature)
	  $top_avg_line			= $line;
	} elsif ($words[0] eq 'LOW') {
	  $low_avg_line			= $line;
	} elsif ($words[0] eq 'POP') {
	  $pop_avg_line			= $line;
	}

      } elsif ($words[3] eq 'std-dev') {
	if ($nwords != 40) {			# check the number, but not format, of expected fields
	  printf BLAH "stopping *** number of fields in std-dev line mismatch\n";
	  exit;
	} else {
	  $std_dev_line{$words[0]}	= 'seen';
	}
        if ($words[0] eq 'POP') {		# store population std deviations for features
	  $pop_dev_line			= $line;
#unused	} elsif ($words[0] eq 'TOP') {
#	  $top_dev_line			= $line;
#	} elsif ($words[0] eq 'LOW') {
#	  $low_dev_line			= $line;
	}

      } elsif (($words[0] eq 'ZDF') and ($words[3] eq 'zscore-diff')) {
	if ($nwords != 40) {			# check the number, but not format, of expected fields
	  printf BLAH "stopping *** number of fields in ZDF zscore-diff line mismatch\n";
	  exit;
	} else {
	  $zdiff_line			= $line;
	  $zdf_values			= 'seen';
	}
      } elsif (($words[0] eq 'ZDF') and ($words[3] eq 'use_for_prob')) {
	if ($nwords != 40) {			# check the number, but not format, of expected fields
	  printf BLAH "stopping *** number of fields in ZDF use_for_prob line mismatch\n";
	  exit;
	} else {
	  $use_for_prob_line		= $line;
	  $zdf_use			= 'seen'
	}
      }

    }						# end not comment line
  }						# end not null line
}						# end line read

for $key_here ('POP', 'TOP', 'LOW' ) {
 if (($headers_line{$key_here} ne 'seen') or ($average_line{$key_here} ne 'seen') or ($std_dev_line{$key_here} ne 'seen')) {
   printf BLAH "stopping **** headers/average/std_dev line unseen for $key_here\n";
   exit;
 }
 if ($AVG_line{$key_here} ne 'seen') {
   printf BLAH "stopping **** AVG line unseen for $key_here\n";
   exit;
 }
}
if (($zdf_values ne 'seen') or ($zdf_use ne 'seen')) {
  printf BLAH "stopping **** ZDF zscore-diff/use_for_prob line unseen\n";
}

# store the reference values needed for prediction

$low_prop	= $low_prop_avg;
$top_prop	= $top_prop_avg;
@lows		= split (",",$low_avg_line);		# values run from 4-39 (inclusive), with array index starting at 0
@tops		= split (",",$top_avg_line);		# as above
@pops		= split (",",$pop_avg_line);		# as above
@pop_devs	= split (",",$pop_dev_line);		# as above
@zdiffs		= split (",",$zdiff_line);		# as above, zdiff_scores remain signed at this point
@use		= split (",",$use_for_prob_line);	# as above, values are y,n
@heads		= split (",",$headers_store);		# as above

# read and store data values for this sequence set

$file_here		= 'seq_composition.txt';
open (COMP, "<$file_here") or die "cannot open seq_composition.txt\n";
$headers_found		= "no";
$nseqs			= 0;
while ($line_spaces = <COMP>) {
  chomp $line_spaces;
  $line					= $line_spaces;
  $line					=~ s/\s//g;
  @words				= split (",",$line);
  $nwords				= @words;
  if ($headers_found eq "no") {
    if ((exists $words[0]) and (exists $words[1])) {
      if (($words[0] eq "WHOLE-SEQ") and ($words[1] eq "ORF-ID")) {
        $headers_found	= "yes";
#        @heads_comp	= @words;		# not used currently
      }
    }
  }
  if (exists $words[0]) {
    if ($words[0] eq "WHOLE-SEQ") {		# only process WHOLE-SEQ here - will need other of our code for seq-profiles
      if ($words[1] ne "ORF-ID") {		# check not on header line
        if ($nwords != 38) {			# flag, ID, 36 comp etc data fields
	  printf BLAH "stopping **** fields ne 38 in composition file data line\n";
	  exit;
	} else {
	  $nseqs++;
	  $ids[$nseqs]			= $words[1];
	  for ($n=1; $n<=36; $n++) { $data[$nseqs][$n]	= $words[$n+1]; }
	}
      }
    }
  }
}
close (COMP);

$file_here		= 'seq_prediction.txt';
open (PREDA, ">$file_here") or die "cannot open seq_prediction.txt\n";

# before the sequence data writes, output things that don't change

$wt_sum			= 0;
for ($d=1; $d<=36; $d++) {
  if ($use[$d+3] eq 'y') {
    $wt_feature[$d]	= abs ($zdiffs[$d+3]);		# get feature weighting for prediction (mostly zero probably)
    $wt_sum		+= $wt_feature[$d];
  } else {
    $wt_feature[$d]	= 0;
  }
}

printf PREDA "LEGEND 35 sequence features are calculated, including 20 amino acid compositions\n";
printf PREDA "LEGEND All features are calculated over a sliding 21 amino acid window\n";
printf PREDA "LEGEND   KmR=KminusR DmE=DminusE KpR=KplusR PmN=K+R-D-E PpN=K+R+D+N aro=F+W+Y\n";
printf PREDA "LEGEND   fld = border of folded (pos) / unfolded (neg) Uversky et al Proteins 2000 41:415\n";
printf PREDA "LEGEND   dis = disorder propensity Rune & Linding NAR 2003 31:3701\n";
printf PREDA "LEGEND   bet = beta strand propensities Costantini et al 2006 BBRC 342:441\n";
printf PREDA "LEGEND   mem = Kyte-Doolittle hydropathy normalised (-1 to +1) JMB 157:105\n";
printf PREDA "LEGEND Only a subset of features used for prediction, according to best fit against data\n";
printf PREDA "LEGEND   underlying solubility data: cell-free expression Niwa et al PNAS 2009 106:4201\n";
printf PREDA "LEGEND Features used for prediction are scaled 0 - 1 over range in underlying dataset\n";
printf PREDA "HEADERS PREDICTIONS LINE,ID,percent-sol,scaled-sol,population-sol,pI\n";
printf PREDA "HEADERS FEATURES ORIGINAL,ID";
for ($d=1; $d<=3; $d++) { printf PREDA ",$heads[$d+3]"; }
for ($d=5; $d<=36; $d++) { printf PREDA ",$heads[$d+3]"; }
printf PREDA "\n";
printf PREDA "HEADERS FEATURES PLOT,ID,";
printf PREDA "KmR,DmE,len,";
for ($d=5; $d<=24; $d++) { printf PREDA "$heads[$d+3],"; }
printf PREDA "KpR,DpE,PmN,PpN,aro,pI,mem,chr,fld,dis,ent,bet\n";
printf PREDA "\n";

for ($n=1; $n<=$nseqs; $n++) {
  $wt_sum		= 0;
  $pred_sum		= 0;
  for ($d=1; $d<=36; $d++) {

    if ($d != 4) {				# get the values of (X-Xavg-pop) / std-dev-pop i.e. a neg/pos scale
      $norm_dev[$d]	= ($data[$n][$d] - $pops[$d+3]) / $pop_devs[$d+3];
    }

    $wt_feature[$d]	= 0;			# initialise to zero, since will only overwrite if feature used for prediction
    if ($use[$d+3] eq 'y') {			# we are using this feature (based on correlation analysis)

      $data_here	= $data[$n][$d];
      if ($lows[$d+3] <= $tops[$d+3]) {		# range as the words say, bounds check 
        if ($data[$n][$d] < $lows[$d+3]) { $data_here	= $lows[$d+3]; }
	if ($data[$n][$d] > $tops[$d+3]) { $data_here	= $tops[$d+3]; }
      } else {					# inverted range, bounds check the other way around
        if ($data[$n][$d] > $lows[$d+3]) { $data_here	= $lows[$d+3]; }
	if ($data[$n][$d] < $tops[$d+3]) { $data_here	= $tops[$d+3]; }
      }

      $pred_here	= $low_prop + ($top_prop - $low_prop) * ( ($data_here - $lows[$d+3]) / ($tops[$d+3] - $lows[$d+3]) );
      $wt_here		= abs ($zdiffs[$d+3]);
      $pred_sum		+= $wt_here*$pred_here;
      $wt_sum		+= $wt_here;
      $wt_feature[$d]	= $wt_here;

    }
  }
  $prediction		= $pred_sum / $wt_sum;
  $prediction_scaled	= ($prediction - $low_prop) / ($top_prop - $low_prop);
  $population_scaled	= ($population_avg - $low_prop) / ($top_prop - $low_prop);
  $pI_here		= $data[$n][30];
  printf PREDA "SEQUENCE PREDICTIONS,$ids[$n]";			# output sequence-specific data
  printf PREDA ",%6.3f,%6.3f,%6.3f,%6.3f\n", $prediction,$prediction_scaled,$population_scaled,$pI_here;
  printf PREDA "SEQUENCE WEIGHTS,$ids[$n]";
  for ($d=1; $d<=3; $d++) {
    $wt_print		= $wt_feature[$d]/$wt_sum;
    printf PREDA ",%6.3f", $wt_print;
  }
  for ($d=5; $d<=36; $d++) {
    $wt_print		= $wt_feature[$d]/$wt_sum;
    printf PREDA ",%6.3f", $wt_print;
  }
  printf PREDA "\n";						# output the devs/std-dev for each feature, for this sequence
  printf PREDA "SEQUENCE DEVIATIONS,$ids[$n]";
  for ($d=1; $d<=3; $d++) { printf PREDA ",%6.3f", $norm_dev[$d]; }
  for ($d=5; $d<=36; $d++) { printf PREDA ",%6.3f", $norm_dev[$d]; }
  printf PREDA "\n\n";

}

close (PREDA);
close (BLAH);

exit;

