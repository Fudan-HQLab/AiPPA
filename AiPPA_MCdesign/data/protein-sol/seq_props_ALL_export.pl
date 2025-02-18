#!/usr/bin/perl -w


$file_here		= 'blah.txt';
open (BLAH, ">>$file_here");	# file for comments, throughout server code

$winlenP	= 21;		# window for calculation of all props except searching for charge border
$winlenP2	= 10;		# = (winlenP-1)/2
if ((2*$winlenP2+1) != $winlenP) {
	printf BLAH "window or half window wrong\n";
	exit;
}

# winlenB/2 used to search the charge border props around STY sites
$winlenB	=  5;		#
$winlenB2	=  2;		# 
if ((2*$winlenB2+1) != $winlenB) {
	printf BLAH "window or half window wrong\n";
	exit;
}

$nbins_fi	= 30;
$nbins_dis	= $nbins_fi;

# env variable option to neutralise around STY in the charge border analysis ie test importance of Plation motif

if (exists $ENV{motifneutral}) {
  $bord_neutralise	= $ENV{motifneutral};
} else {
  $bord_neutralise	= "no";
}
# $len_neutral		= 5;	# not used currently
$len_neutral2		= 2;

# env variable to plot vertical lists of properties for each seq, for subsequent data handling/plotting

if (exists $ENV{extra_output}) {
  $extra		= $ENV{extra_output};
} else {
  $extra		= "no";
}

if ((exists $ENV{his_charged}) and ($ENV{his_charged} eq 'yes')) {
  printf BLAH "\nHIS will be set at charge +1\n\n";
} else {
  printf BLAH "\nHIS will be set at charge 0\n\n";
}
printf BLAH "column o/p file for each sequence = $extra - set with extra_output env variable\n\n";

# List the amino acids, for seq entropy (will convert to uc here at least)

@aa_single	= ("A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y");

$dis {'A'}	= -0.26154;
$dis {'C'}	= -0.01515;
$dis {'D'}	=  0.22763;
$dis {'E'}	= -0.20469;
$dis {'F'}	= -0.22557;
$dis {'G'}	=  0.43323;
$dis {'H'}	= -0.00122;
$dis {'I'}	= -0.42224;
$dis {'K'}	= -0.10009;
$dis {'L'}	= -0.33793;
$dis {'M'}	= -0.22590;
$dis {'N'}	=  0.22989;
$dis {'P'}	=  0.55232;
$dis {'Q'}	= -0.18768;
$dis {'R'}	= -0.17659;
$dis {'S'}	=  0.14288;
$dis {'T'}	=  0.00888;
$dis {'V'}	= -0.38618;
$dis {'W'}	= -0.24338;
$dis {'Y'}	= -0.20751;
$dis {'X'}	=  0.0;

$chr {'A'}	=  0.0;
$chr {'C'}	=  0.0;
$chr {'D'}	= -1.0;
$chr {'E'}	= -1.0;
$chr {'F'}	=  0.0;
$chr {'G'}	=  0.0;
$chr {'H'}	=  0.0;
$chr {'I'}	=  0.0;
$chr {'K'}	=  1.0;
$chr {'L'}	=  0.0;
$chr {'M'}	=  0.0;
$chr {'N'}	=  0.0;
$chr {'P'}	=  0.0;
$chr {'Q'}	=  0.0;
$chr {'R'}	=  1.0;
$chr {'S'}	=  0.0;
$chr {'T'}	=  0.0;
$chr {'V'}	=  0.0;
$chr {'W'}	=  0.0;
$chr {'Y'}	=  0.0;
$chr {'X'}	=  0.0;

$dis {'a'}	= -0.26154;
$dis {'c'}	= -0.01515;
$dis {'d'}	=  0.22763;
$dis {'e'}	= -0.20469;
$dis {'f'}	= -0.22557;
$dis {'g'}	=  0.43323;
$dis {'h'}	= -0.00122;
$dis {'i'}	= -0.42224;
$dis {'k'}	= -0.10009;
$dis {'l'}	= -0.33793;
$dis {'m'}	= -0.22590;
$dis {'n'}	=  0.22989;
$dis {'p'}	=  0.55232;
$dis {'q'}	= -0.18768;
$dis {'r'}	= -0.17659;
$dis {'s'}	=  0.14288;
$dis {'t'}	=  0.00888;
$dis {'v'}	= -0.38618;
$dis {'w'}	= -0.24338;
$dis {'y'}	= -0.20751;
$dis {'x'}	=  0.0;

$chr {'a'}	=  0.0;
$chr {'c'}	=  0.0;
$chr {'d'}	= -1.0;
$chr {'e'}	= -1.0;
$chr {'f'}	=  0.0;
$chr {'g'}	=  0.0;
$chr {'h'}	=  0.0;
$chr {'i'}	=  0.0;
$chr {'k'}	=  1.0;
$chr {'l'}	=  0.0;
$chr {'m'}	=  0.0;
$chr {'n'}	=  0.0;
$chr {'p'}	=  0.0;
$chr {'q'}	=  0.0;
$chr {'r'}	=  1.0;
$chr {'s'}	=  0.0;
$chr {'t'}	=  0.0;
$chr {'v'}	=  0.0;
$chr {'w'}	=  0.0;
$chr {'y'}	=  0.0;
$chr {'x'}	=  0.0;

if ((exists $ENV{his_charged}) and ($ENV{his_charged} eq 'yes')) {
  $chr {'H'}	=  1.0;
  $chr {'h'}	=  1.0;
}

$KD {'A'}	=  1.8;
$KD {'C'}	=  2.5;
$KD {'D'}	= -3.5;
$KD {'E'}	= -3.5;
$KD {'F'}	=  2.8;
$KD {'G'}	= -0.4;
$KD {'H'}	= -3.2;
$KD {'I'}	=  4.5;
$KD {'K'}	= -3.9;
$KD {'L'}	=  3.8;
$KD {'M'}	=  1.9;
$KD {'N'}	= -3.5;
$KD {'P'}	= -1.6;
$KD {'Q'}	= -3.5;
$KD {'R'}	= -4.5;
$KD {'S'}	= -0.8;
$KD {'T'}	= -0.7;
$KD {'V'}	=  4.2;
$KD {'W'}	= -0.9;
$KD {'Y'}	= -1.3;
$KD {'X'}	=  0.0;

$KD {'a'}	=  1.8;
$KD {'c'}	=  2.5;
$KD {'d'}	= -3.5;
$KD {'e'}	= -3.5;
$KD {'f'}	=  2.8;
$KD {'g'}	= -0.4;
$KD {'h'}	= -3.2;
$KD {'i'}	=  4.5;
$KD {'k'}	= -3.9;
$KD {'l'}	=  3.8;
$KD {'m'}	=  1.9;
$KD {'n'}	= -3.5;
$KD {'p'}	= -1.6;
$KD {'q'}	= -3.5;
$KD {'r'}	= -4.5;
$KD {'s'}	= -0.8;
$KD {'t'}	= -0.7;
$KD {'v'}	=  4.2;
$KD {'w'}	= -0.9;
$KD {'y'}	= -1.3;
$KD {'x'}	=  0.0;

# Normalise the KD hydropathy values.

@KD_keys	= keys %KD;
foreach $keyhere (@KD_keys) {
  $KDtmp		= $KD{$keyhere} + 4.5;
  $KDnorm{$keyhere}	= $KDtmp / 9.0;
  printf BLAH "KDnorm for $keyhere is $KDnorm{$keyhere}\n";
}

# get the sequences

$file_here		= 'seq_props.in';
open (SEQS_IN, "$file_here") or die "cannot open seq_props.in\n";
my $nseq		= 0;
while ($line 		= <SEQS_IN>) {
	chomp $line;
	my @words 	= split ("",$line);
	if (exists $words[0]) {
	    if ($words[0] eq ">") {			# new protein
		$nseq++;
		$seqid[$nseq]	= $line;
		$seqtmp{$seqid[$nseq]}	= "";
	    }
	    else {
		if ($nseq == 0) {
			printf BLAH "did not see first seq id\n";
			exit;
		}
		$seqtmp{$seqid[$nseq]}	.= $line;
	    }
	}
}
$nseqs			= $nseq;
close (SEQS_IN);

for ($nseq=1; $nseq<=$nseqs; $nseq++) {		# now put * Plations into flags
	$seqhere	= $seqtmp{$seqid[$nseq]};
	$lentmp		= length $seqhere;
	@words		= split ("",$seqhere);
	$naa		= 0;
	for ($n=1; $n<=$lentmp; $n++) {
	  if ($words[$n-1] ne " ") {
	    if ($words[$n-1] eq "*") {
	      if ($naa == 0) {
	        printf BLAH "STOPPING - first seq char is a * - Plation\n";
		exit;
	      }
	      $phosupd[$naa]	= "P";
	    } else {
	      $naa++;
	      $sequpd[$naa]	= $words[$n-1];
	      $phosupd[$naa]	= " ";
	    }
	  }
	}
	$seqstr		= "";
	$phostr		= "";
	for ($n=1; $n<=$naa; $n++) {
	  $seqstr	.= $sequpd[$n];
	  $phostr	.= $phosupd[$n];
	}
	$seq{$seqid[$nseq]}	= $seqstr;
	$pho{$seqid[$nseq]}	= $phostr;
}

# calculate the P (by motif) and Order (by windowed propensity) values.
# also calculate the overall hydropathy and net charge values, per aa, output with overall seq info

$naasTOT		= 0;
$ndisorderTOT		= 0;

open (PSEQS, ">", "STYprops.out") or die "cannot open STYprops.out\n";
#printf PSEQS "P/unP   dis   netQ  fracQ  meanQ     KD  FldInd    ent  chk1  bord\n";

for ($h=1; $h<=$nbins_dis; $h++) {
  $n_phos[$h]		= 0;
  $n_unphos[$h]		= 0;
  $net_phos[$h]		= 0;
  $net_unphos[$h]	= 0;
  $frac_phos[$h]	= 0;
  $frac_unphos[$h]	= 0;
  $n_nf_phos[$h]	= 0;		# n_nf counters where we have complete windows around STY
  $n_nf_unphos[$h]	= 0;
  $nbord_phos[$h]	= 0;
  $nbord_unphos[$h]	= 0;
  $meanQ_phos[$h]	= 0;
  $meanQ_unphos[$h]	= 0;
  $KDnorm_phos[$h]	= 0;
  $KDnorm_unphos[$h]	= 0;
  $ent_phos[$h]		= 0;
  $ent_unphos[$h]	= 0;
}
for ($hfi=1; $hfi<=$nbins_fi; $hfi++) {
  $nfi_phos[$hfi]		= 0;
  $nfi_unphos[$hfi]		= 0;
  $nfi_nf_phos[$h]		= 0;
  $nfi_nf_unphos[$h]		= 0;
  $nbordfi_phos[$hfi]		= 0;
  $nbordfi_unphos[$hfi]		= 0;
  $entfi_phos[$hfi]		= 0;
  $entfi_unphos[$hfi]		= 0;

  $nFQ_nf_phos[$h]		= 0;
  $nFQ_nf_unphos[$h]		= 0;
  $nbordFQ_phos[$hfi]		= 0;
  $nbordFQ_unphos[$hfi]		= 0;
  $nMQ_nf_phos[$h]		= 0;
  $nMQ_nf_unphos[$h]		= 0;
  $nbordMQ_phos[$hfi]		= 0;
  $nbordMQ_unphos[$hfi]		= 0;
  $nKD_nf_phos[$h]		= 0;
  $nKD_nf_unphos[$h]		= 0;
  $nbordKD_phos[$hfi]		= 0;
  $nbordKD_unphos[$hfi]		= 0;
}
for ($nseq=1; $nseq<=$nseqs; $nseq++) {				# OPEN SEQUENCES LOOP
	printf PSEQS "STARTID $seqid[$nseq]\n";
	printf PSEQS "P/unP    dis    netQ   fracQ   meanQ      KD   FldInd     ent   chk1   bord meanQ_PM\n";

	$seqhere	= $seq{$seqid[$nseq]};
	$phohere	= $pho{$seqid[$nseq]};
	$seqlen[$nseq]	= length $seqhere;
	@words		= split ("",$seqhere);
	@wordsP		= split ("",$phohere);

	$propcharhere	= "";
	$signhere	= "";
#	$chrhere	= "";
	$chrhereA	= "";
	$chrhereB	= "";
	$ndisorder	= 0;

	$KDnorm_tot	= 0;
	$chr_tot	= 0;
	$dis_tot	= 0;
	$ndis_tot	= 0;
	for $key (@aa_single) {
		$aa_count{$key}	= 0;
	}

#	$motifcharhere	= "";
#	$skip		= 0;
#	$nmotifs	= 0;
	for ($naa=1; $naa<=$seqlen[$nseq]; $naa++) {		# OPEN AAs in SEQs LOOP
		$naasTOT++;

		if (exists $dis{$words[$naa-1]}) {
		  $dis_tot += $dis{$words[$naa-1]};
		  $ndis_tot++;
		}
#boxing - loop below added for debugging
	      if (exists $KDnorm{$words[$naa-1]}) {
		$KDnorm_tot	= $KDnorm_tot + $KDnorm{$words[$naa-1]};
		$chr_tot	= $chr_tot + $chr{$words[$naa-1]};
		$aa_lc		= $words[$naa-1];
		$AA_UC		= uc ($aa_lc);
		if (exists $aa_count{$AA_UC}) {
		  $aa_count{$AA_UC}++;
		} else {
#		  printf BLAH "STOPPING AA (UC) not known in entropy count\n";
		}

	      } else {
		printf BLAH " KDnorm PROBLEM naa words = $naa $words[$naa-1]\n";
	      }

		$propsum		= 0;
		$nprop			= 0;
		$wlow			= $naa - $winlenP2;
		$wtop			= $naa + $winlenP2;

		$chrsum			= 0;

		for ($nw=$wlow; $nw<=$wtop; $nw++){		# second the disorder
			if (($nw>=1) and ($nw<=$seqlen[$nseq])) {
			    if (exists $dis{$words[$nw-1]}) {
				$propsum += $dis{$words[$nw-1]};
				$nprop++;
			    }

			    if (exists $chr{$words[$nw-1]}) {
			        if ($bord_neutralise eq "yes") {
				  $ndist	= $naa - $nw;
				  $ndistabs	= abs($ndist);
				  if ($ndistabs > $winlenP2) {
				    printf BLAH "STOPPING ndist gt winlenP2\n";
				    exit;
				  }
				  if ($ndistabs > $len_neutral2) {
				    $chrsum += $chr{$words[$nw-1]};
				  }
				} else {
				  $chrsum += $chr{$words[$nw-1]};
				}
			    }

			}
		}
		if ($nprop != 0) {
			$prophere	= $propsum/$nprop;
			$windis[$naa]	= $prophere;
			if ($prophere > 0) {
				$propcharhere	.= "*";
				$ndisorder++;
				$ndisorderTOT++;
			} else {
				$propcharhere	.= " ";
			}
		} else {
			$propcharhere	.= " ";
		}

		if      ($chrsum < 0) {
		  $signinc		= "-";
		} elsif ($chrsum > 0) {
		  $signinc		= "+";
		} else {
		  $signinc		= " ";
		}
		$signhere		.= $signinc;

		$abschr			= abs($chrsum);
		if ($abschr > 9) {
		  $chrhereA		.= int($abschr/10);
		  $chrhereB		.= $abschr - 10*int($abschr/10);
		} else {
		  $chrhereA		.= " ";
		  $chrhereB		.= $abschr;
		}

		$all_winQ[$nseq][$naa]	= $chrsum;
	}							# CLOSE AAs in SEQs LOOP
#	$motifchar{$seqid[$nseq]}	= $motifcharhere;
	$propchar{$seqid[$nseq]}	= $propcharhere;
	$signchar{$seqid[$nseq]}	= $signhere;
	$chrcharA{$seqid[$nseq]}	= $chrhereA;
	$chrcharB{$seqid[$nseq]}	= $chrhereB;

	$KDnorm_seqs{$seqid[$nseq]}	= $KDnorm_tot / $seqlen[$nseq];
	$chrmean_seqs{$seqid[$nseq]}	= abs ($chr_tot) / $seqlen[$nseq];
	$foldind_seqs{$seqid[$nseq]}	= 2.785 * $KDnorm_seqs{$seqid[$nseq]} - $chrmean_seqs{$seqid[$nseq]} - 1.151;
	$dis_seqs{$seqid[$nseq]}	= $dis_tot / $ndis_tot;

	$chr_tot_seqs{$seqid[$nseq]}	= $chr_tot / $seqlen[$nseq];
	$ent_sum		= 0;
	$frac_sum		= 0;
	for $key (@aa_single) {
		if ($aa_count{$key} != 0) {
		  $frac_here		= $aa_count{$key} / $seqlen[$nseq];
		  $incr_here		= -$frac_here * (log($frac_here) / log(2.0));
		  $ent_sum		+= $incr_here;
		  $frac_sum		+= $frac_here
		}
	}
	$ent_seqs{$seqid[$nseq]}	= $ent_sum;
#	$frac_seqs{$seqid[$nseq]}	= $frac_sum;	# not used currently

	@wordsPROP			= split ("",$propcharhere);
	@wordsSIGN			= split ("",$signhere);
#	@wordsCHAR			= split ("",$chrhere);

# at the moment we're only processing S, T, Y phos and non-phos - of course non-phos is well-dodgy
	for ($h=1; $h<=$nbins_dis; $h++) {
	  $n_phos_here[$h]	= 0;
  	  $n_unphos_here[$h]	= 0;
  	  $net_phos_here[$h]	= 0;
  	  $net_unphos_here[$h]	= 0;
  	  $frac_phos_here[$h]	= 0;
  	  $frac_unphos_here[$h]	= 0;
	  $n_nf_phos_here[$h]	= 0;
	  $n_nf_unphos_here[$h]	= 0;
  	  $nbord_phos_here[$h]	= 0;
  	  $nbord_unphos_here[$h] = 0;
	  $meanQ_phos_here[$h]	= 0;
	  $meanQ_unphos_here[$h] = 0;
	  $KDnorm_phos_here[$h]	= 0;
	  $KDnorm_unphos_here[$h] = 0;
	  $ent_phos_here[$h]	= 0;
	  $ent_unphos_here[$h]	= 0;
	}
	for ($hfi=1; $hfi<=$nbins_fi; $hfi++) {
  	  $nfi_phos_here[$hfi]			= 0;
  	  $nfi_unphos_here[$hfi]		= 0;
	  $nfi_nf_phos_here[$hfi]		= 0;
	  $nfi_nf_unphos_here[$hfi]		= 0;
	  $nbordfi_phos_here[$hfi]		= 0;
	  $nbordfi_unphos_here[$hfi]		= 0;
	  $entfi_phos_here[$hfi]		= 0;
	  $entfi_unphos_here[$hfi]		= 0;

	  $nFQ_nf_phos_here[$hfi]		= 0;
	  $nFQ_nf_unphos_here[$hfi]		= 0;
	  $nbordFQ_phos_here[$hfi]		= 0;
	  $nbordFQ_unphos_here[$hfi]		= 0;
	  $nMQ_nf_phos_here[$hfi]		= 0;
	  $nMQ_nf_unphos_here[$hfi]		= 0;
	  $nbordMQ_phos_here[$hfi]		= 0;
	  $nbordMQ_unphos_here[$hfi]		= 0;
	  $nKD_nf_phos_here[$hfi]		= 0;
	  $nKD_nf_unphos_here[$hfi]		= 0;
	  $nbordKD_phos_here[$hfi]		= 0;
	  $nbordKD_unphos_here[$hfi]		= 0;
}
	for ($naa=1; $naa<=$seqlen[$nseq]; $naa++) {	# OPEN AAs in SEQs LOOP

# CHARGE SIGN CHANGES in winlenB ooooooooooooooooooooooooooooooo
	  $nb1			= $naa - $winlenB2;
	  $nb2			= $naa + $winlenB2;
	  if (($nb1 >= 1) and ($nb2 <= $seqlen[$nseq])) {
	    $pos	= 'no';
	    $neg	= 'no';
	    for ($nb=$nb1; $nb<=$nb2; $nb++) {
	      if      ($wordsSIGN[$nb-1] eq '+') {
	        $pos	= 'yes';
	      } elsif ($wordsSIGN[$nb-1] eq '-') {
	        $neg	= 'yes';
	      }
	    }
	    if (($pos eq 'yes') and ($neg eq 'yes')) {
	      $bord	= 'yes';
	    } else {
	      $bord	= 'no';
	    }
	  }
# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

# NET CHARGE AND CHARGE FRACTION in P WINDOW ooooooooooooooooooooo
	  $nb1			= $naa - $winlenP2;
	  $nb2			= $naa + $winlenP2;
	  if (($nb1 >= 1) and ($nb2 <= $seqlen[$nseq])) {
	    $winP	= 'yes';
	    $sumQ	= 0;
	    $netQ	= 0;
	    $KDnorm_tot	= 0;
	    $chr_tot	= 0;
	    for $key (@aa_single) {
	      $aa_count{$key}	= 0;
	    }
	    $subseq		= "";
	    for ($nb=$nb1; $nb<=$nb2; $nb++) {
	      $subseq	.= $words[$nb-1];
	      if (exists $chr{$words[$nb-1]}) {
		$netQ	+= $chr{$words[$nb-1]};
		$pm	= $chr{$words[$nb-1]};
		if (abs($pm) > 0.9) {
		  $sumQ++;
		}
	      }
# boxing
if (exists $KDnorm{$words[$nb-1]}) {
	      $KDnorm_tot	= $KDnorm_tot + $KDnorm{$words[$nb-1]};
} else {
  printf BLAH "KDnorm PROBLEM 2 nb-1 words[nb-1] = $nb-1 $words[$nb-1]\n ";
}
	      $aa_lc		= $words[$nb-1];
	      $AA_UC		= uc ($aa_lc);
	      if (exists $aa_count{$AA_UC}) {
		$aa_count{$AA_UC}++;
	      } else {
#		  printf BLAH "STOPPING AA (UC) not known in entropy count\n";
	      }

	    }
	    $fracQ	= $sumQ / $winlenP;
	    $meanQ_win		= abs ($netQ) / $winlenP;
	    $meanQ_winNET	=     ($netQ) / $winlenP;
	    $KDnorm_win	= $KDnorm_tot / $winlenP;
	    $ent_sum		= 0;
	    $frac_sum		= 0;
	    for $key (@aa_single) {
	      if ($aa_count{$key} != 0) {
	        $frac_here	= $aa_count{$key} / $winlenP;
	        $incr_here	= -$frac_here * (log($frac_here) / log(2.0));
		$ent_sum	+= $incr_here;
		$frac_sum	+= $frac_here
	      }
	    }
	    $ent_win	= $ent_sum;
	    $frac_win	= $frac_sum;

	  } else {
	    $winP	= 'no';
	  }
# oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

#	  $am		= $words[$naa-1];		# not used currently
#	  if (($am eq "S") or ($am eq "T") or ($am eq "Y") or ($am eq "s") or ($am eq "t") or ($am eq "y")) {
	  if ((exists $words[$naa-1]) and ($winP eq 'yes')) {
	    $prophere		= $wordsPROP[$naa-1];
	    $bin		= 1 + int (($windis[$naa]+0.5)*$nbins_dis);

	    if ($bin < 1) { $bin = 1; }				# cludge to avoid oub
	    if ($bin > $nbins_dis) { $bin = $nbins_dis; }	# cludge to avoid oub

	    if (($bin < 1) or ($bin > $nbins_dis)) {
	      printf BLAH "STOPPING - bin for disorder not in 1 to $nbins_dis range\n";
	      exit;
	    }
# bin Uversky fold from theoretical min, max = -2.151, 1.634 from the Prilusky paper
	    $foldind_here	= 2.785 * $KDnorm_win - $meanQ_win - 1.151;
	    $binfi		= 1 + int (($foldind_here + 2.151)/((1.634+2.151)/$nbins_fi));
	    if (($binfi < 1) or ($binfi > $nbins_fi)) {
	      printf BLAH "STOPPING - bin for Uversky fold not in 1 to $nbins_fi range\n";
	      exit;
	    }

# for bins of fracQ (FQ), meanQ (MQ), KD; assume each has max range of 0 to 1
	    $binFQ		= 1 + int (($fracQ)*$nbins_dis);
	    if ($binFQ == $nbins_dis+1) { $binFQ = $nbins_dis; }
	    if (($binFQ < 1) or ($binFQ > $nbins_dis)) {
	      printf BLAH "STOPPING - bin for fracQ not in 1 to $nbins_dis range; $binFQ $fracQ\n";
	      exit;
	    }
	    $binMQ		= 1 + int (($meanQ_win)*$nbins_dis);
	    if ($binMQ == $nbins_dis+1) { $binMQ = $nbins_dis; }
	    if (($binMQ < 1) or ($binMQ > $nbins_dis)) {
	      printf BLAH "STOPPING - bin for meanQ not in 1 to $nbins_dis range\n";
	      exit;
	    }
	    $binKD		= 1 + int (($KDnorm_win)*$nbins_dis);
	    if (($binKD < 1) or ($binKD > $nbins_dis)) {
	      printf BLAH "STOPPING - bin = $binKD for KD not in 1 to $nbins_dis range, KDnorm_win=$KDnorm_win\n";
	      exit;
	    }

	    if ($wordsP[$naa-1] eq "P") {
	      $n_phos_here[$bin]++;
	      $nfi_phos_here[$binfi]++;
	      if ($winP eq 'yes') {
	        if ($bord eq 'yes') {		# only increment bord bins where winp='yes'
	          $nbord_phos_here[$bin]++;	# i.e. binned bord props should be normalised with n_nf...
	          $nbordfi_phos_here[$binfi]++;

	          $nbordFQ_phos_here[$binFQ]++;
	          $nbordMQ_phos_here[$binMQ]++;
	          $nbordKD_phos_here[$binKD]++;

		  $bord_write	= 1;
	        } else {
	          $bord_write	= 0;
	        }
	        $n_nf_phos_here[$bin]++;

	        $net_phos_here[$bin]	+= $netQ;
	        $frac_phos_here[$bin]	+= $fracQ;
		$meanQ_phos_here[$bin]	+= $meanQ_win;
		$KDnorm_phos_here[$bin]	+= $KDnorm_win;
		$ent_phos_here[$bin]	+= $ent_win;

	        $nfi_nf_phos_here[$binfi]++;
		$entfi_phos_here[$binfi]	+= $ent_win;

	        $nFQ_nf_phos_here[$binFQ]++;
	        $nMQ_nf_phos_here[$binMQ]++;
	        $nKD_nf_phos_here[$binKD]++;

		$foldind		= 2.785 * $KDnorm_win - $meanQ_win - 1.151;
		printf PSEQS "PHOS ";
		printf PSEQS "%7.3f %7.3f %7.3f", $windis[$naa], $netQ, $fracQ;
		printf PSEQS "%8.3f %7.3f %8.3f %7.3f %6.3f", $meanQ_win, $KDnorm_win, $foldind, $ent_win, $frac_win;
		printf PSEQS "%7.3f  ", $bord_write;
		printf PSEQS "%7.3f  ", $meanQ_winNET;
		printf PSEQS "$subseq\n";
	      }
	    } else {
	      $n_unphos_here[$bin]++;
	      $nfi_unphos_here[$binfi]++;
	      if ($winP eq 'yes') {		# only increment bord bins where winP='yes'
	        if ($bord eq 'yes') {
	          $nbord_unphos_here[$bin]++;
	          $nbordfi_unphos_here[$binfi]++;

	          $nbordFQ_unphos_here[$binFQ]++;
	          $nbordMQ_unphos_here[$binMQ]++;
	          $nbordKD_unphos_here[$binKD]++;

		  $bord_write	= 1;
	        } else {
	          $bord_write	= 0;
	        }
	        $n_nf_unphos_here[$bin]++;
	        $net_unphos_here[$bin]	+= $netQ;
	        $frac_unphos_here[$bin]	+= $fracQ;
		$meanQ_unphos_here[$bin]	+= $meanQ_win;
		$KDnorm_unphos_here[$bin]	+= $KDnorm_win;
		$ent_unphos_here[$bin]		+= $ent_win;

	        $nfi_nf_unphos_here[$binfi]++;
		$entfi_unphos_here[$binfi]	+= $ent_win;

	        $nFQ_nf_unphos_here[$binFQ]++;
	        $nMQ_nf_unphos_here[$binMQ]++;
	        $nKD_nf_unphos_here[$binKD]++;

		$foldind		= 2.785 * $KDnorm_win - $meanQ_win - 1.151;
		printf PSEQS "NONP ";
		printf PSEQS "%7.3f %7.3f %7.3f", $windis[$naa], $netQ, $fracQ;
		printf PSEQS "%8.3f %7.3f %8.3f %7.3f %6.3f", $meanQ_win, $KDnorm_win, $foldind, $ent_win, $frac_win;
		printf PSEQS "%7.3f  ", $bord_write;
		printf PSEQS "%7.3f  ", $meanQ_winNET;
		printf PSEQS "$subseq\n";
	      }
	    }
	  }
	}						# CLOSE AAs in SEQs LOOP

	$nSTYunphos_processed	= 0;
	$nSTYphos_processed	= 0;
	$nbordunphos_processed	= 0;
	$nbordphos_processed	= 0;
	for ($h=1; $h<=$nbins_dis; $h++) {		# get sequence numbers of phos-STY and bord/phos-STY
	  $nSTYphos_processed		+= $n_nf_phos_here[$h];
	  $nSTYunphos_processed		+= $n_nf_unphos_here[$h];
	  $nbordphos_processed		+= $nbord_phos_here[$h];
	  $nbordunphos_processed	+= $nbord_unphos_here[$h];
	}
	$nSTYhere			= $nSTYphos_processed + $nSTYunphos_processed;
	if ($nSTYhere != 0) {
	  $frac_phos_seqs{$seqid[$nseq]}	= $nSTYphos_processed / $nSTYhere;
	  $frac_bord_seqs{$seqid[$nseq]}	= ($nbordphos_processed + $nbordunphos_processed) / $nSTYhere;
	} else {
	  $frac_phos_seqs{$seqid[$nseq]}	= 99;
	  $frac_bord_seqs{$seqid[$nseq]}	= 99;
	}

	printf PSEQS "KD meanQ FldInd Dis Ent FracPhos FracBord = ";
#	printf PSEQS "%6.3f %6.3f ", $KDnorm_seqs{$seqid[$nseq]}, $chrmean_seqs{$seqid[$nseq]};
	printf PSEQS "%6.3f %6.3f ", $KDnorm_seqs{$seqid[$nseq]}, $chr_tot_seqs{$seqid[$nseq]};
	printf PSEQS "%6.3f %6.3f %6.3f ", $foldind_seqs{$seqid[$nseq]}, $dis_seqs{$seqid[$nseq]}, $ent_seqs{$seqid[$nseq]};
	printf PSEQS "%6.3f %6.3f ", $frac_phos_seqs{$seqid[$nseq]}, $frac_bord_seqs{$seqid[$nseq]};
	printf PSEQS "\n";
	printf PSEQS "ENDID $seqid[$nseq]\n\n";

	for ($h=1; $h<=$nbins_dis; $h++) {
	  $n_phos[$h]		+= $n_phos_here[$h];
	  $n_unphos[$h]		+= $n_unphos_here[$h];
	  $net_phos[$h]		+= $net_phos_here[$h];
	  $net_unphos[$h]	+= $net_unphos_here[$h];
	  $frac_phos[$h]	+= $frac_phos_here[$h];
	  $frac_unphos[$h]	+= $frac_unphos_here[$h];
	  $n_nf_phos[$h]	+= $n_nf_phos_here[$h];
	  $n_nf_unphos[$h]	+= $n_nf_unphos_here[$h];
	  $nbord_phos[$h]	+= $nbord_phos_here[$h];
	  $nbord_unphos[$h]	+= $nbord_unphos_here[$h];
	  $meanQ_phos[$h]	+= $meanQ_phos_here[$h];
	  $meanQ_unphos[$h]	+= $meanQ_unphos_here[$h];
	  $KDnorm_phos[$h]	+= $KDnorm_phos_here[$h];
	  $KDnorm_unphos[$h]	+= $KDnorm_unphos_here[$h];
	  $ent_phos[$h]		+= $ent_phos_here[$h];
	  $ent_unphos[$h]	+= $ent_unphos_here[$h];
	}
	for ($hfi=1; $hfi<=$nbins_fi; $hfi++) {
	  $nfi_phos[$hfi]		+= $nfi_phos_here[$hfi];
	  $nfi_unphos[$hfi]		+= $nfi_unphos_here[$hfi];
	  $nfi_nf_phos[$hfi]		+= $nfi_nf_phos_here[$hfi];
	  $nfi_nf_unphos[$hfi]		+= $nfi_nf_unphos_here[$hfi];
	  $nbordfi_phos[$hfi]		+= $nbordfi_phos_here[$hfi];
	  $nbordfi_unphos[$hfi]		+= $nbordfi_unphos_here[$hfi];
	  $entfi_phos[$hfi]		+= $entfi_phos_here[$hfi];
	  $entfi_unphos[$hfi]		+= $entfi_unphos_here[$hfi];

	  $nFQ_nf_phos[$hfi]		+= $nFQ_nf_phos_here[$hfi];
	  $nFQ_nf_unphos[$hfi]		+= $nFQ_nf_unphos_here[$hfi];
	  $nbordFQ_phos[$hfi]		+= $nbordFQ_phos_here[$hfi];
	  $nbordFQ_unphos[$hfi]		+= $nbordFQ_unphos_here[$hfi];
	  $nMQ_nf_phos[$hfi]		+= $nMQ_nf_phos_here[$hfi];
	  $nMQ_nf_unphos[$hfi]		+= $nMQ_nf_unphos_here[$hfi];
	  $nbordMQ_phos[$hfi]		+= $nbordMQ_phos_here[$hfi];
	  $nbordMQ_unphos[$hfi]		+= $nbordMQ_unphos_here[$hfi];
	  $nKD_nf_phos[$hfi]		+= $nKD_nf_phos_here[$hfi];
	  $nKD_nf_unphos[$hfi]		+= $nKD_nf_unphos_here[$hfi];
	  $nbordKD_phos[$hfi]		+= $nbordKD_phos_here[$hfi];
	  $nbordKD_unphos[$hfi]		+= $nbordKD_unphos_here[$hfi];

	}


}							# CLOSE SEQUENCES LOOP
close (PSEQS);

$file_here		= 'bins.txt';
open (BINS, ">$file_here") or die "cannot open bins.txt\n";
printf BINS "dis_bin win_netQ_P win_netQ_U win_fracQ_P win_fracQ_U ";
printf BINS "Q_diffPU fracQ_diffPU ";
printf BINS "meanQ_P meanQ_U  meanKD_P meanKD_U\n";
for ($h=1; $h<=$nbins_dis; $h++) {
	  $meanQ_phos[$h]	+= $meanQ_phos_here[$h];
	  $meanQ_unphos[$h]	+= $meanQ_unphos_here[$h];
	  $KDnorm_phos[$h]	+= $KDnorm_phos_here[$h];
	  $KDnorm_unphos[$h]	+= $KDnorm_unphos_here[$h];
	  $ent_phos[$h]		+= $ent_phos_here[$h];
	  $ent_unphos[$h]	+= $ent_unphos_here[$h];
#  $mean_frac_unphos	= $frac_unphos[$h] / $n_unphos[$h];
  if ($n_unphos[$h] != 0) {
    $n_PUratio[$h]	= $n_phos[$h] / $n_unphos[$h];
  } else {
    $n_PUratio[$h]	= 0;
  }
  if ($n_nf_phos[$h] != 0) {
    $mean_net_phos	= $net_phos[$h] / $n_nf_phos[$h];
    $mean_frac_phos	= $frac_phos[$h] / $n_nf_phos[$h];
    $mean_meanQ_phos[$h]	= $meanQ_phos[$h] / $n_nf_phos[$h];
    $mean_KDnorm_phos[$h]	= $KDnorm_phos[$h] / $n_nf_phos[$h];
    $mean_ent_phos[$h]		= $ent_phos[$h] / $n_nf_phos[$h];
  } else {
    $mean_net_phos	= 0;
    $mean_frac_phos	= 0;
    $mean_meanQ_phos[$h]	= 0;
    $mean_KDnorm_phos[$h]	= 0;
    $mean_ent_phos[$h]		= 0;
  }
  if ($n_nf_unphos[$h] != 0) {
    $mean_net_unphos	= $net_unphos[$h] / $n_nf_unphos[$h];
    $mean_frac_unphos	= $frac_unphos[$h] / $n_nf_unphos[$h];
    $mean_meanQ_unphos[$h]	= $meanQ_unphos[$h] / $n_nf_unphos[$h];
    $mean_KDnorm_unphos[$h]	= $KDnorm_unphos[$h] / $n_nf_unphos[$h];
    $mean_ent_unphos[$h]	= $ent_unphos[$h] / $n_nf_unphos[$h];
  } else {
    $mean_net_unphos	= 0;
    $mean_frac_unphos	= 0;
    $mean_meanQ_unphos[$h]	= 0;
    $mean_KDnorm_unphos[$h]	= 0;
    $mean_ent_unphos[$h]	= 0;
  }
  $netQ_diff[$h]	= $mean_net_phos - $mean_net_unphos;
  #changed below line due to printf error newer perl Max/Jim/Desnoyers 020518
  #$fracQ_diff[$h]	= $mean_frac_phos - $mean_frac_unphos;
  printf BINS "%7.0f", $h;
  printf BINS "%11.3f %10.3f", $mean_net_phos, $mean_net_unphos;
  printf BINS "%12.3f %11.3f", $mean_frac_phos, $mean_frac_unphos;
  printf BINS "%9.3f %12.3f", $h, $netQ_diff[$h];
  #changed below line due to printf error newer perl Max/Jim/Desnoyers 020518
  #printf BINS "%9.3f %12.3f", $h, $netQ_diff[$h], $fracQ_diff[$h];
  printf BINS "%8.3f %7.3f", $mean_meanQ_phos[$h], $mean_meanQ_unphos[$h];
  printf BINS "%10.3f %8.3f\n", $mean_KDnorm_phos[$h], $mean_KDnorm_unphos[$h];
}

# Newer dis bins below to mirror the Uversky fold Table

$n_phos_tot		= 0;
$n_unphos_tot		= 0;
$n_nf_phos_tot		= 0;
$n_nf_unphos_tot	= 0;
for ($h=1; $h<=$nbins_dis; $h++) {
  $n_phos_tot		+= $n_phos[$h];
  $n_unphos_tot		+= $n_unphos[$h];
  $n_nf_phos_tot	+= $n_nf_phos[$h];
  $n_nf_unphos_tot	+= $n_nf_unphos[$h];
}
printf BINS "\n\n phos_tot, unphos_tot, nf_phos_tot, nf_unphos_tot = $n_phos_tot $n_unphos_tot $n_nf_phos_tot $n_nf_unphos_tot\n";
printf BINS "\n NOTE that other than enrich_PU, all enr/enrich props use 'nf' - complete wins around STY";
printf BINS "\n NOTE that enrich = 1.0 where a bin P/U ratio for that prop = P/U ratio overall all";
printf BINS "\n\nDis_bin    Dis       n_P       n_U   n_nf_P    n_nf_U  ratio_PU enrich_PU  n_bord_P n_bord_U  bord/STY rel_brdPU";
printf BINS "     ent_P    ent_U   ent/STY rel_entPU\n";
for ($h=1; $h<=$nbins_dis; $h++) {
  $top		= $n_nf_phos[$h] * $n_nf_unphos_tot;	# calc enrich with nwins P/U
  $bot		= $n_nf_unphos[$h] * $n_nf_phos_tot;
  if (($top != 0) and ($bot != 0)) {
    $enrich_nf_PU_here	= $top/$bot;
  } else {
    $enrich_nf_PU_here	= 1.0;
  }
  if ($n_nf_unphos[$h] != 0) {
    $ratio_nf_PU_here	= $n_nf_phos[$h] / $n_nf_unphos[$h];
  } else {
    $ratio_nf_PU_here	= 1;
  }

  $bot			= $n_nf_phos[$h] + $n_nf_unphos[$h];
  $top			= $nbord_phos[$h] + $nbord_unphos[$h];
  if (($top != 0) and ($bot != 0)) {
    $STY_in_bord_here	= $top/$bot;
  } else {
    $STY_in_bord_here	= 1;
  }

  $bot			= $n_nf_phos[$h] + $n_nf_unphos[$h];
  $top			= $ent_phos[$h] + $ent_unphos[$h];
  if (($top != 0) and ($bot != 0)) {
    $STY_ent_here	= $top/$bot;
  } else {
    $STY_ent_here	= 1;
  }

  $top			= $nbord_phos[$h]*$n_nf_unphos[$h];
  $bot			= $nbord_unphos[$h]*$n_nf_phos[$h];
  if (($top != 0) and ($bot != 0)) {
    $rel_enr_PU_bord_here	= $top/$bot;
  } else {
    $rel_enr_PU_bord_here	= 1.0;
  }
  $top			= $ent_phos[$h]*$n_nf_unphos[$h];
  $bot			= $ent_unphos[$h]*$n_nf_phos[$h];
  if (($top != 0) and ($bot != 0)) {
    $rel_enr_PU_ent_here		= $top/$bot;
  } else {
    $rel_enr_PU_ent_here		= 1.0;
  }
  $dis_here		= (($h-0.5)/$nbins_dis) - 0.5;
  printf BINS "%7.0f %6.3f %9.0f %9.0f", $h, $dis_here, $n_phos[$h], $n_unphos[$h];
  printf BINS "%  9.0f %9.0f %9.4f %9.3f", $n_nf_phos[$h], $n_nf_unphos[$h], $ratio_nf_PU_here, $enrich_nf_PU_here;
  printf BINS " %9.0f %8.0f %9.4f %9.3f", $nbord_phos[$h], $nbord_unphos[$h], $STY_in_bord_here, $rel_enr_PU_bord_here;
  printf BINS " %9.0f %8.0f %9.3f %9.3f\n", $ent_phos[$h], $ent_unphos[$h], $STY_ent_here, $rel_enr_PU_ent_here;
}
printf BINS "\n";
printf BINS " See the equivalent Table for more information on these properties\n";
printf BINS " Note: eg winlenB=11 get higher PtoU bord for higher dis - ?? charge props of Plation spec\n";
printf BINS " But overall trend is fewer STY border at higher dis, perhaps due to more charge at higher dis\n";
printf BINS " So if subs seqs have more Q, then enr_bordPU would go other way\n";

$nfi_phos_tot		= 0;
$nfi_unphos_tot		= 0;
$nfi_nf_phos_tot	= 0;
$nfi_nf_unphos_tot	= 0;
for ($hfi=1; $hfi<=$nbins_fi; $hfi++) {
  $nfi_phos_tot		+= $nfi_phos[$hfi];
  $nfi_unphos_tot	+= $nfi_unphos[$hfi];
  $nfi_nf_phos_tot	+= $nfi_nf_phos[$hfi];
  $nfi_nf_unphos_tot	+= $nfi_nf_unphos[$hfi];
}
printf BINS "\n phos_tot, unphos_tot, nf_phos_tot, nf_unphos_tot = $nfi_phos_tot $nfi_unphos_tot $nfi_nf_phos_tot $nfi_nf_unphos_tot\n";
printf BINS "\n NOTE that other than enrich_PU, all enr/enrich props use 'nf' - complete wins around STY";
printf BINS "\n NOTE that enrich = 1.0 where a bin P/U ratio for that prop = P/U ratio overall all";
printf BINS "\n\n FI_bin FldInd       n_P       n_U   n_nf_P    n_nf_U  ratio_PU enrich_PU  n_bord_P n_bord_U  bord/STY rel_brdPU";
printf BINS "     ent_P    ent_U   ent/STY rel_entPU\n";
for ($hfi=1; $hfi<=$nbins_fi; $hfi++) {
  $top		= $nfi_nf_phos[$hfi] * $nfi_nf_unphos_tot;	# calc enrich with nwins P/U
  $bot		= $nfi_nf_unphos[$hfi] * $nfi_nf_phos_tot;
  if (($top != 0) and ($bot != 0)) {
    $enrich_nf_PU_here	= $top/$bot;
  } else {
    $enrich_nf_PU_here	= 1.0;
  }
  if ($nfi_nf_unphos[$hfi] != 0) {
    $ratio_nf_PU_here	= $nfi_nf_phos[$hfi] / $nfi_nf_unphos[$hfi];
  } else {
    $ratio_nf_PU_here	= 1;
  }

  $bot			= $nfi_nf_phos[$hfi] + $nfi_nf_unphos[$hfi];
  $top			= $nbordfi_phos[$hfi] + $nbordfi_unphos[$hfi];
  if (($top != 0) and ($bot != 0)) {
    $STY_in_bord_here	= $top/$bot;
  } else {
    $STY_in_bord_here	= 1;
  }

  $bot			= $nfi_nf_phos[$hfi] + $nfi_nf_unphos[$hfi];
  $top			= $entfi_phos[$hfi] + $entfi_unphos[$hfi];
  if (($top != 0) and ($bot != 0)) {
    $STY_ent_here	= $top/$bot;
  } else {
    $STY_ent_here	= 1;
  }

  $top			= $nbordfi_phos[$hfi]*$nfi_nf_unphos[$hfi];
  $bot			= $nbordfi_unphos[$hfi]*$nfi_nf_phos[$hfi];
  if (($top != 0) and ($bot != 0)) {
    $rel_enr_PU_bord_here	= $top/$bot;
  } else {
    $rel_enr_PU_bord_here	= 1.0;
  }
  $top			= $entfi_phos[$hfi]*$nfi_nf_unphos[$hfi];
  $bot			= $entfi_unphos[$hfi]*$nfi_nf_phos[$hfi];
  if (($top != 0) and ($bot != 0)) {
    $rel_enr_PU_ent_here		= $top/$bot;
  } else {
    $rel_enr_PU_ent_here		= 1.0;
  }
  $fi_here		= -2.151 + ($hfi-0.5)*((1.634+2.151)/$nbins_fi);
  printf BINS "%7.0f %6.3f %9.0f %9.0f", $hfi, $fi_here, $nfi_phos[$hfi], $nfi_unphos[$hfi];
  printf BINS "%  9.0f %9.0f %9.4f %9.3f", $nfi_nf_phos[$hfi], $nfi_nf_unphos[$hfi], $ratio_nf_PU_here, $enrich_nf_PU_here;
  printf BINS " %9.0f %8.0f %9.4f %9.3f", $nbordfi_phos[$hfi], $nbordfi_unphos[$hfi], $STY_in_bord_here, $rel_enr_PU_bord_here;
  printf BINS " %9.0f %8.0f %9.3f %9.3f\n", $entfi_phos[$hfi], $entfi_unphos[$hfi], $STY_ent_here, $rel_enr_PU_ent_here;
}
printf BINS "\nMaking sense of the bins:\n";
printf BINS "Fold Value    below zero = unfolded, above zero = folded (Uversky / Prilusky)\n";
printf BINS "n_P/U         numbers of phos/unphos STY picked up from seqs, wherever in seq\n";
printf BINS "n_nf_P/U      nos of phos/unphos STY that are within complete winlenP (eg 21) windows\n";
printf BINS "              these nf numbers of P/U are used for subsequent ratio/enrichment calcs\n";
printf BINS "ratio_PU      simple ratio of phos / unphos in each FI bin\n";
printf BINS "enrich_PU     enrichment of P to U, normalised such that average over all bins = 1\n";
printf BINS "              the classic result is that we should have > 1 for unfolded proteins\n";
printf BINS "n_bord_P/U    STY phos/unphos nos (in comp winlenP) at charge borders (tested in winlenB)\n";
printf BINS "bord/STY      for all STY, the fraction in charge border as function of FI bin\n";
printf BINS "              see a general trend, presumably related to more charge for less folded proteins\n";
printf BINS "rel_brdPU     enrichment of charge borders P to U, normalised by ratio of P to U in bin\n";
printf BINS "              we're looking here - may be some enrichment for P close to fold boundary\n";
printf BINS "ent_P/U       sum over winlenP sequence entropies for phos / unphos sites\n";
printf BINS "ent/STY       for all STY, the average seq entropy per site in this bin\n";
printf BINS "rel_entPU     enrichment of these entropes P to U, normalised by ratio of P to U in bin\n";
printf BINS "              what to say about the seq entropy P to U props ??\n";

# We see rel_brdPU variation on dis and FI - but what about vs charge and KD prop bins separately?
printf BINS "\n We see rel_bord variation with dis and FI bins - what about charge and KD bins?";
printf BINS "\n\nnumbin  fracQ  bord/STY rel_brdPU  meanQ  bord/STY rel_brdPU     KD  bord/STY rel_brdPU\n";
for ($h=1; $h<=$nbins_dis; $h++) {
  printf BINS "%7.0f", $h;
  $bot			= $nFQ_nf_phos[$h] + $nFQ_nf_unphos[$h];
  $top			= $nbordFQ_phos[$h] + $nbordFQ_unphos[$h];
  if (($top != 0) and ($bot != 0)) {
    $STY_in_bord_here	= $top/$bot;
  } else {
    $STY_in_bord_here	= 1;
  }
  $top			= $nbordFQ_phos[$h]*$nFQ_nf_unphos[$h];
  $bot			= $nbordFQ_unphos[$h]*$nFQ_nf_phos[$h];
  if (($top != 0) and ($bot != 0)) {
    $rel_enr_PU_bord_here	= $top/$bot;
  } else {
    $rel_enr_PU_bord_here	= 1.0;
  }
  printf BINS " fracQ %9.4f %9.3f", $STY_in_bord_here, $rel_enr_PU_bord_here;
  $bot			= $nMQ_nf_phos[$h] + $nMQ_nf_unphos[$h];
  $top			= $nbordMQ_phos[$h] + $nbordMQ_unphos[$h];
  if (($top != 0) and ($bot != 0)) {
    $STY_in_bord_here	= $top/$bot;
  } else {
    $STY_in_bord_here	= 1;
  }
  $top			= $nbordMQ_phos[$h]*$nMQ_nf_unphos[$h];
  $bot			= $nbordMQ_unphos[$h]*$nMQ_nf_phos[$h];
  if (($top != 0) and ($bot != 0)) {
    $rel_enr_PU_bord_here	= $top/$bot;
  } else {
    $rel_enr_PU_bord_here	= 1.0;
  }
  printf BINS "  meanQ %9.4f %9.3f", $STY_in_bord_here, $rel_enr_PU_bord_here;
  $bot			= $nKD_nf_phos[$h] + $nKD_nf_unphos[$h];
  $top			= $nbordKD_phos[$h] + $nbordKD_unphos[$h];
  if (($top != 0) and ($bot != 0)) {
    $STY_in_bord_here	= $top/$bot;
  } else {
    $STY_in_bord_here	= 1;
  }
  $top			= $nbordKD_phos[$h]*$nKD_nf_unphos[$h];
  $bot			= $nbordKD_unphos[$h]*$nKD_nf_phos[$h];
  if (($top != 0) and ($bot != 0)) {
    $rel_enr_PU_bord_here	= $top/$bot;
  } else {
    $rel_enr_PU_bord_here	= 1.0;
  }
  printf BINS "     KD %9.4f %9.3f\n", $STY_in_bord_here, $rel_enr_PU_bord_here;
}

close (BINS);

# output the sequences, with P and Order annotation lines

$file_here		= 'seq_props.out';
open (SEQS_OUT, ">$file_here") or die "cannot open seq_props.out\n";
# open (SEQS_SUM, ">POsummary.out") or die "cannot open PO.out\n";

# printf SEQS_SUM "VALUES OVER THE ENTIRE INPUT SET follow: N genes = $nseqs\n\n";
# printf SEQS_SUM "N genes with >= 1 Motif          = $nwithmotif\n";
# printf SEQS_SUM "N genes with >= 1 Motif also DIS = $nwithmotifDIS\n";
# $answerA	= $totalmotifs/$nseqs;
# printf SEQS_SUM "AVG Motifs per gene              = $answerA\n";
# $answerB	= $totalmotifsDIS/$nseqs;
# printf SEQS_SUM "AVG Motifs per gene also DIS     = $answerB\n";
# printf SEQS_SUM "Perc disorder for all genes      = $PercDISTOT\n\n";

# printf SEQS_SUM "VALUES FOR EACH GENE follow\n\n";
for ($nseq=1; $nseq<=$nseqs; $nseq++) {
	printf SEQS_OUT "$seqid[$nseq]\n";
#	printf SEQS_SUM "$seqid[$nseq]\n";

	printf SEQS_OUT "KD meanQ FldInd Dis Ent FracPhos FracBord = ";
	printf SEQS_OUT "%6.3f %6.3f ", $KDnorm_seqs{$seqid[$nseq]}, $chrmean_seqs{$seqid[$nseq]};
	printf SEQS_OUT "%6.3f %6.3f %6.3f ", $foldind_seqs{$seqid[$nseq]}, $dis_seqs{$seqid[$nseq]}, $ent_seqs{$seqid[$nseq]};
	printf SEQS_OUT "%6.3f %6.3f ", $frac_phos_seqs{$seqid[$nseq]}, $frac_bord_seqs{$seqid[$nseq]};
	printf SEQS_OUT "\n";

	printf SEQS_OUT "SEQ=$seq{$seqid[$nseq]}\n";
	printf SEQS_OUT "DIS=$propchar{$seqid[$nseq]}\n";
	printf SEQS_OUT "CHR=$signchar{$seqid[$nseq]}\n";
	printf SEQS_OUT "CHR=$chrcharA{$seqid[$nseq]}\n";
	printf SEQS_OUT "CHR=$chrcharB{$seqid[$nseq]}\n";


	printf SEQS_OUT "\n";


	if ($extra eq "yes") {			# optional output of vertical data list, one file per sequence
	    $filehere	= 'seq_'.$nseq.'.txt';
	    open (EXTRA, ">$filehere") or die "cannot open $filehere\n";
	    printf EXTRA "$seqid[$nseq]\n";
	    @seqextra	= split ("",$seq{$seqid[$nseq]});
	    $aatop	= length $seq{$seqid[$nseq]};
	    @disextra	= split ("",$propchar{$seqid[$nseq]});
	    @signextra	= split ("",$signchar{$seqid[$nseq]});
	    @chrextraA	= split ("",$chrcharA{$seqid[$nseq]});
	    @chrextraB	= split ("",$chrcharB{$seqid[$nseq]});
	    for ($aa=1; $aa<=$aatop; $aa++) {
	        if ($disextra[$aa-1] eq "*") {
		    $disout	= "1";
		} else {
		    $disout	= "0";
		}
		if ($chrextraA[$aa-1] eq " ") {
		  $chrout	= $chrextraB[$aa-1];
		} else {
		  $chrout	= $chrextraA[$aa-1].$chrextraB[$aa-1];
		}
#	        printf EXTRA "$seqextra[$aa-1]  $disout  $all_winQ[$nseq][$aa]  check=$signextra[$aa-1]$chrextra[$aa-1]\n";
	        printf EXTRA "$seqextra[$aa-1]  $disout  $all_winQ[$nseq][$aa]  check=$signextra[$aa-1]$chrout\n";
	    }
	    close (EXTRA);
	}
}
close (SEQS_OUT);
# close (SEQS_SUM);

close (BLAH);

exit;
