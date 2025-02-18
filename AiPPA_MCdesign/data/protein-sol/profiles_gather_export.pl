#!/usr/bin/perl -w

$file_here		= 'blah.txt';
open (BLAH, ">>$file_here");		# file for comments, throughout server code

open (my $PROPS, "<", "STYprops.out") or die "cannot open STYprops.out profiles_gather\n";
@props	= <$PROPS>;
close ($PROPS);

$nseq				= 0;
foreach $line (@props) {
  chomp $line;
  @words			= split (" ",$line);
  $nwords			= @words;

  if (exists $words[0]) {			# non-null line
    if ($words[0] eq 'STARTID') {		# new sequence - store the ID
      $id_here			= $words[1];
      $nseq++;

      $seqs{$id_here}		= $nseq;
      $nwin			= 0;
    } elsif ($words[0] eq 'NONP')	{		# a window line, assume associate with id_here
      $nwin++;
      $KD[$nseq][$nwin]		= $words[5];
      $FI[$nseq][$nwin]		= $words[6];
      $ent[$nseq][$nwin]	= $words[7];
      $meanQ[$nseq][$nwin]	= $words[10];
    } elsif ($words[0] eq 'ENDID') {		# have the profiles
      if ($words[1] ne $id_here) {
        printf BLAH "**** STOPPING - IDs mismatch in profiles_gather.pl\n";
	exit;
      }
      $numwin[$nseq]		= $nwin;	# store the number of windows (naa - 20) for this sequence
    }
  }
}

# read in the seq_prediction_OLD.txt file and write out the updated version with profiles added (interstitial)

$file_here		= 'seq_prediction_OLD.txt';
open (OLD, "<$file_here") or die "cannot open seq_prediction_OLD.txt";
@old	= <OLD>;
close (OLD);
$file_here		= 'seq_prediction.txt';
open (NEW, ">$file_here") or die "cannot open seq_prediction.txt";

foreach $line (@old) {
  chomp $line;
  @words			= split (",",$line);
  $nwords			= @words;
  if (exists $words[0]) {			# non-null line
    if ($words[0] eq 'SEQUENCE DEVIATIONS') {	# write matched seq profiles after this line
      printf NEW "$line\n";				# write the SEQUENCE DEVIATIONS line
      $id_here			= $words[1];

      if (exists $seqs{$id_here}) {
        $nseq_here		= $seqs{$id_here};
	$nwin_here		= $numwin[$nseq_here];
	printf NEW "SEQUENCE PROFILE KyteDoolittle,$id_here";
	for ($n=1; $n<=$nwin_here; $n++) { printf NEW ",$KD[$nseq_here][$n]"; }
	printf NEW "\n";
	printf NEW "SEQUENCE PROFILE Uversky??,$id_here";
	for ($n=1; $n<=$nwin_here; $n++) { printf NEW ",$FI[$nseq_here][$n]"; }
	printf NEW "\n";
	printf NEW "SEQUENCE PROFILE entropy,$id_here";
	for ($n=1; $n<=$nwin_here; $n++) { printf NEW ",$ent[$nseq_here][$n]"; }
	printf NEW "\n";
	printf NEW "SEQUENCE PROFILE charge,$id_here";
	for ($n=1; $n<=$nwin_here; $n++) { printf NEW ",$meanQ[$nseq_here][$n]"; }
	printf NEW "\n";
      } else {

# ?? - sequence unmatched - what to do?
      }
    } else {
      printf NEW "$line\n";			# all other (than SEQUENCE DEVIATIONS) lines write and carry on
    }
  } else {					# null line write
    printf NEW "$line\n";
  }
}

close (NEW);
close (BLAH);

exit;
