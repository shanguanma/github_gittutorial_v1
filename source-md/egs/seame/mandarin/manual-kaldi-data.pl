#!/usr/bin/perl -w
use utf8;
use open qw(:std :utf8);
use strict;
# note: transcript.txt format is as follows:
# <spkerId> <wavId> <start> <end> <text> 
# transcript.txt is input file of the script.
# 
# utt2spk is definded as the format.
# the format is as follows: 
# <segId> <wavId>
# segments is definded as the foramt
# the format is as follows: 
# <segId>  <startId> <endId>
# note:<segId> is <wavId-startId-endId>
# 
# The reason for this definition is that 
# it is consistent with the callhome 
# data set used in speaker diarization.
my $numArgs = scalar @ARGV;
if ($numArgs != 1) {
  die "\nExample cat transcript.txt | $0 tmpdir\n\n";
}
my ($tgtdir) = @ARGV;
print STDERR "## LOG ($0): stdin expected\n";
open(SEG, ">$tgtdir/segments") or die;
open(U2S, ">$tgtdir/utt2spk") or die;
open(TXT, "|dos2unix>$tgtdir/text") or die;
while(<STDIN>) {
  chomp;
  m:^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$:g or next;
  my ($spkerId, $wavId, $start, $end, $text) = ($1, $2, $3, $4, $5);
  if($start >= $end) {
    print STDERR "## WARNING ($0): bad line $_\n";
    next;
  }
  my $startId = sprintf("%07d", $start*100);
  my $endId = sprintf("%07d", $end*100);
  my $segId = sprintf("%s-%s-%s", $wavId, $startId, $endId);
  print SEG "$segId $wavId $start $end\n";
  print U2S "$segId $wavId\n";
  print TXT "$segId $text\n";
}
close SEG;
close U2S;
close TXT;
`utils/utt2spk_to_spk2utt.pl < $tgtdir/utt2spk > $tgtdir/spk2utt`;
print STDERR "## LOG ($0): stdin ended\n";
