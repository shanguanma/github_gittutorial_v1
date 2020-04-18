#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);
# begin sub
sub SplitFileName {
  my ($fileName, $pathName, $baseName) = @_;
  $fileName =~ m:(^.*)\/([^\/]+)$:g;
  $$pathName = $1;
  $$baseName = $2;
}
# end sub
my $numArgs = scalar @ARGV;
if ($numArgs != 3) {
  die "\n[Example]: $0 <transcript.txt> <wavlist> <tgtdir>\n\n";
}
my ($transcript, $wavlist, $tgtdir) = @ARGV;
`[ -d $tgtdir ] || mkdir -p $tgtdir`;
my $wavScpFile = $tgtdir . '/' . 'wav.scp';
print STDERR "## LOG ($0): making '$wavScpFile'\n";
open(WAVSCP, ">$wavScpFile") or die;
open(F, "$wavlist") or die;
while(<F>) {
  chomp;
  my ($pathName, $baseName); SplitFileName($_, \$pathName, \$baseName); $baseName =~ m:(.*)\.([^\.]+$):g; $baseName = $1;
  print WAVSCP "$baseName $_\n";
}
close F;
close WAVSCP;
open(F, "$transcript") or die;
open(UTT2SPK, ">$tgtdir/utt2spk") or die;
open(SEG, ">$tgtdir/segments") or die;
open(TEXT, ">$tgtdir/text") or die;
while(<F>) {
  chomp;
  m:^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$:g;
  my ($speakerId, $segFile, $start, $end, $utterance) = ($1, $2, $3, $4, $5);
  my $segId = sprintf("%s-%05d-%05d", $speakerId, $start*100, $end*100);
  print SEG "$segId $segFile $start $end\n";
  print TEXT "$segId $utterance\n";
  print UTT2SPK "$segId $speakerId\n";
}
close F;
close UTT2SPK;
close SEG;
close TEXT;
`utils/utt2spk_to_spk2utt.pl < $tgtdir/utt2spk >$tgtdir/spk2utt`;

