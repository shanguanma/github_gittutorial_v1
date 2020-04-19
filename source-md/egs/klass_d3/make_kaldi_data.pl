#!/usr/bin/perl -w 
use strict;
use utf8;
use open qw(:std :utf8);

my $numArgs = scalar @ARGV;
if ($numArgs != 1) {
  die "\n\nExample: cat formatted_text_grid.txt| $0  dir\n\n";
}
my ($dir) = @ARGV;

# begin sub

# begin sub

print STDERR "## LOG ($0): stdin expected ...\n";
open(SEG, ">$dir/segments") or die;
open(TXT, ">$dir/text") or die;
open(U2S, ">$dir/utt2spk") or die;
while(<STDIN>) {
  chomp;
  m/(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$/g or next;
  my ($spkid, $wavid, $start, $end, $text) = ($1, $2, $3, $4, $5);
  my $start_str = sprintf("%07d", $start*100);
  my $end_str = sprintf("%07d", $end*100);
  my $uttid = $spkid . '-' . $start_str . '-' . $end_str;
  print SEG "$uttid $wavid $start $end\n";
  print U2S "$uttid $spkid\n";
  print TXT "$uttid $text\n";
}
print STDERR "## LOG ($0): stdin ended ...\n";
close SEG;
close TXT;
close U2S;
