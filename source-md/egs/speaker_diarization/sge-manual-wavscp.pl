#!/usr/bin/perl -w
use strict;
use utf8; 
use open qw(:std :utf8);

my $numArgs = scalar @ARGV;
if ($numArgs != 1) {
  die "\nExample: cat wav.scp|$0 sge938 > newwav.scp\n\n";
}
my ($corpusId) = @ARGV;

print STDERR "## LOG ($0): stdin expected\n";
my $index = 0;
while(<STDIN>) {
  chomp;
  m:^(\S+)\s+(.*):g or next;
  my ($fileName, $filePath) = ($1, $2);
  $index ++;
  my $fileId = sprintf("%s-%04d", $corpusId, $index);
  my @A = split(/[\-]/, $fileName);
  for(my $i = scalar @A; $i < 3; $i++) {
    push @A, "";
  }
  for (my $i = 0; $i < 1; $i ++) {
    $fileId .= '-' . $A[$i];
  }
  print "$fileId $filePath\n";
}
print STDERR "## LOG ($0): stdin ended\n";
