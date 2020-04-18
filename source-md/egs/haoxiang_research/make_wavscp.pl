#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);


print STDERR "## LOG ($0): stdin expected ...\n";
while(<STDIN>) {
  chomp;
  m/(.*)\/([^\/]+)\.wav.pr.wav/g or next;
  my ($wavid, $wavfile) = ($2, $_);
  my @A = split(/[\-]/, $wavid);
  #my $ref = \$A[1];
  #$$ref = sprintf("%d", $$ref);
  $wavid = join("", @A);
  print "$wavid $wavfile\n";
}
print STDERR "## LOG ($0): stdin ended ...\n";
