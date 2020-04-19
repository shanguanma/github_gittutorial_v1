#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);

print STDERR "## LOG ($0): stdin expected ...\n";
while(<STDIN>) {
  chomp;
  m/(\S+)\s+(\S+)\s+(.*)/g or next;
  my ($speaker, $label, $contents) = ($1, $2, $3);
  $speaker =~ s/m$//;
  my @A = split(/\-/, $label);
  my $s = \$A[1];
  $$s = sprintf("%02d", $$s);
  $label = join("-", @A);
  $label =~ s/_version\d+$//g;
  print "$speaker $label $contents\n";
}
print STDERR "## LOG ($0): stdin ended ...\n";
