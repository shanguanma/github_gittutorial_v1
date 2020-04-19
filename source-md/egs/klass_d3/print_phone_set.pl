#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);

print STDERR "## LOG ($0): stdin expected ...\n";
open(O, "|sort -u") or die;
while(<STDIN>) {
  chomp;
  m/(\S+)\s+(.*)$/g or next;
  my $phone_str = $2;
  my @A = split(/\s+/, $phone_str);
  for(my $i = 0; $i < scalar @A; $i ++) {
    print O "$A[$i]\n";
  }
}
print STDERR "## LOG ($0): stdin ended ...\n";
close O;
