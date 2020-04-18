#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);

my $numArgs = scalar @ARGV;

if ($numArgs != 1) {
  die "\nExample: cat wav.scp | $0 wavlist.txt\n\n";
}
my ($wavListFile) = @ARGV;
open(F, "$wavListFile") or die;
my %vocab = ();
while(<F>) {
  chomp;
  m:(.*)\/([^\/]+)\.wav:g or next;
  $vocab{$2} = $_;
}
close F;
print STDERR "## LOG ($0): stdin expected\n";
while(<STDIN>) {
  chomp;
  m:([\S]+\.wav):g or next;
  if(not -e $1) {
    my $wavFile = $1;
    $wavFile =~ m:(.*)\/([^\/]+)\.wav:g;
    my $wavName = $2;
    if (exists $vocab{$wavName}) {
      s:[\S]+\.wav:$vocab{$wavName}:g;
    } else {
       print STDERR "File '$wavFile' does not exist ...\n";
       next;
    }
  }
  print "$_\n";
}
print STDERR "## LOG ($0): stdin ended\n";
