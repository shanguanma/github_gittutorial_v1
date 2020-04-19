#!/usr/bin/perl -w 
use strict;
use utf8;
use open qw(:std :utf8);

my $numArgs = scalar @ARGV;
if ($numArgs != 1) {
  die "\n\nExample cat text | $0 dict.txt > subset_dict.txt\n\n";
}
my ($dictFile) = @ARGV;
# begin sub
sub InitWordDict {
  my ($sDictFile, $wordDictVocab) = @_;
  open (F, "$sDictFile") or die "## ERROR (InitWordDict, ", __LINE__, "): cannot open file $sDictFile\n";
  while(<F>) {
    chomp;
    m/(\S+)\s+(.*)/ or next;
    my ($word, $phones) = ($1, $2);
    my @A = split(/\s+/, $phones);
    $phones = join(" ", @A);
    my $dictLine = sprintf("%s\t%s", $word, $phones);
    if (not exists $$wordDictVocab{$word}) {
      my %vocab = ();
      my $ref = $$wordDictVocab{$word} = \%vocab;
      $$ref{$dictLine} ++;
    } else {
      my $ref = $$wordDictVocab{$word};
      $$ref{$dictLine} ++;
    }
  }
  close F;
}
# end sub
my %vocab = ();
InitWordDict($dictFile, \%vocab);

print STDERR "## LOG ($0): stdin expected ...\n";
my %unique = ();
open(OUTPUT, "|sort -u") or die;
while(<STDIN>) {
  chomp;
  my @A = split(/\s+/);
  for(my $i = 0; $i < scalar @A; $i ++) {
    my $word = $A[$i];
    next if(exists $unique{$word});
    $unique{$word} ++;
    next if (not exists $vocab{$word});
    my $prons = $vocab{$word};
    foreach my $pron (keys%$prons) {
      print OUTPUT "$pron\n";
    }
  }
}
print STDERR "## LOG ($0): stdin ended ...\n";
close OUTPUT;
