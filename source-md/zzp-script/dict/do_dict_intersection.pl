#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);

my $numArgs = scalar @ARGV;

my $usage = <<"EOF";

# Do intersection between two given lexicon.

[Example]:

 cat lexicon01.txt | \
 $0 lexicon02.txt intersect_lexicon_01.txt intersect_lexicon_02.txt

 By Haihua Xu, TL\@NTU, 2019.

EOF

if($numArgs != 3) {
  die $usage;
}

my ($lexfile02, $output_file01, $output_file02) = @ARGV;

# begin sub 
sub LoadVocab {
  my ($vocab, $inputfile) = @_;
  open (F, "$inputfile") or die "## ERROR ($0, ", __LINE__, "): cannot open file '$inputfile'\n";
  my %unique = ();
  while(<F>) {
    chomp;
    my @A = split(/\s+/);
    my $word = shift @A;  $word = lc $word;
    my $pron = join(" ", @A);
    my $word_pron = sprintf("%s\t%s", $word, $pron);
    next if(exists $unique{$word_pron});
    $unique{$word_pron} ++;
    my $array;
    if(not exists $$vocab{$word}) {
       my @B = ();
       $$vocab{$word} = \@B;
       $array = $$vocab{$word};
    } else {
      $array = $$vocab{$word};
    }
    push @$array, $word_pron; 
  }
  close F;
}
# end sbu

my %vocab = ();
LoadVocab(\%vocab, $lexfile02);
my $counter = 0;
my %unique = ();
my %unique2 = ();
open(OUTPUT1, ">$output_file01") or die "## ERROR ($0, ", __LINE__, "): cannot open file '$output_file01'\n";
open(OUTPUT2, ">$output_file02") or die "## ERROR ($0, ", __LINE__< "): cannot open file '$output_file02'\n";
while(<STDIN>) {
  chomp;
  $counter ++;
  if($counter == 1) {
    print STDERR "## LOG ($0): stdin seen ... \n";
  }
  my @A = split(/\s+/);
  my $word = shift @A;
  my $pron = join(" ", @A);
  my $word_pron = sprintf("%s\t%s", $word, $pron);
  next if exists $unique{$word_pron};
  $unique{$word_pron} ++;
  if(exists $vocab{$word}) {
    print OUTPUT1 "$word_pron\n";
    my $array = $vocab{$word};
    for(my $i = 0; $i < scalar @$array; $i ++) {
      next if exists $unique2{$$array[$i]};
      $unique2{$$array[$i]} ++;
      print OUTPUT2 "$$array[$i]\n";
    }
  }
}
close OUTPUT1;
close OUTPUT2;
