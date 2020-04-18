#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);
use Getopt::Long;

my $make_transfer_dict;
my $transfer_text;
my $trasfer_dict = '';
my $word_dict = '';
my $print_oov_dict;
my $from=1;
my $tgtdir= '';
GetOptions('make_transfer_dict|make-transfer-dict'=>\$make_transfer_dict,
           'transfer-text|transfer_text'=>\$transfer_text,
           'transfer-dict|transfer_dict=s'=>\$transfer_dict,
           'word-dict|word_dict=s'=>\$word_dict,
	   'print_oov_dict|prin-oov-dict'=>\$print_oov_dict,
           'tgtdir=s'=>\$tgtdir,
           'from=i'=>\$from) or die;
# begin sub
sub Trim {
  my ($s) = @_;
  $$s =~ s/^\s+|\s+$//g;
}
sub LoadDict {
  my ($vocab, $dictFile) = @_;
  open(F, "$dictFile") or die;
  
}
sub PrintOovDict {
  my ($wordDict, $words, $oovDict) = @_;
  
}
sub TransferWords {
  my ($transferDict, $words) = @_;
  
}

# end sub
my %transferDict = ();
my %wordDict = ();
my %oovDict = ();
print STDERR "## LOG ($0): stdin expected\n";
while(<STDIN>) {
  chomp;
  my $words = '';
  my $label = '';
  my @A = split(" ");
  for(my $i = 0; $i < $from -1; $i ++) {
    $label .= $A[$i] + ' ';
  }
  my $i = 0;
  $i = $from -1 if($from >=1);
  while($i < scalar @A) {
    my $word = $A[$i];
    $words .= $word + ' ';
    $i ++;
  }
  Trim(\$words);
  next if($words =~ /^$/);
  if($transfer_text && $transfer_dict ne '') {
    TransferWords(\%transferDict, \$words);
  }
  PrintOovDict(\%wordDict, $words, \%oovDict);
  print $label, $words, "\n";
}
print STDERR "## LOG ($0): stdin ended\n";

if ($print_oov_dict) {

}
if ($make_transfer_dict) {

}
