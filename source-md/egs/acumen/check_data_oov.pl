#!/usr/bin/perl -w
# By Haihua, TL@NTU, 2019
use strict;
use utf8;
use open qw(:std :utf8);
use Getopt::Long;

our $transfer_dict_file = "";
our $from = 2;
GetOptions("transfer_dict|transfer-dict=s", \$transfer_dict_file,
	  "from=i", \$from) or die;

my $usage = << 'EOF';

  Usage:
  source-scripts/egs/w2019/blizzard/check_data_oov.pl   [option] <dict>  <dir>
  [options]:
  --from=            # for kaldi text, we sohuld set 2, ignoring uttid field, default (2)
  --translate-dict=  # translate dict, translating the oov word list, with human intervention.
  for example :
   ./source-md/egs/haihua-source/egs/w2019/blizzard/check_data_oov.pl run_ubs2022_english/data/ubs_dictv0.1/lexicon.txt  data/train_imda_part3_ivr_msf_i2r_imda_boundarymic_codec  
  By Haihua, TL@NTU, 2019

EOF
my $numArgs = scalar @ARGV;

if ($numArgs != 2) {
  print STDERR $usage;
  exit 1;
}

my ($src_dict_file, $dir) = @ARGV;

# begin sub
sub InsertVocab {
  my ($vocab, $word, $pron) = @_;
  $pron =~ s:\s+: :g; $pron =~ s:^\s+::g; $pron =~ s:\s+$::g; 
  if (exists $$vocab{$word}) {
    my $pronVocab = $$vocab{$word};
    if (not exists $$pronVocab{$pron}) {
       $$pronVocab{$pron} ++;
    }
  } else {
    my %temp = ();
    my $pronVocab = $$vocab{$word} = \%temp;
    $$pronVocab{$pron} ++;
  }
}
sub LoadVocab {
  my ($inFile, $vocab) = @_;
  open (F, "$inFile") or die "## ERROR ($0): cannot open '$inFile'\n";
  while(<F>) {
    chomp;
    m:^(\S+)\s*(.*)$:g or next;
    my ($word, $pron) = ($1, $2);
    InsertVocab($vocab, $word, $pron);
  }
  close F;
}
#
sub CheckOov {
  my ($vocab, $dir, $oov_count_file, $oov_rate_file) = @_;
  if(! -e "$dir/text") {
    die "file '$dir/text' expected out there ...\n";
  }
  open(TEXT, "$dir/text") or die "## ERROR ($0): cannot open file '$dir/text'\n";
  my %oovVocab = ();
  my $total_words = 0;
  my $oov_words = 0;
  my $oov_rate = 0;
  while(<TEXT>) {
    chomp;
    my @A = split(/\s+/);
    for(my $i = $from - 1; $i < scalar @A; $i++) {
      my $word = $A[$i];
      if(not exists $$vocab{$word}) {
	$oovVocab{$word} ++;
	$oov_words ++;
      }
      $total_words ++;
    }
  }
  close TEXT;
  open(OOV_RATE, ">$oov_rate_file") or die;
  if($total_words != 0) {
    $oov_rate = $oov_words / $total_words;
    $oov_rate = sprintf("%.5f", $oov_rate);
  }
  print OOV_RATE "$total_words  $oov_words $oov_rate\n";
  close OOV_RATE;
  open(OOV, "|sort -k2nr>$oov_count_file") or die;
  my $TRANSFER;
  if( ! -e "$dir/oov_transfer_dict.txt") {
    open($TRANSFER, ">$dir/oov_transfer_dict.txt") or die;
  }
  # write oov transfer dict if necessary
  foreach my $word (keys%oovVocab) { 
    print OOV "$word\t$oovVocab{$word}\n";
    if(defined($TRANSFER)) {
      print $TRANSFER "$word\t$word\n";
    }
  }
  close OOV;
  if(defined($TRANSFER)) {
    close  $TRANSFER;
  }
}
# translate the text with the dictionary,
#  and we are ignoring the out-of-vocabulary words of the dictionary.
# That means we only translate the words we know.
sub LoadTransferVocab {
  my ($vocab, $dict_file) = @_;
  open(F, "$dict_file") or die "## ERROR: cannot open file $dict_file\n";
  while(<F>) {
    chomp;
    m:(^\S+)\s*(.*)$:g or next;
    my $word = $1;
    my $phoneStr = $2;
    $$vocab{$1} = $2;
  }
  close(F);
}
#
sub TransferText {
  my ($vocab, $dir) = @_;
  my($stext, $ttext) = ("$dir/text", "$dir/transferred_text.gz");
  my $retcode = system("gzip -c $stext > $ttext");
  if($retcode != 0) {
    die "## LOG ($0): failed to make '$ttext' from '$stext'\n";
  }
  open(F, "gzip -cd $ttext|") or 
    die "## LOG ($0): failed to open file '$ttext'";
  open(OUTPUT, ">$stext") or 
    die "## ERROR ($0): failed to open file '$ttext' to write\n";
  while(<F>) {
    chomp;
    my @A = split(/\s+/);
    for(my $i = $from-1; $i < @A; $i ++) {
      if(exists $$vocab{$A[$i]}) {
	 $A[$i] = $$vocab{$A[$i]};
      }
    }
    my $line = join(' ', @A);
    $line =~ s:\s+: :g; 
    print OUTPUT "$line\n";
  }
  close F;
  close OUTPUT;
  $retcode = system("[ -f $ttext ] && rm $ttext");
  if ($retcode != 0) {
    die "## ERROR ($0): failed to remove file '$ttext'\n";
  }
}
# end sub

my %vocab = ();

LoadVocab($src_dict_file, \%vocab);
my $oov_count_file = "$dir/oov_count.txt";
if (! -e $oov_count_file) {
  CheckOov(\%vocab, $dir, $oov_count_file, "$dir/oov_rate");
}
if ( $transfer_dict_file ne "" ) { ## transfer the text
  my %transfer_vocab = ();
  LoadTransferVocab(\%transfer_vocab, $transfer_dict_file);
  TransferText(\%transfer_vocab, $dir);
  my $oov_count_file = "$dir/oov_count02.txt";
  CheckOov(\%vocab, $dir, $oov_count_file, "$dir/oov_rate02");
}
