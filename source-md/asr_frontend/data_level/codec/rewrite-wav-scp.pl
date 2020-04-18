#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);

my $numArg = scalar @ARGV;

if ($numArg != 2) {
  die "\nUsage Example: cat segments | $0 8000 wav.scp > new_wav.scp\n\n";
}
my ($rate, $wavscp) = @ARGV;

# begin sub
sub LoadVocab {
  my ($filename, $vocab) = @_;
  open (F, "$filename") or die;
  while(<F>) {
    chomp;
    m/(\S+)\s+(.*)/g or next;
    $$vocab{$1} = $2;
  }
  close F;
}
sub IsNormalWavFile {
  my ($filename) = @_;
  my @A = split(/\s+/, $filename);
  return 0 if @A > 1;
  return 1;
}
# end sub

my %vocab = ();
# load wavscp file to the hash table for 
# the next-step indexing using the wavid 
# from segments file
LoadVocab($wavscp, \%vocab);

# open(F, ">$tgtdir/wav.scp") or die "## ERROR ($0): cannot open file '$tgtdir/wav.scp' to write\n";
print STDERR "## LOG ($0): stdin expected ...\n";
while(<STDIN>) {
  chomp;
  my @A = split(/\s+/);
  die if scalar @A != 4;
  my $wavId = $A[1];
  my $segId = $A[0];
  my $start_time = $A[2];
  my $dur = $A[3] - $A[2];
  $dur = sprintf("%.2f", $dur);
  die "## ERROR ($0): illegal segment line $_ \n" if $dur <= 0;
  die "## ERROR ($0): no wavid '$wavId' found in the specified wav.scp\n"
  if not exists $vocab{$wavId};
  my $wav_rspecifer = $vocab{$wavId};
  if(IsNormalWavFile($wav_rspecifer) == 1) {  ## here normal wave file means a single wave file without pipe operation
    $wav_rspecifer =  "sox -t wav $wav_rspecifer -c 1 -t wav -r $rate - trim $start_time $dur |";
  }elsif ($wav_rspecifer =~ m:trim\s(\d+): ){
    $wav_rspecifer =  "$wav_rspecifer";
  }else {  ## its a pipe-based specifier
    if ($wav_rspecifer =~ m:speed\s(\d+): ){
      $wav_rspecifer =~ s:\.wav -t wav - speed 0.9 \|$:\.wav -c 1 -t wav -r $rate - speed 0.9 trim $start_time $dur \|:g;
      $wav_rspecifer =~ s:\.wav -t wav - speed 1.1 \|$:\.wav -c 1 -t wav -r $rate - speed 1.1 trim $start_time $dur \|:g;
      $wav_rspecifer =~ s:\.WAV -t wav - speed 0.9 \|$:\.WAV -c 1 -t wav -r $rate - speed 0.9 trim $start_time $dur \|:g;
      $wav_rspecifer =~ s:\.WAV -t wav - speed 1.1 \|$:\.WAV -c 1 -t wav -r $rate - speed 1.1 trim $start_time $dur \|:g;
      $wav_rspecifer =~ s:\s+\|\s+sox -t wav - -t wav - speed 0.9 \|$: \| sox -c 1 -t wav - -t wav -r $rate - speed 0.9 trim $start_time $dur \|:g;
      $wav_rspecifer =~ s:\s+\|\s+sox -t wav - -t wav - speed 1.1 \|$: \| sox -c 1 -t wav - -t wav -r $rate - speed 1.1 trim $start_time $dur \|:g;
      $wav_rspecifer =~ s:-t\s+wav\s+-\s+\|\s+sox -t wav - -t wav -r $rate - speed:-c 1 -t wav -r $rate - speed:g;
    }else{
      if ($wav_rspecifer =~ m:sox: ){
        if ($wav_rspecifer =~ m:-t\s+wav\s+-\s+\|$: ) {
          $wav_rspecifer =~ s:-t\s+wav\s+-\s+\|$:-c 1 -t wav -r $rate - trim $start_time $dur \|:g;
        }else{
          $wav_rspecifer = "$wav_rspecifer sox -c 1 -t wav - -t wav -r $rate - trim $start_time $dur |";
        }
      }else{
        $wav_rspecifer = "$wav_rspecifer sox -c 1 -t wav - -t wav -r $rate - trim $start_time $dur |";
      }
    }
  }
  print "$segId $wav_rspecifer\n";
}
print STDERR "## LOG ($0): stdin ended ...\n";
