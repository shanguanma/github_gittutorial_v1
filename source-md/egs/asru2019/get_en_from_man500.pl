#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);



while (<STDIN>){
  chomp;
  m/(\S*)\s+(.+)/g or die;
  my $utt=$1;
  my $txt=$2;
  if($txt =~ m/[a-z]+/g){
    print("$utt\n")
}
}
