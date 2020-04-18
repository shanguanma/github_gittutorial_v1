#!/usr/bin/perl -w
use strict;
use utf8;
use open qw(:std :utf8);
# begin sub

# end sub
my $numArgs = scalar @ARGV;
if ($numArgs != 3) {
  die "\n[Example]: $0 <transcript.txt> <wavlist> <tgtdir>\n\n";
}
my ($transcript, $wavlist, $tgtdir) = @ARGV;
`[ -d $tgtdir ] || mkdir -p $tgtdir` || die;

