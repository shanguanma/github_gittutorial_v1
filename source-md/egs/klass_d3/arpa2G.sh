#!/bin/bash

echo
echo "$0 $@"
echo

. path.sh

# begin options

# end options

. parse_options.sh || exit 1

function Usage {
 cat<<END
 $0  <arpa-lm-file> <lang> <lang_test>

END
}

if [ $# -ne 3 ]; then
  Usage && exit 1
fi

arpa_lm=$1
lang=$2
lang_test=$3


[ -d $lang_test ] || mkdir -p $lang_test

cp -r $lang/* $lang_test

gunzip -c "$arpa_lm" | \
arpa2fst --disambig-symbol=#0 \
 --read-symbol-table=$lang_test/words.txt - $lang_test/G.fst

echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $lang_test/G.fst

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=$lang/phones.txt --osymbols=$lang/words.txt $lang/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize $lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize $lang_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose $lang_test/L_disambig.fst $lang_test/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose $lang/L_disambig.fst $lang_test/G.fst | \
   fstisstochastic || echo "[log:] LG is not stochastic"



