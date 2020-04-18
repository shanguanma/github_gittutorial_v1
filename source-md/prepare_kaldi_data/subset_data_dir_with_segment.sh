#1/usr/bin/env bash

# Ma Duo 2020-3-18

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0

cmd="slurm.pl  --quiet --exclude=node06,node05"
stage=1
nj=20

log "$0 $*"
. utils/parse_options.sh || exit 1

