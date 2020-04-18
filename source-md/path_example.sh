# this is your anconda envronment
# /home3/md510/anaconda3/ is my anconda path. you should change it to your anaconda path. 
__conda_setup="$('/home3/md510/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home3/md510/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home3/md510/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home3/md510/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
 
# activate a your environment, for example, this environment name is pytorch_1.2
conda activate  pytorch_1.2  

