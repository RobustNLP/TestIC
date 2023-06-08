# First, copy this script and `ofa_inference.py` to the 
# cloned OFA repo. Then, run this script from the OFA repo.
# OFA can be cloned by running:
# git clone https://github.com/OFA-Sys/OFA.git


export IN_DIR=/some/path/with/images
export OUT_FILE=out.tsv

python ofa_inference.py --in_dir $IN_DIR --out_file $OUT_FILE
