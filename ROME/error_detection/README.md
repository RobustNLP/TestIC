# Error detection with processing method for synonyms and word groups

This program is to detect errors according two metamorphic relations from tsv files and a pkl file.

## Files
### tsv files
Every IC system has a pair of tsv files to record the generated captions. The two files are aligned line by line.
Take the IC system ofa_base as an example:

`ofa_base_ancestor.tsv` contains all the ofa_base generated captions of ancestor image
`ofa_base_descendant` contains all the ofa_base generated captions of descendant image

### pkl files
The file `name_img_id_dict.pkl` records some basic information about the generated image pairs, including 'image_id','file_name' which are strings, and 'removed_object_ancestor', 'removed_object_descendant' which are sets.

## Implementation Principle
1. use stanza to pos tag for every word in a caption pair
2. with assistance of nltk's wordnet, generate a pair of sets which consist of keywords in ancestor and descendant seperately
3. report suspicious issues according to metamorphic relations



## Environment Setup
```
conda create --name synonym python=3.8.10
conda activate synonym
pip install stanza=='1.4.0'
pip install nltk=='3.6.1'
```

## Run 

```
python synonym_clean.py
```

report_issues can be found in the file_folder: out->IC_name




