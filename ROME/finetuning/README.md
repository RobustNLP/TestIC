## Fintuning
This folder shows our caption dataset (excluding the image features, which can be large in size), our finetuned checkpoint, and how to evaluate the checkpoint.

#### Dataset
We used `caption_data.json` as the input for captions for finetuning Oscar.
For this task, synthesized images generated from the training data of COCO
Caption and
combined into Oscar's original training data.
Due to the limit of data size, we could not place the full extracted image features and labels file here.

#### Evaluating the finetuned checkpoint
`checkpoint-5-65000` in this directory holds the finetuned checkpoint for evaluation. 

We use Oscar to evaluate the finetuned checkpoint, first, install the Oscar Github repository by following the scripts in `setup_oscar_script.sh`. Then, we can evaluate the checkpoint using the script in `oscar_eval_script.sh`. The evaluation scores will be displayed in the terminal.