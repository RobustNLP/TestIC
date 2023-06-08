# Clone the Oscar and VinVL repo:
# https://github.com/microsoft/Oscar.git

export MODEL_DIR=/path/to/model
export EVAL_DATA_DIR=/path/to/coco_eval

python oscar/run_captioning.py --do_eval --data_dir $EVAL_DATA_DIR --test_yaml test.yaml --per_gpu_eval_batch_size 10 --num_beams 5 --max_gen_length 20 --eval_model_dir $MODEL_DIR