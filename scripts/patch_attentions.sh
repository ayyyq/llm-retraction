DEVICE=0
ckpts_dir="meta-llama"
MODEL_NAMES=("Llama-3.1-8B-Instruct")
DEV_SETS=("wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation")
max_new_tokens=128
temperature=0
template="continuation"

for MODEL_NAME in ${MODEL_NAMES[@]}
do
  for DEV_SET in ${DEV_SETS[@]}
  do
    CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/get_attentions.py \
      --model_path ${ckpts_dir}/${MODEL_NAME} \
      --input_filename data/${DEV_SET}.jsonl \
      --output_filename probe-outputs/${DEV_SET}/${MODEL_NAME}/t${temperature}_fixed_is/eager/output.jsonl \
      --template ${template} \
      --max_new_tokens ${max_new_tokens} \
      --temperature ${temperature} \
      --fixed_token " is"
  done
done

DEVICE=0
ckpts_dir="meta-llama"
MODEL_NAME="Llama-3.1-8B-Instruct"
max_new_tokens=128
temperature=0
template="continuation"

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/intervention_save_attentions.py \
  --model_path ${ckpts_dir}/${MODEL_NAME} \
  --attn_implementation eager \
  --train_filename probe-outputs/universal_truthfulness/truthfulness_train/${MODEL_NAME}/t0/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/${MODEL_NAME}/wikidata_test_continuation.jsonl \
  --template ${template} \
  --save_dir intervention-outputs/${MODEL_NAME}/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager \
  --max_new_tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --target_layer 6 7 8 9 10 11 12 13 14 \
  --alpha 1.2 \
  --fixed_token " is"

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/intervention_save_attentions.py \
  --model_path ${ckpts_dir}/${MODEL_NAME} \
  --attn_implementation eager \
  --train_filename probe-outputs/universal_truthfulness/truthfulness_train/${MODEL_NAME}/t0/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/${MODEL_NAME}/wikidata_test_continuation.jsonl \
  --template ${template} \
  --save_dir intervention-outputs/${MODEL_NAME}/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager \
  --max_new_tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --target_layer 6 7 8 9 10 11 12 13 14 \
  --alpha 1.2 \
  --positive_steer \
  --fixed_token " is"

# patch value vectors
DEVICE=0
ckpts_dir="meta-llama"
MODEL_NAME="Llama-3.1-8B-Instruct"
max_new_tokens=128
temperature=0
template="continuation"

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/patch_attentions.py \
  --model_path ${ckpts_dir}/${MODEL_NAME} \
  --train_filename intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl \
  --fixed_token " is" \
  --patch_value_vectors

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/patch_attentions.py \
  --model_path ${ckpts_dir}/${MODEL_NAME} \
  --train_filename intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl \
  --fixed_token " is" \
  --patch_value_vectors

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/patch_attentions.py \
  --model_path ${ckpts_dir}/${MODEL_NAME} \
  --train_filename intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl \
  --patch_value_vectors

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/patch_attentions.py \
  --model_path ${ckpts_dir}/${MODEL_NAME} \
  --train_filename intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl \
  --patch_value_vectors

DEVICE=0,1,2,3
ckpts_dir="meta-llama"
MODEL_NAMES=("Llama-3.3-70B-Instruct")
DEV_SETS=("intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/patch_value_vectors" "intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/patch_value_vectors" "intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/wo_fixed_token/patch_value_vectors" "intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/wo_fixed_token/patch_value_vectors")
TEMPERATURE=0
DATA_SOURCE="wikidata"

for MODEL_NAME in ${MODEL_NAMES[@]}
do
  for DEV_SET in ${DEV_SETS[@]}
  do
    input_filename="${DEV_SET}/output.jsonl"
    name_input_filename="${DEV_SET}/llm_judge/extract_answers_input.jsonl"
    name_output_filename="${DEV_SET}/llm_judge/extract_answers_output.jsonl"
    retraction_input_filename="${DEV_SET}/llm_judge/retraction_input.jsonl"
    retraction_output_filename="${DEV_SET}/llm_judge/retraction_output.jsonl"

    python src/evaluation/evaluate.py \
      --input_filename ${input_filename} \
      --name_filename ${input_filename} \
      --func detect_retraction_prompt \
      --continuation \
      --data_source ${DATA_SOURCE}

    CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/inference.py \
      --model_name ${ckpts_dir}/${MODEL_NAME} \
      --input_filename ${retraction_input_filename} \
      --output_filename ${retraction_output_filename} \
      --stop "<|eot_id|>" \
      --max_tokens 512 \
      --temperature ${TEMPERATURE} \
      --chat \
      --max_model_len 100000

#    python src/evaluation/evaluate.py \
#      --input_filename ${input_filename} \
#      --retraction_filename ${retraction_output_filename} \
#      --func evaluate_continuation \
#      --data_source ${DATA_SOURCE}
  done
done

for MODEL_NAME in ${MODEL_NAMES[@]}
do
  for DEV_SET in ${DEV_SETS[@]}
  do
    input_filename="${DEV_SET}/output.jsonl"
    name_input_filename="${DEV_SET}/llm_judge/extract_answers_input.jsonl"
    name_output_filename="${DEV_SET}/llm_judge/extract_answers_output.jsonl"
    retraction_input_filename="${DEV_SET}/llm_judge/retraction_input.jsonl"
    retraction_output_filename="${DEV_SET}/llm_judge/retraction_output.jsonl"

    python src/evaluation/evaluate.py \
      --input_filename ${input_filename} \
      --retraction_filename ${retraction_output_filename} \
      --func evaluate_continuation \
      --data_source ${DATA_SOURCE}
  done
done

## patch attn_weights
#CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/patch_attentions.py \
#  --model_path ${ckpts_dir}/${MODEL_NAME} \
#  --train_filename intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer/wikidata_test_continuation/layer6-14_alpha1.2/output.jsonl \
#  --test_filename data/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl \
#  --patch_attn_weights \
#  --heads_positions 0 \
#  --heads_topk 48 \
#  --topk_heads_dir intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/top_heads/wikidata_test_continuation/negative-positive-steer_fixed-is_eager_layer6-14_alpha1.2
#
#CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/patch_attentions.py \
#  --model_path ${ckpts_dir}/${MODEL_NAME} \
#  --train_filename intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/positive_steer/wikidata_test_continuation/layer6-14_alpha1.2/output.jsonl \
#  --test_filename data/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl \
#  --patch_attn_weights \
#  --heads_positions 0 \
#  --heads_topk 48 \
#  --topk_heads_dir intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/top_heads/celebrity_test_continuation/negative-positive-steer_fixed-is_eager_layer6-14_alpha1.2
