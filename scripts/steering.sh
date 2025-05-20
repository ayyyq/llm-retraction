DEVICE=0
FULL_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME=$(basename "$FULL_MODEL_NAME")
max_new_tokens=128
temperature=0
template="continuation"

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/intervention.py \
  --model_path ${FULL_MODEL_NAME} \
  --train_filename probe-outputs/universal_truthfulness/truthfulness_train/${MODEL_NAME}/t0/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/${MODEL_NAME}/wikidata_test_continuation.jsonl \
  --template ${template} \
  --save_dir intervention-outputs/${MODEL_NAME}/universal_truthfulness_train/t0/negative_steer/wikidata_test_continuation \
  --max_new_tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --alpha 1.2 \
  --target_layer 6 7 8 9 10 11 12 13 14

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/intervention.py \
  --model_path ${FULL_MODEL_NAME} \
  --train_filename probe-outputs/universal_truthfulness/truthfulness_train/${MODEL_NAME}/t0/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/${MODEL_NAME}/wikidata_test_continuation.jsonl \
  --template ${template} \
  --save_dir intervention-outputs/${MODEL_NAME}/universal_truthfulness_train/t0/positive_steer/wikidata_test_continuation \
  --max_new_tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --alpha 1.2 \
  --target_layer 6 7 8 9 10 11 12 13 14 \
  --positive_steer

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/intervention.py \
  --model_path ${FULL_MODEL_NAME} \
  --train_filename probe-outputs/universal_truthfulness/truthfulness_train/${MODEL_NAME}/t0/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/${MODEL_NAME}/wikidata_test_continuation.jsonl \
  --template ${template} \
  --save_dir intervention-outputs/${MODEL_NAME}/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager \
  --max_new_tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --alpha 1.2 \
  --target_layer 6 7 8 9 10 11 12 13 14 \
  --fixed_token " is" \
  --attn_implementation "eager"

CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/intervention.py \
  --model_path ${FULL_MODEL_NAME} \
  --train_filename probe-outputs/universal_truthfulness/truthfulness_train/${MODEL_NAME}/t0/output.jsonl \
  --test_filename data/wikidata/wikidata_continuation/${MODEL_NAME}/wikidata_test_continuation.jsonl \
  --template ${template} \
  --save_dir intervention-outputs/${MODEL_NAME}/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager \
  --max_new_tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --alpha 1.2 \
  --target_layer 6 7 8 9 10 11 12 13 14 \
  --positive_steer \
  --fixed_token " is" \
  --attn_implementation "eager"

# llm_judge continuation
DEVICE=0,1,2,3
ckpts_dir="meta-llama"
MODEL_NAMES=("Llama-3.3-70B-Instruct")
DEV_SETS=("intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer/wikidata_test_continuation/layer6-14_alpha1.2" "intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/positive_steer/wikidata_test_continuation/layer6-14_alpha1.2" "intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2" "intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2")
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

    python src/evaluation/evaluate.py \
      --input_filename ${input_filename} \
      --retraction_filename ${retraction_output_filename} \
      --func evaluate_continuation \
      --data_source ${DATA_SOURCE}
  done
done
