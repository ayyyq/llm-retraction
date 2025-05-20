DEVICE=0
ckpts_dir="meta-llama"
MODEL_NAME="Llama-3.1-8B-Instruct"
DEV_SETS=("universal_truthfulness/truthfulness_train" "universal_truthfulness/truthfulness_test")
max_new_tokens=128
temperature=0
template="continuation"

for DEV_SET in ${DEV_SETS[@]}
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/get_activations.py \
    --model_path ${ckpts_dir}/${MODEL_NAME} \
    --input_filename data/${DEV_SET}.jsonl \
    --output_filename probe-outputs/${DEV_SET}/${MODEL_NAME}/t${temperature}/output.jsonl \
    --template ${template} \
    --max_new_tokens ${max_new_tokens} \
    --temperature ${temperature}
done

# continuation
DEVICE=0
ckpts_dir="meta-llama"
MODEL_NAME="Llama-3.1-8B-Instruct"
DEV_SETS=("wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation" "wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_train_continuation" "celebrity/celebrity_continuation/Llama-3.1-8B-Instruct/celebrity_test_continuation" "celebrity/celebrity_continuation/Llama-3.1-8B-Instruct/celebrity_train_continuation")
max_new_tokens=128
temperature=0
template="continuation"

for DEV_SET in ${DEV_SETS[@]}
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/get_activations.py \
    --model_path ${ckpts_dir}/${MODEL_NAME} \
    --input_filename data/${DEV_SET}.jsonl \
    --output_filename probe-outputs/${DEV_SET}/${MODEL_NAME}/t${temperature}/output.jsonl \
    --template ${template} \
    --max_new_tokens ${max_new_tokens} \
    --temperature ${temperature} \
    --do_generate
done

DEVICE=0
ckpts_dir="meta-llama"
MODEL_NAME="Llama-3.1-8B-Instruct"
DEV_SETS=("wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation"  "celebrity/celebrity_continuation/Llama-3.1-8B-Instruct/celebrity_test_continuation")
max_new_tokens=128
temperature=0
template="continuation"

for DEV_SET in ${DEV_SETS[@]}
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/expr/get_activations.py \
    --model_path ${ckpts_dir}/${MODEL_NAME} \
    --input_filename data/${DEV_SET}.jsonl \
    --output_filename probe-outputs/${DEV_SET}/${MODEL_NAME}/t${temperature}_fixed_is/eager/output.jsonl \
    --template ${template} \
    --max_new_tokens ${max_new_tokens} \
    --temperature ${temperature} \
    --do_generate \
    --fixed_token " is" \
    --attn_implementation "eager"
done

# llm_judge continuation
DEVICE=0,1,2,3
ckpts_dir="meta-llama"
MODEL_NAMES=("Llama-3.3-70B-Instruct")
DEV_SETS=("probe-outputs/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation/Llama-3.1-8B-Instruct/t0" "probe-outputs/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation/Llama-3.1-8B-Instruct/t0_fixed_is/eager" "probe-outputs/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_train_continuation/Llama-3.1-8B-Instruct/t0")
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


DEVICE=0,1,2,3
ckpts_dir="meta-llama"
MODEL_NAMES=("Llama-3.3-70B-Instruct")
DEV_SETS=("probe-outputs/celebrity/celebrity_continuation/Llama-3.1-8B-Instruct/celebrity_test_continuation/Llama-3.1-8B-Instruct/t0" "probe-outputs/celebrity/celebrity_continuation/Llama-3.1-8B-Instruct/celebrity_test_continuation/Llama-3.1-8B-Instruct/t0_fixed_is/eager" "probe-outputs/celebrity/celebrity_continuation/Llama-3.1-8B-Instruct/celebrity_train_continuation/Llama-3.1-8B-Instruct/t0")
TEMPERATURE=0
DATA_SOURCE="celebrity"

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