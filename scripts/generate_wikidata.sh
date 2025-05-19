# positive verification questions
DEVICE=0
ckpts_dir="meta-llama"
MODEL_NAMES=("Llama-3.1-8B-Instruct")
DEV_SETS=("wikidata/wikidata_test_where" "wikidata/wikidata_test_profession" "wikidata/wikidata_train_where" "wikidata/wikidata_train_profession")
TEMPERATURE=0

for MODEL_NAME in ${MODEL_NAMES[@]}
do
  for DEV_SET in ${DEV_SETS[@]}
  do
    CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/inference.py \
      --model_name ${ckpts_dir}/${MODEL_NAME} \
      --input_filename data/${DEV_SET}.jsonl \
      --output_filename vllm-outputs/${DEV_SET}/${MODEL_NAME}/t${TEMPERATURE}/output.jsonl \
      --max_tokens 128 \
      --temperature ${TEMPERATURE} \
      --chat
  done
done

# temperature sampling
DEVICE=0
ckpts_dir="meta-llama"
MODEL_NAME="Llama-3.1-8B-Instruct"
DEV_SETS=("wikidata/wikidata_test_free" "wikidata/wikidata_train_free")
TEMPERATURE=0.7
TOP_P=0.95

for DEV_SET in ${DEV_SETS[@]}
do
  for ((seed=0; seed<5; seed++))
  do
    CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/inference.py \
      --model_name ${ckpts_dir}/${MODEL_NAME} \
      --input_filename data/${DEV_SET}.jsonl \
      --output_filename vllm-outputs/${DEV_SET}/${MODEL_NAME}/t${TEMPERATURE}_p${TOP_P}/seed${seed}/output.jsonl \
      --max_tokens 128 \
      --temperature ${TEMPERATURE} \
      --top_p ${TOP_P} \
      --seed ${seed} \
      --chat
  done
done

# evaluate free
DEVICE=0,1,2,3
ckpts_dir="meta-llama"
MODEL_NAMES=("Llama-3.3-70B-Instruct")
DEV_SETS=("vllm-outputs/wikidata/wikidata_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed0" "vllm-outputs/wikidata/wikidata_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed1" "vllm-outputs/wikidata/wikidata_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed2" "vllm-outputs/wikidata/wikidata_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed3" "vllm-outputs/wikidata/wikidata_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed4" "vllm-outputs/wikidata/wikidata_train_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed0" "vllm-outputs/wikidata/wikidata_train_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed1" "vllm-outputs/wikidata/wikidata_train_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed2" "vllm-outputs/wikidata/wikidata_train_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed3" "vllm-outputs/wikidata/wikidata_train_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed4")
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
      --func extract_answers_prompt \
      --data_source ${DATA_SOURCE}

    CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/inference.py \
      --model_name ${ckpts_dir}/${MODEL_NAME} \
      --input_filename ${name_input_filename} \
      --output_filename ${name_output_filename} \
      --stop "<|eot_id|>" "<|im_end|>" \
      --max_tokens 512 \
      --temperature ${TEMPERATURE} \
      --chat \
      --max_model_len 100000

    python src/evaluation/evaluate.py \
      --input_filename ${input_filename} \
      --name_filename ${name_output_filename} \
      --func detect_retraction_prompt \
      --data_source ${DATA_SOURCE}

    CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/inference.py \
      --model_name ${ckpts_dir}/${MODEL_NAME} \
      --input_filename ${retraction_input_filename} \
      --output_filename ${retraction_output_filename} \
      --stop "<|eot_id|>" "<|im_end|>" \
      --max_tokens 512 \
      --temperature ${TEMPERATURE} \
      --chat \
      --max_model_len 100000

    python src/evaluation/evaluate.py \
      --input_filename ${input_filename} \
      --name_filename ${name_output_filename} \
      --retraction_filename ${retraction_output_filename} \
      --func evaluate_free \
      --data_source ${DATA_SOURCE}
  done
done

# negative verification questions: First run get_self_where_query function (Step 2.1) in data/wikidata/generate_wikidata.py
DEVICE=0
ckpts_dir="meta-llama"
MODEL_NAME="Llama-3.1-8B-Instruct"
DEV_SETS=("wikidata/wikidata_test_free" "wikidata/wikidata_train_free")
TEMPERATURE=0

for DEV_SET in ${DEV_SETS[@]}
do
  for ((seed=0; seed<5; seed++))
  do
    CUDA_VISIBLE_DEVICES=${DEVICE} python -u src/inference.py \
      --model_name ${ckpts_dir}/${MODEL_NAME} \
      --input_filename vllm-outputs/${DEV_SET}/${MODEL_NAME}/t0.7_p0.95/seed${seed}/where/where_query.jsonl \
      --output_filename vllm-outputs/${DEV_SET}/${MODEL_NAME}/t0.7_p0.95/seed${seed}/where/where_query_output.jsonl \
      --max_tokens 128 \
      --temperature ${TEMPERATURE} \
      --chat
  done
done