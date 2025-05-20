batch_size=64
CUDA_VISIBLE_DEVICES=0 python -u src/expr/probe.py \
          --train_filename probe-outputs/universal_truthfulness/truthfulness_train/Llama-3.1-8B-Instruct/t0/output.jsonl \
          --test_filename probe-outputs/universal_truthfulness/truthfulness_test/Llama-3.1-8B-Instruct/t0/output.jsonl probe-outputs/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation/Llama-3.1-8B-Instruct/t0/llm_judge_results.jsonl probe-outputs/celebrity/celebrity_continuation/Llama-3.1-8B-Instruct/celebrity_test_continuation/Llama-3.1-8B-Instruct/t0/llm_judge_results.jsonl \
          --test_tag truthfulness_test wikidata_test_continuation celebrity_test_continuation \
          --activations_name hs_activations.h5 \
          --save_dir probe-outputs/universal_truthfulness/truthfulness_train/Llama-3.1-8B-Instruct/t0/best_singleprobe_wobias_known_results \
          --batch_size ${batch_size} \
          --lr 1e-3 \
          --weight_decay 1e-3 \
          --num_epochs 50 \
          --wandb_prefix known_truthfulness_train_Llama-3.1-8B-Instruct \
          --compute_grouped_acc \
          --compute_retraction_acc \
          --singlelinear \
          --num_layers 32
