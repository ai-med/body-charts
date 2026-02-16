INSTR_FILE="../prompts/instruct_verify_en.txt"

export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION

python batch_process_verify.py \
  --input-dir disputed_summaries \
  --output-jsonl verified_ids_medgem.jsonl \
  --instruction-file "$INSTR_FILE" \
  --model medgemma \
  --batch-size 6 \
  --max-tokens 150 \
  --max-model-len 7000 \
  --correct-na
