# medgemma 
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION
OUTDIR="output_medgem"
INPUT="reports_ctrate.csv"
INSTR_FILE="../prompts/instruct_extract_en.txt"

python batch_process_reports_json2_en.py \
   --input "$INPUT" \
   --output-dir "$OUTDIR" \
   --input-format ct-rate \
   --model medgemma \
   --instruction-file "$INSTR_FILE" \
   --batch-size 5 \
   --temperature 0 \
   --top-p 1 \
   --gpu-mem 0.92 \
   --guided-json 

INPUT="reports_inspect.csv"
python batch_process_reports_json2_en.py \
   --input "$INPUT" \
   --output-dir "$OUTDIR" \
   --input-format inspect \
   --model medgemma \
   --instruction-file "$INSTR_FILE" \
   --batch-size 5 \
   --temperature 0 \
   --top-p 1 \
   --gpu-mem 0.92 \
   --guided-json 
