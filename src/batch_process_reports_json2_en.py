import os
import csv
import json
import argparse
import hashlib
import time
import re
from collections import OrderedDict
from typing import List, Dict, Any, Iterable

from open_llm_model import OpenLLMModel

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# --- English KANON list ---
ID_TO_NAME_EN = [
    (1, "liver"),
    (2, "gallbladder"),
    (3, "spleen"),
    (4, "pancreas"),
    (5, "kidneys"),
    (6, "kidney cysts"),
    (7, "adrenal glands"),
    (8, "stomach"),
    (9, "duodenum"),
    (10, "small intestine"),
    (11, "colon"),
    (12, "lungs"),
    (13, "trachea"),
    (14, "esophagus"),
    (15, "heart"),
    (16, "aorta"),
    (17, "pulmonary veins"),
    (18, "brachiocephalic trunk"),
    (19, "subclavian artery"),
    (20, "common carotid artery"),
    (21, "brachiocephalic vein"),
    (22, "vena cava"),
    (23, "portal vein and splenic vein"),
    (24, "common iliac artery"),
    (25, "common iliac vein"),
    (26, "thyroid gland"),
    (27, "vertebrae"),
    (28, "sacrum"),
    (29, "ribs"),
    (30, "sternum"),
    (31, "costal cartilages"),
    (32, "brain"),
    (33, "clavicle"),
    (34, "scapula"),
    (35, "hip"),
    (36, "gluteus muscles"),
    (37, "iliopsoas muscle"),
    (38, "atrial appendage"),
    (39, "urinary bladder"),
    (40, "prostate"),
]
ID2NAME_EN: Dict[int, str] = dict(ID_TO_NAME_EN)
NAME2ID_EN: Dict[str, int] = {v: k for k, v in ID2NAME_EN.items()}

# --- Guided decoding schema (English, compact v2) ---
SCHEMA_COMPACT2_EN = {
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "Structure_ID": {"type": "integer"},
      "Structure_KANON": {"type": "string"},
      "Structure_Report": {"type": "string"},
      "Description": {"type": "string"},
      "State": {"type": "string", "enum": ["normal", "abnormal"]}
    },
    "required": ["Structure_ID", "Structure_KANON", "Structure_Report", "Description", "State"],
    "additionalProperties": False
  }
}

# ---------- Helpers ----------

def stable_shard_index(key: str, num_shards: int) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % max(1, num_shards)


def sanitize_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in {"-", "_"}) else "-" for c in s)[:255]


def build_chat_texts(model: OpenLLMModel, instruction: str, texts: List[str], user_prefix: str = "") -> List[str]:
    msgs_list = []
    for t in texts:
        content = f"{user_prefix}{t}" if user_prefix else t
        msgs_list.append([
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ])
    return [
        model.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in msgs_list
    ]


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
# Add a small helper for debug prints
def _dprint(enabled: bool, msg: str):
    if enabled:
        print(msg, flush=True)

# ---------- Parsing helpers ----------

def _slice_to_endjson(s: str, end_marker: str) -> str:
    if end_marker:
        j = s.find(end_marker)
        if j != -1:
            return s[:j]
    return s


def _sanitize_json_text(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("\ufeff", "").replace("\u00a0", " ")
    s = re.sub(r"^```[a-zA-Z]*\n", "", s)
    s = re.sub(r"\n```\s*$", "", s)
    # Keep from the first JSON start token: '[' for arrays or '{' for objects
    first_obj = s.find("{")
    first_arr = s.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if starts:
        s = s[min(starts):]
    # Drop trailing commas before closers
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Balance only when missing closers
    open_braces = s.count("{")
    close_braces = s.count("}")
    open_brackets = s.count("[")
    close_brackets = s.count("]")
    if open_brackets > close_brackets or open_braces > close_braces:
        s = s.rstrip()
        s = re.sub(r",\s*$", "", s)
        s += "]" * max(0, open_brackets - close_brackets)
        s += "}" * max(0, open_braces - close_braces)
    return s.strip()

def _normalize_quotes_for_json(s: str) -> str:
    # Extra fallback normalization used after parse failures.
    return (
        s.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"')
         .replace("’", "'").replace("‘", "'").replace("‚", "'")
    )

def _parse_model_json(response_text: str) -> Any:
    txt = _sanitize_json_text(response_text)
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        # Try with trailing-comma fix
        txt2 = re.sub(r",\s*([}\]])", r"\1", txt)
        try:
            obj = json.loads(txt2)
        except json.JSONDecodeError:
            # Fallback: normalize smart quotes
            qtxt2 = _normalize_quotes_for_json(txt2)
            try:
                obj = json.loads(qtxt2)
            except json.JSONDecodeError:
                qtxt = _normalize_quotes_for_json(txt)
                obj = json.loads(qtxt)
    # If the model returned a JSON string that itself contains JSON, parse once more
    if isinstance(obj, str):
        s = obj.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                obj = json.loads(s)
            except Exception:
                try:
                    obj = json.loads(_normalize_quotes_for_json(s))
                except Exception:
                    pass
    return obj

# ---------- CSV reading and formatting ----------

def read_records_csv_en(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def format_report_en(row: Dict[str, Any], cols: Dict[str, str]) -> str:
    parts: List[str] = []
    vol = row.get(cols["id"], "")
    if vol:
        parts.append(f"Volume: {vol}")
    ci = row.get(cols["clinical"], "")
    if ci:
        parts.append(f"Clinical Information: {ci}")
    tech = row.get(cols["technique"], "")
    if tech:
        parts.append(f"Technique: {tech}")
    find = row.get(cols["findings"], "")
    if find:
        parts.append(f"Findings: {find}")
    impr = row.get(cols["impressions"], "")
    if impr:
        parts.append(f"Impressions: {impr}")
    return "\n".join(parts)

def format_report_inspect(row: Dict[str, Any]) -> str:
    # INSPECT: only id + impressions
    rid = row.get("impression_id", "") or row.get("id", "")
    impr = row.get("impressions", "")
    s = []
    if rid:
        s.append(f"ID: {rid}")
    if impr:
        s.append(f"Impressions: {str(impr).lower()}")
    return "\n".join(s)

# ---------- Post validation ----------

def _post_validate_and_fix_compact_en(parsed_any: Any) -> List[Dict[str, Any]]:
    items = parsed_any if isinstance(parsed_any, list) else []
    cleaned: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        sid = it.get("Structure_ID")
        canon = it.get("Structure_KANON")
        struct_rep = it.get("Structure_Report", "")
        descr = it.get("Description", "")
        state = it.get("State", None)

        # map ID <-> name with canon
        if isinstance(sid, int) and sid in ID2NAME_EN:
            canon = ID2NAME_EN[sid]
        elif isinstance(canon, str) and canon in NAME2ID_EN:
            sid = NAME2ID_EN[canon]
            canon = canon
        else:
            continue  # skip if not in canon list

        item: Dict[str, Any] = {
            "Structure_ID": int(sid),
            "Structure_KANON": str(canon),
            "Structure_Report": str(struct_rep),
            "Description": str(descr),
        }
        if state is not None:
            item["State"] = state
        cleaned.append(item)
    return cleaned

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV file")
    ap.add_argument("--output", help="Path to append JSONL output (optional)")
    ap.add_argument("--output-dir", help="Directory to write one JSON per record (optional)")
    ap.add_argument("--model", default="llama-3.3-awq-kosbu-fp8kv")
    ap.add_argument("--instruction", help="Instruction text inline (optional)")
    ap.add_argument("--instruction-file", help="Path to instruction file", default="instructions_compact2_en.txt")
    # Input format only; ct-rate uses fixed 'VolumeName' as id
    ap.add_argument("--input-format", choices=["ct-rate", "inspect"], default="ct-rate",
                    help="CSV layout: 'ct-rate' (VolumeName + sections) or 'inspect' (impression_id + impressions)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--user-prefix", default="")
    ap.add_argument("--stop-text", default="END_JSON")
    ap.add_argument("--ids-file", help="Optional newline-separated list of ids to include")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--guided-json", action="store_true", help="Enable grammar-constrained decoding to force valid JSON (xgrammar backend)")
    # ct-rate column names (override if your CSV differs)
    ap.add_argument("--col-clinical", default="ClinicalInformation_EN")
    ap.add_argument("--col-technique", default="Technique_EN")
    ap.add_argument("--col-findings", default="Findings_EN")
    ap.add_argument("--col-impressions", default="Impressions_EN")
    # NEW: debug flag
    ap.add_argument("--debug", action="store_true", help="Print the exact prompt fed to the LLM and the raw LLM output; also write .prompt.txt and .raw.txt files next to JSON output when --output-dir is used")
    args = ap.parse_args()

    if not args.instruction and not args.instruction_file:
        raise SystemExit("Provide --instruction or --instruction-file")
    instruction = ""
    if args.instruction_file:
        with open(args.instruction_file, "r", encoding="utf-8") as f:
            instruction = f.read().strip()
    else:
        instruction = args.instruction or ""

    model = OpenLLMModel.create_model(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=max(8, args.batch_size),
        gpu_memory_utilization=args.gpu_mem,
    )

    rows = read_records_csv_en(args.input)

    records: List[Dict[str, Any]] = []
    if args.input_format == "inspect":
        # Fixed columns for INSPECT
        for i, r in enumerate(rows):
            rid = str(r.get("impression_id", r.get("id", i)))
            text = format_report_inspect(r)
            records.append({
                "id": rid,
                "text": text,
            })
    else:
        # ct-rate: hard-code VolumeName as id
        colmap = {
            "id": "VolumeName",
            "clinical": args.col_clinical,
            "technique": args.col_technique,
            "findings": args.col_findings,
            "impressions": args.col_impressions,
        }
        for i, r in enumerate(rows):
            rid = str(r.get("VolumeName", i))
            text = format_report_en(r, colmap)
            records.append({
                "id": rid,
                "text": text,
            })

    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            wanted = {line.strip() for line in f if line.strip()}
        records = [r for r in records if r["id"] in wanted]

    if args.num_shards and args.num_shards > 1:
        shard_records = [r for r in records if stable_shard_index(r["id"], args.num_shards) == args.shard_index]
        print(f"Sharding enabled: shard {args.shard_index}/{args.num_shards}, selected {len(shard_records)}", flush=True)
    else:
        shard_records = records
        print(f"No sharding: processing all {len(shard_records)} records", flush=True)

    os.makedirs(args.output_dir, exist_ok=True) if args.output_dir else None

    processed = 0
    start_time = time.time()

    # Replace batching loop to keep batches full when skipping existing outputs
    i = 0
    while i < len(shard_records):
        if args.output_dir and not args.overwrite:
            batch = []
            j = i
            while j < len(shard_records) and len(batch) < args.batch_size:
                rec = shard_records[j]
                j += 1
                out_path = os.path.join(args.output_dir, sanitize_filename(str(rec.get("id"))) + ".json")
                if os.path.exists(out_path):
                    continue  # skip already processed
                batch.append(rec)
            i = j  # advance cursor to where we stopped scanning
            if not batch:
                continue  # all skipped in this window; keep scanning
        else:
            batch = shard_records[i: i + args.batch_size]
            i += args.batch_size
            if not batch:
                break

        ids = [r["id"] for r in batch]
        texts = [r["text"] for r in batch]
        chat_texts = build_chat_texts(model, instruction, texts, user_prefix=args.user_prefix)

        # DEBUG: show exact prompts sent to the LLM
        if args.debug:
            for rid, prompt_text in zip(ids, chat_texts):
                _dprint(True, f"\n==== LLM INPUT (id={rid}) ====\n{prompt_text}\n==== END INPUT (id={rid}) ====\n")

        use_guidance = bool(args.guided_json)
        # Deterministic decoding for guided runs; keep user args otherwise.
        #temp = 0.0 if use_guidance else args.temperature
        #topp = 1.0 if use_guidance else args.top_p
        temp = args.temperature
        topp = args.top_p
        stop = None if use_guidance else ([args.stop_text] if args.stop_text else None)

        backend_gram = "xgrammar:disable-any-whitespace" if re.search(r"qwen", args.model, re.IGNORECASE) else "xgrammar"
        guided_schema = SCHEMA_COMPACT2_EN
        sp = SamplingParams(
            temperature=temp,
            top_p=topp,
            max_tokens=args.max_tokens,
            stop=stop,
            guided_decoding=GuidedDecodingParams(json=guided_schema, backend=backend_gram) if use_guidance else None,
        )

        if args.debug:
            _dprint(True, f"SamplingParams: {sp}")

        try:
            outs = model.llm.generate(chat_texts, sp)
        except Exception as e:
            # ...existing error handling...
            # (unchanged)
            for rid in ids:
                row = {"id": rid, "error": str(e)}
                if args.output_dir:
                    out_path = os.path.join(args.output_dir, sanitize_filename(rid) + ".json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, ensure_ascii=False)
                if args.output:
                    write_jsonl(args.output, [row])
            continue

        out_rows = []
        for rec, o, prompt_text in zip(batch, outs, chat_texts):
            # Exact raw text returned by the model (before any sanitization)
            model_text = o.outputs[0].text if (o and o.outputs) else ""
            if args.debug:
                _dprint(True, f"\n==== LLM OUTPUT (id={rec.get('id')}) ====\n{model_text}\n==== END OUTPUT (id={rec.get('id')}) ====\n")

            # Keep current pipeline behavior (strip <think> then parse)
            raw_text = re.sub(r"(?is)<think>.*?</think>", "", model_text)
            if not use_guidance:
                raw_text = _slice_to_endjson(raw_text, args.stop_text)

            parse_err = None
            try:
                parsed_any = _parse_model_json(raw_text)
            except Exception as pe:
                parse_err = str(pe)
                parsed_any = []

            cleaned_list = _post_validate_and_fix_compact_en(parsed_any)

            final = OrderedDict()
            final["id"] = rec.get("id")
            final["_parse_ok"] = parse_err is None
            final["abnormal_structures_en"] = cleaned_list
            if parse_err:
                final["_parse_error"] = parse_err
                final["_raw"] = raw_text
            out_rows.append(final)

            # Write sidecar debug files when output_dir is set
            if args.debug and args.output_dir:
                base = os.path.join(args.output_dir, sanitize_filename(str(rec.get("id") or "unknown")))
                try:
                    with open(base + ".prompt.txt", "w", encoding="utf-8") as pf:
                        pf.write(prompt_text)
                    with open(base + ".raw.txt", "w", encoding="utf-8") as rf:
                        rf.write(model_text)
                except Exception as _e:
                    _dprint(True, f"[debug] Failed to write sidecar files for id={rec.get('id')}: {_e}")

        if args.output_dir:
            for row in out_rows:
                out_path = os.path.join(args.output_dir, sanitize_filename(str(row.get("id") or "unknown")) + ".json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(row, f, ensure_ascii=False, indent=2)
        if args.output:
            write_jsonl(args.output, out_rows)

        processed += len(out_rows)
        if processed % (args.batch_size * 10) == 0:
            elapsed = time.time() - start_time
            print(f"Processed {processed} items in shard {args.shard_index}/{args.num_shards} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
