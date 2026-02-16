import os
import json
import argparse
import re
import time
from typing import List, Dict, Any, Iterable, Optional
from pathlib import Path

from open_llm_model import OpenLLMModel


# ---------- IO helpers ----------

def list_json_files(input_dir: str) -> List[str]:
    files = []
    for name in os.listdir(input_dir):
        if not name.lower().endswith(".json"):
            continue
        p = os.path.join(input_dir, name)
        if os.path.isfile(p):
            files.append(p)
    files.sort()
    return files


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


def stable_shard_index(key: str, num_shards: int) -> int:
    import hashlib
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % max(1, num_shards)


# ---------- Parsing helpers (defensive, though guided decoding should suffice) ----------

def _sanitize_json_text(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("\ufeff", "").replace("\u00a0", " ")
    s = re.sub(r"^```[a-zA-Z]*\n", "", s)
    s = re.sub(r"\n```\s*$", "", s)
    first_obj = s.find("{")
    first_arr = s.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if starts:
        s = s[min(starts):]
    s = re.sub(r",\s*([}\]])", r"\1", s)
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


def _parse_model_json(response_text: str) -> Any:
    txt = _sanitize_json_text(response_text)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        txt2 = re.sub(r",\s*([}\]])", r"\1", txt)
        return json.loads(txt2)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Verify disputed structures (English) and write accnr + kept IDs to JSONL.")
    ap.add_argument("--input-dir", required=True, help="Directory with dispute-pack JSON files (one per accnr)")
    ap.add_argument("--output-jsonl", required=True, help="Output JSONL path; one row per accnr: {accnr, ids}")
    ap.add_argument("--instruction-file", default=None, help="Verifier instruction file (auto: DE or EN)")
    ap.add_argument("--ctrate", action="store_true", help="Use English (ctrate) mode; default is German")
    ap.add_argument("--model", default="llama-3.3-awq-kosbu-fp8kv", help="Model preset key")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="Do not attempt to skip rows already present in output JSONL")
    ap.add_argument("--debug", action="store_true", help="Print exact prompts and model outputs; forces --batch-size 1")
    ap.add_argument("--correct-na", action="store_true",
                    help="Re-run only rows with ids == 'NA' in existing JSONL and merge updates")
    args = ap.parse_args()

    # Force batch size 1 in debug mode (before model init)
    if args.debug:
        args.batch_size = 1

    with open(args.instruction_file, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    # Build model with guided decoding for list of integers
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    SCHEMA_ID_LIST = {
        "type": "array",
        "items": {"type": "integer"},
        "additionalProperties": False,
    }

#    backend_gram = "xgrammar:disable-any-whitespace" if re.search(r"qwen", args.model, re.IGNORECASE) else "xgrammar"
    backend_gram = "xgrammar:disable-any-whitespace" if re.search(r"qwen25", args.model, re.IGNORECASE) else "xgrammar"
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        guided_decoding=GuidedDecodingParams(json=SCHEMA_ID_LIST, backend=backend_gram),
    )

    model = OpenLLMModel.create_model(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=max(8, args.batch_size),
        gpu_memory_utilization=args.gpu_mem,
    )

    # scan existing JSONL
    existing = set()
    na_targets = set()
    if os.path.exists(args.output_jsonl):
        try:
            with open(args.output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    acc = str(row.get("accnr") or "")
                    if not acc:
                        continue
                    existing.add(acc)
                    if args.correct_na and str(row.get("ids")) == "NA":
                        na_targets.add(acc)
        except Exception:
            pass
        if args.correct_na:
            print(f"Correct-NA mode: found {len(na_targets)} NA targets in existing JSONL.", flush=True)
        elif existing:
            print(f"Resume mode: {len(existing)} rows already in output; will skip.", flush=True)

    # list files and sharding (keep as is)
    files = list_json_files(args.input_dir)
    if args.num_shards and args.num_shards > 1:
        sharded = []
        for p in files:
            acc = os.path.basename(p).rsplit(".", 1)[0]
            if stable_shard_index(acc, args.num_shards) == args.shard_index:
                sharded.append(p)
        print(f"Sharding enabled: shard {args.shard_index}/{args.num_shards}, selected {len(sharded)} of {len(files)} files", flush=True)
        files = sharded
    else:
        print(f"No sharding: processing all {len(files)} files", flush=True)

    # Optional resume: build a set of accnrs already present in output JSONL (only if not overwriting)
    existing = set()
    if (not args.overwrite) and os.path.exists(args.output_jsonl):
        try:
            with open(args.output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        a = row.get("accnr")
                        if a:
                            existing.add(str(a))
                    except Exception:
                        continue
            if existing:
                print(f"Resume mode: {len(existing)} rows already in output; will skip.", flush=True)
        except Exception:
            pass

    t0 = time.time()
    total = 0
    # buffer rows when --correct-na so we can merge after the run
    all_rows: List[Dict[str, Any]] = []

    for i in range(0, len(files), args.batch_size):
        batch_paths = files[i : i + args.batch_size]

        accnrs: List[str] = []
        user_payloads: List[str] = []
        skip_mask: List[bool] = []
        error_mask: List[bool] = []

        for p in batch_paths:
            base = os.path.basename(p)
            acc = base.rsplit(".", 1)[0]
            if (not args.overwrite) and (not args.correct_na) and (acc in existing):
                if args.debug:
                    print(f"[DEBUG] Skipping existing accnr={acc}", flush=True)
                continue
            if args.correct_na and acc not in na_targets:
                # only re-run NA targets
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                if args.debug:
                    print(f"[DEBUG] Load error for accnr={acc}; will skip.", flush=True)
                accnrs.append(acc)
                user_payloads.append("[]")
                skip_mask.append(True)
                error_mask.append(True)
                continue

            # language-specific array extraction with fallback
            if args.ctrate:
                arr = obj.get("abnormal_structures_en")
                if arr is None:
                    arr = obj.get("disputed_structures")
                if arr is None:
                    arr = []
            else:
                arr = obj.get("abnormal_structures_compact") or []

            accnrs.append(str(obj.get("accnr") or acc))
            if not arr:
                if args.debug:
                    print(f"[DEBUG] No payload for accnr={acc}; skipping LLM call.", flush=True)
                user_payloads.append("[]")
                skip_mask.append(True)
                error_mask.append(False)
            else:
                # Only pass the expected fields to the verifier (DE or EN)
                cleaned = []
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    if args.ctrate:
                        sd = {
                            "Structure_ID": it.get("Structure_ID") if "Structure_ID" in it else it.get("Struktur_ID"),
                            "Structure_KANON": it.get("Structure_KANON") if "Structure_KANON" in it else it.get("Struktur_KANON"),
                            "Descriptions": it.get("Descriptions") or it.get("Beschreibungen") or [],
                        }
                    else:
                        sd = {
                            "Struktur_ID": it.get("Struktur_ID") if "Struktur_ID" in it else it.get("Structure_ID"),
                            "Struktur_KANON": it.get("Struktur_KANON") if "Struktur_KANON" in it else it.get("Structure_KANON"),
                            "Beschreibungen": it.get("Beschreibungen") or it.get("Descriptions") or [],
                        }
                    cleaned.append(sd)
                user_payloads.append(json.dumps(cleaned, ensure_ascii=False))
                skip_mask.append(False)
                error_mask.append(False)

        if not accnrs:
            continue

        # Build chats
        msgs_list = []
        for txt in user_payloads:
            msgs_list.append([
                {"role": "system", "content": instruction},
                {"role": "user", "content": txt},
            ])
        chat_texts = [
            model.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in msgs_list
        ]

        # Indices that will be sent to LLM
        results: List[Optional[List[int]]] = [None] * len(accnrs)
        need_indices = [idx for idx, skip in enumerate(skip_mask) if not skip]

        # Debug: print exact prompts
        if args.debug:
            for k, idx in enumerate(need_indices):
                print("====== DEBUG REQUEST BEGIN ======", flush=True)
                print(f"[accnr]={accnrs[idx]}", flush=True)
                print(chat_texts[idx], flush=True)
                print("====== DEBUG REQUEST END ========\n", flush=True)

        try:
            if need_indices:
                inputs = [chat_texts[idx] for idx in need_indices]
                outs = model.llm.generate(inputs, sp)
            else:
                outs = []
        except Exception as e:
            if args.debug:
                print(f"[DEBUG] LLM generate error: {e}", flush=True)
            # LLM/batch failure => mark as error (NA)
            for idx in need_indices:
                results[idx] = None
        else:
            for k, idx in enumerate(need_indices):
                o = outs[k]
                raw_text = o.outputs[0].text if (o and o.outputs) else ""
                # Debug: print raw output before parsing
                if args.debug:
                    print("====== DEBUG OUTPUT BEGIN =======", flush=True)
                    print(f"[accnr]={accnrs[idx]}", flush=True)
                    print(raw_text, flush=True)
                    print("====== DEBUG OUTPUT END =========\n", flush=True)
                raw_text = re.sub(r"(?is)<think>.*?</think>", "", raw_text)
                try:
                    parsed = _parse_model_json(raw_text)
                    if isinstance(parsed, list):
                        ids = [int(x) for x in parsed if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit())]
                    else:
                        ids = None
                except Exception:
                    ids = None
                results[idx] = ids

        # Fill skipped: distinguish valid-empty vs error
        for idx, skip in enumerate(skip_mask):
            if skip:
                results[idx] = None if error_mask[idx] else []

        rows = []
        for acc, ids in zip(accnrs, results):
            # ids == None means error => write "NA"
            out_ids = "NA" if ids is None else (ids or [])
            rows.append({"accnr": acc, "ids": out_ids})

        if args.correct_na:
            all_rows.extend(rows)
        else:
            write_jsonl(args.output_jsonl, rows)
        total += len(rows)

        if total % (args.batch_size * 10) == 0:
            elapsed = time.time() - t0
            print(f"Processed {total} items so far in {elapsed:.1f}s", flush=True)

    if args.correct_na:
        merge_correct_na(Path(args.output_jsonl), all_rows)
        print(f"Done. Merged {len(all_rows)} corrected rows into {args.output_jsonl}", flush=True)
    else:
        elapsed = time.time() - t0
        print(f"Done. Wrote {total} rows to {args.output_jsonl} in {elapsed:.1f}s", flush=True)


def merge_correct_na(base_path: Path, new_rows: List[Dict[str, Any]]):
    """Replace only ids=='NA' rows with non-NA updates; keep order; append new accnr if missing."""
    if not base_path.exists():
        write_jsonl(str(base_path), new_rows)
        return
    base_objs = []
    idx_by_acc = {}
    with base_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            acc = str(obj.get("accnr") or "")
            if acc and acc not in idx_by_acc:
                idx_by_acc[acc] = len(base_objs)
                base_objs.append(obj)
    new_by_acc = {}
    for r in new_rows:
        acc = str(r.get("accnr") or "")
        if acc:
            new_by_acc[acc] = r
    for acc, pos in idx_by_acc.items():
        if acc in new_by_acc:
            old_ids = base_objs[pos].get("ids")
            new_ids = new_by_acc[acc].get("ids")
            if str(old_ids) == "NA" and new_ids is not None and str(new_ids) != "NA":
                base_objs[pos] = new_by_acc[acc]
    for acc, row in new_by_acc.items():
        if acc not in idx_by_acc:
            base_objs.append(row)
    tmp = base_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for obj in base_objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(base_path)


if __name__ == "__main__":
    main()
