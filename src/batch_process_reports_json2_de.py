import os
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

ID_TO_NAME = [
    (1,"Leber"), (2,"Gallenblase"), (3,"Milz"), (4,"Pankreas"), (5,"Niere"),
    (6,"Nierenzyste"), (7,"Nebenniere"), (8,"Magen"), (9,"Duodenum"),
    (10,"Dünndarm"), (11,"Kolon"), (12,"Lunge"), (13,"Trachea"),
    (14,"Ösophagus"), (15,"Herz"), (16,"Aorta"), (17,"Pulmonalvenen"),
    (18,"Truncus brachiocephalicus"), (19,"Arteria subclavia"),
    (20,"Arteria carotis communis"), (21,"Vena brachiocephalica"),
    (22,"Vena cava"), (23,"Vena portae und Vena splenica"),
    (24,"Arteria iliaca communis"), (25,"Vena iliaca communis"),
    (26,"Schilddrüse"), (27,"Wirbelkörper"), (28,"Os sacrum"), (29,"Rippen"),
    (30,"Sternum"), (31,"Costalknorpel"), (32,"Gehirn"), (33,"Clavicula"),
    (34,"Scapula"), (35,"Hüfte"), (36,"Glutealmuskeln"),
    (37,"Musculus iliopsoas"), (38,"Vorhofohr"), (39,"Harnblase"), (40,"Prostata"),
]
ID2NAME: Dict[int, str] = dict(ID_TO_NAME)
NAME2ID: Dict[str, int] = {v: k for k, v in ID2NAME.items()}

# --- xgrammar-friendly schema (no oneOf/const/min/max) ---
SCHEMA_FULL = {
  "type": "object",
  "properties": {
    "abnormal_structures": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "structure_id": {"type": "integer"},
          "structure": {"type": "string"},
          "finding": {"type": "string"}
        },
        "required": ["structure_id","structure","finding"],
        "additionalProperties": False
      }
    },
    "diagnoses": {
      "type": "array",
      "items": {
        "type":"object",
        "properties": {
          "diagnosis": {"type":"string"},
          "certainty": {"type":"integer"},
          "inferred": {"type":"boolean"}
        },
        "required": ["diagnosis","certainty","inferred"],
        "additionalProperties": False
      }
    },
    "contrast_agent_used": {"type":"string"}
  },
  "required": ["abnormal_structures","diagnoses","contrast_agent_used"],
  "additionalProperties": False
}

# New compact schema (root is an array)
# SCHEMA_COMPACT = {
#   "type": "array",
#   "items": {
#     "type": "object",
#     "properties": {
#       "Struktur_ID": {"type": "integer"},
#       "Struktur_KANON": {"type": "string"},
#       "Struktur_Befund": {"type": "string"},
#       "Beschreibung": {"type": "string"}
#     },
#     "required": ["Struktur_ID", "Struktur_KANON", "Struktur_Befund", "Beschreibung"],
#     "additionalProperties": False
#   }
# }
SCHEMA_COMPACT = {
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "Struktur_ID": {"type": "integer"},
      "Struktur_KANON": {"type": "string", "pattern": "^[^\\n\\r]*$"},
      "Struktur_Befund": {"type": "string", "pattern": "^[^\\n\\r]*$"},
      "Beschreibung": {"type": "string", "pattern": "^[^\\n\\r]*$"}
    },
    "required": ["Struktur_ID", "Struktur_KANON", "Struktur_Befund", "Beschreibung"],
    "additionalProperties": False
  }
}

# # New compact schema v2 (adds Zustand with enum)
# SCHEMA_COMPACT_V2 = {
#   "type": "array",
#   "items": {
#     "type": "object",
#     "properties": {
#       "Struktur_ID": {"type": "integer"},
#       "Struktur_KANON": {"type": "string"},
#       "Struktur_Befund": {"type": "string"},
#       "Beschreibung": {"type": "string"},
#       "Zustand": {"type": "string", "enum": ["normal", "abnormal"]}
#     },
#     "required": ["Struktur_ID", "Struktur_KANON", "Struktur_Befund", "Beschreibung", "Zustand"],
#     "additionalProperties": False
#   }
# }

# Compact schema v2, avoids unescaped control characters (tabs, newlines, etc.) 
SCHEMA_COMPACT_V2 = {
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "Struktur_ID": {"type": "integer"},
      "Struktur_KANON": {"type": "string", "pattern": "^[^\\t\\n\\r]*$"},
      "Struktur_Befund": {"type": "string", "pattern": "^[^\\t\\n\\r]*$"},
      "Beschreibung": {"type": "string", "pattern": "^[^\\t\\n\\r]*$"},
      "Zustand": {"type": "string", "enum": ["normal", "abnormal"]}
    },
    "required": ["Struktur_ID", "Struktur_KANON", "Struktur_Befund", "Beschreibung", "Zustand"],
    "additionalProperties": False
  }
}

SCHEMA_COMPACT_V2_qwen = {
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "Struktur_ID": {"type": "integer"},
      "Struktur_KANON": {"type": "string"},
      "Struktur_Befund": {"type": "string"},
      "Beschreibung": {"type": "string"},
      "Zustand": {"type": "string", "enum": ["normal", "abnormal"]}
    },
    "required": ["Struktur_ID", "Struktur_KANON", "Struktur_Befund", "Beschreibung", "Zustand"],
    "additionalProperties": False
  }
}

CONTRAST_NORMALIZE = {
    "ja": "Ja", "j": "Ja", "yes": "Ja", "y": "Ja", "true": "Ja",
    "nein": "Nein", "n": "Nein", "no": "Nein", "false": "Nein",
    "unklar": "Unklar", "unbekannt": "Unklar", "unknown": "Unklar", "na": "Unklar", "n/a": "Unklar"
}
ALLOWED_CONTRAST = {"Ja","Nein","Unklar"}

def _normalize_contrast(val: Any) -> str:
    if isinstance(val, str):
        key = val.strip().lower()
        return CONTRAST_NORMALIZE.get(key, "Unklar")
    return "Unklar"

def _post_validate_and_fix(parsed: Dict[str, Any]) -> Dict[str, Any]:
    # ensure required top-level keys exist
    parsed.setdefault("abnormal_structures", [])
    parsed.setdefault("diagnoses", [])
    parsed["contrast_agent_used"] = _normalize_contrast(parsed.get("contrast_agent_used"))

    # fix structure_id/structure mapping; drop invalid items quietly
    cleaned_structs = []
    for s in parsed.get("abnormal_structures", []):
        sid = s.get("structure_id")
        name = s.get("structure")
        finding = s.get("finding", "")
        if isinstance(sid, int) and sid in ID2NAME:
            # correct the name to the canon name if mismatched
            s["structure"] = ID2NAME[sid]
            cleaned_structs.append({"structure_id": sid, "structure": s["structure"], "finding": str(finding)})
        elif isinstance(name, str) and name in NAME2ID:
            # fill in id from name
            s["structure_id"] = NAME2ID[name]
            s["structure"] = name
            cleaned_structs.append({"structure_id": s["structure_id"], "structure": name, "finding": str(finding)})
        # else: skip if neither sid nor name maps to canon set
    parsed["abnormal_structures"] = cleaned_structs

    # clamp certainty and coerce types
    cleaned_dx = []
    for d in parsed.get("diagnoses", []):
        diag = str(d.get("diagnosis", "")).strip()
        # coerce certainty to int and clamp 1..5
        c = d.get("certainty", 3)
        try:
            c = int(c)
        except Exception:
            c = 3
        c = 1 if c < 1 else 5 if c > 5 else c
        inf = bool(d.get("inferred", False))
        if diag:
            cleaned_dx.append({"diagnosis": diag, "certainty": c, "inferred": inf})
    parsed["diagnoses"] = cleaned_dx

    if parsed["contrast_agent_used"] not in ALLOWED_CONTRAST:
        parsed["contrast_agent_used"] = "Unklar"
    return parsed

# New: compact post-fix that enforces canon ID/name and basic type cleanup
def _post_validate_and_fix_compact(parsed_any: Any) -> List[Dict[str, Any]]:
    items = parsed_any if isinstance(parsed_any, list) else []
    cleaned: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        sid = it.get("Struktur_ID")
        canon = it.get("Struktur_KANON")
        befund = it.get("Struktur_Befund", "")
        beschr = it.get("Beschreibung", "")
        zustand = it.get("Zustand", None)  # pass-through if present

        # map ID <-> name with canon
        if isinstance(sid, int) and sid in ID2NAME:
            canon = ID2NAME[sid]
        elif isinstance(canon, str) and canon in NAME2ID:
            sid = NAME2ID[canon]
            canon = canon
        else:
            continue  # skip if not in canon list

        item = {
            "Struktur_ID": int(sid),
            "Struktur_KANON": str(canon),
            "Struktur_Befund": str(befund),
            "Beschreibung": str(beschr),
        }
        if zustand is not None:
            item["Zustand"] = zustand  # no normalization, no filtering
        cleaned.append(item)
    return cleaned

def read_records_full(path: str) -> List[Dict[str, Any]]:
    """Read JSON array or JSONL and return list of full objects."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            data: List[Dict[str, Any]] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
            return data

def format_report_for_model_like_example(obj: Dict[str, Any]) -> str:
    """Build the report text exactly like example_report.format_report_for_model()."""
    fields: List[str] = []
    if "klinische_angaben" in obj:
        fields.append(f"Klinische Angaben: {obj['klinische_angaben']}")
    if "fragestellung" in obj:
        fields.append(f"Fragestellung: {obj['fragestellung']}")
    if "befund" in obj:
        fields.append(f"Befund: {obj['befund']}")
    if "beurteilung" in obj:
        fields.append(f"Beurteilung: {obj['beurteilung']}")
    if "station" in obj:
        fields.append(f"station: {obj['station']}")
    if "untersuchung" in obj:
        fields.append(f"untersuchung: {obj['untersuchung']}")
    return "\n".join(fields)

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

# ---------- Parsing helpers ----------

def _slice_to_endjson(s: str, end_marker: str) -> str:
    if end_marker:
        j = s.find(end_marker)
        if j != -1:
            return s[:j]
    return s

def _sanitize_json_text(s: str) -> str:
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
    # Used only after a parse failure: convert smart quotes to ASCII.
    return (
        s.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"')
         .replace("’", "'").replace("‘", "'").replace("‚", "'")
    )

def _parse_model_json(response_text: str) -> Any:
    txt = _sanitize_json_text(response_text)
    # Try as-is
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
                # Fallback for inner JSON that uses smart quotes
                try:
                    obj = json.loads(_normalize_quotes_for_json(s))
                except Exception:
                    pass
    return obj

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output")
    ap.add_argument("--output-dir")
    ap.add_argument("--model", default="llama-3.3-awq-kosbu-fp8kv")
    ap.add_argument("--instruction")
    ap.add_argument("--instruction-file")
    ap.add_argument("--id-field", default="accnr")
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
    ap.add_argument("--ids-file")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--guided-json", action="store_true",
                   help="Enable grammar-constrained decoding to force valid JSON (xgrammar backend)")
    # New: choose schema/output mode
    ap.add_argument("--schema-type", choices=["full", "compact", "compact2"], default="full",
                   help="Select JSON schema: 'full', 'compact' (no Zustand), or 'compact2' (with Zustand)")
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

    full = read_records_full(args.input)
    all_records = []
    for i, obj in enumerate(full):
        rid = str(obj.get(args.id_field, i))
        text = format_report_for_model_like_example(obj)
        all_records.append({"id": rid, "text": text, "patid": obj.get("patid"), "accnr": obj.get("accnr")})

    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            wanted = {line.strip() for line in f if line.strip()}
        all_records = [r for r in all_records if r["id"] in wanted or (r.get("accnr") and str(r.get("accnr")) in wanted)]

    if args.num_shards and args.num_shards > 1:
        records = [r for r in all_records if stable_shard_index(r["id"], args.num_shards) == args.shard_index]
        print(f"Sharding enabled: shard {args.shard_index}/{args.num_shards}, selected {len(records)}", flush=True)
    else:
        records = all_records
        print(f"No sharding: processing all {len(records)} records", flush=True)

    os.makedirs(args.output_dir, exist_ok=True) if args.output_dir else None

    processed = 0
    start_time = time.time()

    # REPLACED batching loop to keep batches full when skipping existing outputs
    i = 0
    while i < len(records):
        if args.output_dir and not args.overwrite:
            batch = []
            j = i
            while j < len(records) and len(batch) < args.batch_size:
                rec = records[j]
                j += 1
                key = rec.get("accnr") or rec.get("id")
                out_path = os.path.join(args.output_dir, sanitize_filename(str(key)) + ".json")
                if os.path.exists(out_path):
                    continue  # skip already processed
                batch.append(rec)
            i = j  # advance cursor to where we stopped scanning
            if not batch:
                continue  # all skipped in this window; keep scanning
        else:
            batch = records[i: i + args.batch_size]
            i += args.batch_size
            if not batch:
                break

        ids = [r["id"] for r in batch]
        texts = [r["text"] for r in batch]
        chat_texts = build_chat_texts(model, instruction, texts, user_prefix=args.user_prefix)

        use_guidance = bool(args.guided_json)
        # Deterministic decoding for guided runs; keep user args otherwise.
        temp = 0.0 if use_guidance else args.temperature
        topp = 1.0 if use_guidance else args.top_p
        stop = None if use_guidance else ([args.stop_text] if args.stop_text else None)

        is_qwen = bool(re.search(r"qwen3", args.model, re.IGNORECASE))
        is_bio = bool(re.search(r"openbio", args.model, re.IGNORECASE))
        backend_gram = "xgrammar:disable-any-whitespace" if is_qwen else "xgrammar"

        if args.schema_type == "compact2":
            guided_schema = SCHEMA_COMPACT_V2_qwen if (is_qwen or is_bio) else SCHEMA_COMPACT_V2
        elif args.schema_type == "compact":
            guided_schema = SCHEMA_COMPACT
        else:
            guided_schema = SCHEMA_FULL

        sp = SamplingParams(
            temperature=temp,
            top_p=topp,
            max_tokens=args.max_tokens,
            stop=stop,
            guided_decoding=GuidedDecodingParams(json=guided_schema, backend=backend_gram) if use_guidance else None,
        )

        try:
            outs = model.llm.generate(chat_texts, sp)
        except Exception as e:
            for rid in ids:
                row = {"id": rid, "error": str(e)}
                if args.output_dir:
                    out_path = os.path.join(args.output_dir, sanitize_filename(rid) + ".json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, ensure_ascii=False, indent=2)
                if args.output:
                    write_jsonl(args.output, [row])
            continue

        rows = []
        for rec, o in zip(batch, outs):
            raw_text = o.outputs[0].text if (o and o.outputs) else ""
            raw_text = re.sub(r"(?is)<think>.*?</think>", "", raw_text)
            if not use_guidance:
                raw_text = _slice_to_endjson(raw_text, args.stop_text)

            parse_err = None
            try:
                parsed_any = _parse_model_json(raw_text)
            except Exception as pe:
                parse_err = str(pe)
                parsed_any = None if args.schema_type in ("compact", "compact2") else {"abnormal_structures": [], "diagnoses": [], "contrast_agent_used": "Unklar"}

            final = OrderedDict()
            final["patid"] = rec.get("patid")
            final["accnr"] = rec.get("accnr") or rec["id"]
            final["_parse_ok"] = parse_err is None  # NEW

            if args.schema_type in ("compact", "compact2"):
                if parsed_any is None:
                    final["abnormal_structures_compact"] = "NA"
                else:                
                    cleaned_list = _post_validate_and_fix_compact(parsed_any)
                    final["abnormal_structures_compact"] = cleaned_list
                if parse_err:
                    final["_parse_error"] = parse_err
                    final["_raw"] = raw_text
            else:
                parsed = _post_validate_and_fix(parsed_any if isinstance(parsed_any, dict) else {})
                for k, v in parsed.items():
                    if k not in ("patid", "accnr"):
                        final[k] = v
                if parse_err:
                    final["_parse_error"] = parse_err
                    final["_raw"] = raw_text
            rows.append(final)

        if args.output_dir:
            for row in rows:
                out_path = os.path.join(args.output_dir, sanitize_filename(str(row.get("accnr") or "unknown")) + ".json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(row, f, ensure_ascii=False, indent=2) 
        if args.output:
            write_jsonl(args.output, rows)

        processed += len(rows)
        if processed % (args.batch_size * 10) == 0:
            elapsed = time.time() - start_time
            print(f"Processed {processed} items in shard {args.shard_index}/{args.num_shards} in {elapsed:.1f}s", flush=True)

if __name__ == "__main__":
    main()
