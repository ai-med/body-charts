# Body charts from CT across adulthood


![Body charts logo](images/overview.png)

This repository accompanies our work on whole-body **CT reference charts** (organ volume and tissue attenuation in Hounsfield Units) across adulthood, built from routine clinical CT using automated multi-organ segmentation and evidence-grounded report filtering.

## Interactive reference charts
- https://ai-med.de/body-charts-v2/

## Evidence-grounded, cross-model LLM filtering


### LLM report filtering (batch scripts)

The folder `src/` includes lightweight **bash wrappers** and **Python entry points** that were used to run the evidence-grounded, cross-model LLM filtering at scale.

- `run_batch_extract.sh`  
  Runs the **extraction stage** (structured JSON summaries) with `vLLM` using guided decoding. Model selection is controlled via `--model` (e.g., `medgemma`), and prompts are provided via `--instruction-file` (e.g., `prompts/instruct_extract_en.txt`).

- `run_batch_verify.sh`  
  Runs the **verification stage** on disputed cases (e.g., `disputed_summaries/`) and produces a JSONL file of verified decisions. This corresponds to the second-stage cross-checking of extracted findings using a separate verification prompt (e.g., `prompts/instruct_verify_en.txt`).

- `batch_process_reports_json2_en.py`  
  Python driver for the extraction stage. 

- `batch_process_verify.py`  
  Python driver for the verification stage. 


### Core LLM inference & model tooling
- vllm==0.8.5.post1
- torch==2.6.0
- torchvision==0.21.0+cu126
- torchaudio==2.6.0
- transformers==4.52.4
- accelerate==1.5.2
- safetensors==0.5.3
- tokenizers==0.21.1
- huggingface-hub==0.32.3
- xgrammar==0.1.18

### Quantization / memory-efficient inference (optional, depending on models)
- bitsandbytes==0.45.3
- xformers==0.0.29.post2
- flash-attn==2.7.3
- cupy-cuda12x==13.4.1




## GAMLSS model fitting (minimal reference implementation)

The folder [`gamlss/`](./gamlss) contains two GAMLSS model-fitting routines used in this work. The code is intended to document the core modeling algorithm (distributional regression with candidate fractional-polynomial search and information-criterion selection).

### Included scripts

- `fit_gamlss_model_volume.R`  
  Fits organ **volume** reference models using GAMLSS (default family: `GG`).  
  Candidate models are defined by a small grid of fractional-polynomial (FP) orders for `mu` and `sigma` (with `nu` held constant). The best candidate is selected by **AIC** and **BIC**, and “final” models are refit using the implied FP powers expressed via `bfp(Age, powers=...)`. The model adjusts for `Sex`, `Manufacturer`, `Contrast` and includes a random `Study` effect (optional `kvp`).

- `fit_gamlss_model_ST1.R`  
  Fits **attenuation (HU)** models using the `ST1` family.  
  As above, the routine evaluates a small set of FP candidates, selects the BIC-best specification, and refits a final model using `bfp()` powers. The model adjusts for `Sex`, `Manufacturer`, includes a random `Study` effect, and can optionally include `kvp`.



