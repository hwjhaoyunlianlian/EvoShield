# EvoShield

EvoShield is a confidence-gated, attack-type-specific defense for mitigating LLM jailbreaks. The code reproduces the main experimental pipelines reported in Tables 3-6.

## Quick Start

Install the required packages:

```bash
pip install numpy torch transformers sentence-transformers accelerate
```

Build the router:

```bash
python -m evoshield.cli build --config your_config.json --registered-data your_train_data.jsonl --knowledge your_knowledge.json --fallback-prompt your_fallback_prompt.txt --output your_artifacts
```

Inspect one routing result:

```bash
python -m evoshield.cli route --artifacts your_artifacts --query "your query"
```

## Data Format

Use JSON Lines (`.jsonl`) for attack data. Each line should follow this format:

```json
{"sample_id":"sample_001","base_request_id":"base_001","base_request":"original request","attack_prompt":"generated jailbreak prompt","attack_type":"PAIR","target_model":"Llama-3-8B","split":"test"}
```

Registered training data may also use the shorter format:

```json
{"text":"generated jailbreak prompt","label":"PAIR"}
```

The registered attack types are `PAP`, `PAIR`, `ECLIPSE`, `ADAPTIVE`, `DRL`, and `GCG`.

## Configuration

Example `your_config.json`:

```json
{
  "encoder": {
    "type": "sentence_transformer",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": null,
    "device": null,
    "batch_size": 32,
    "max_seq_length": 512
  },
  "router": {
    "alpha": 0.8,
    "k": 7,
    "gamma_c": 0.3203,
    "gamma_p": 0.8031,
    "epsilon": 1e-8,
    "epochs": 300,
    "learning_rate": 0.15,
    "l2": 0.0001,
    "seed": 42
  }
}
```

Example `your_model_config.json`:

```json
{
  "type": "transformers",
  "model_name": "your/model-name",
  "revision": null,
  "device_map": "auto",
  "torch_dtype": "bfloat16",
  "max_new_tokens": 512,
  "temperature": 0.0,
  "top_p": 1.0
}
```

## Running Instructions

### Table 3: defense results

```bash
python -m evoshield.experiments defense --artifacts your_artifacts --data your_defense_data.jsonl --backend-config your_model_config.json --judge-type harmbench --judge-config your_judge_config.json --output table3_raw.csv

python -m evoshield.experiments summarize --input table3_raw.csv --output table3.csv
```

### Table 4: routing accuracy

```bash
python -m evoshield.experiments routing --artifacts your_artifacts --data your_routing_data.jsonl --output table4.csv
```

### Table 5: unregistered attacks

```bash
python -m evoshield.experiments leave-one-out --config your_config.json --data your_leave_one_out_data.jsonl --output table5.csv
```

### Table 6: few-shot registration

```bash
python -m evoshield.experiments registration --config your_config.json --data your_registration_data.jsonl --support-sizes 1,5,10,20 --seeds 42,43,44 --output table6.csv
```

