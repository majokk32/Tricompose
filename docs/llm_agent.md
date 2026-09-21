# Quick LLM policy connection

TriCompose can call any OpenAI-compatible `chat/completions` endpoint as a
routing policy. This is a lightweight API request; it does not run a model on
the CARC login node.

The client sends only an allowlisted state:

- opaque candidate IDs;
- Qwen candidate scores;
- peer-agreement metrics;
- selection thresholds;
- allowed action names.

It never sends EHR rows, images, report text, patient identifiers, artifact
paths, or hashes. Start with the built-in patient-free synthetic demo.

Configure the endpoint without placing a secret in shell history:

```bash
export TRICOMPOSE_LLM_BASE_URL='https://YOUR_ENDPOINT/v1'
export TRICOMPOSE_LLM_MODEL='YOUR_MODEL_NAME'
read -rsp 'API key: ' TRICOMPOSE_LLM_API_KEY
export TRICOMPOSE_LLM_API_KEY
```

For a local or otherwise unauthenticated OpenAI-compatible endpoint, leave
`TRICOMPOSE_LLM_API_KEY` unset. The model server itself must already be running
on an approved compute service; do not launch local inference on a login node.

Run a synthetic connection test with a new opaque output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  -m tricompose.agent.openai_compatible_policy \
  --demo-state \
  --output-dir artifacts/protected/agent_demos/demo_001
```

After the synthetic test succeeds, an explicitly approved run may use the
protected numeric state from the deterministic selector:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  -m tricompose.agent.openai_compatible_policy \
  --agent-decision artifacts/protected/real_smoke_003/agent/report_selection_v2/agent_decision.json \
  --output-dir artifacts/protected/real_smoke_003/agent/llm_policy_v1
```

The LLM proposes an action, but a local hard guard validates it. In particular,
`select` is overridden to `verify_more` unless the deterministic minimum-score
and margin gates pass and the selected candidate is the eligible top
candidate. This keeps the LLM useful for orchestration without making it the
sole clinical-consistency authority. The protected result follows
`schemas/llm_agent_decision_v1.schema.json`.
