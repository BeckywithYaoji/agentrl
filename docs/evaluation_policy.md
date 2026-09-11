# Evaluation Policy

The following sets are frozen audit sets from Milestone 4: Test 100, Counterfactual 30, OOD 30, and the four Milestone 2B baseline questions. Their IDs and SHA256 hashes are recorded in `artifacts/milestone4_eval/*_summary.json`.

They must not be used to choose reward constants, GRPO settings, or retriever parameters. Reward debugging uses training semantics, validation, and handwritten adversarial tests. The 24-prompt stochastic probe is a readiness smoke test only and uses validation prompts.
