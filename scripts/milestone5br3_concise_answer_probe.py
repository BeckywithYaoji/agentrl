"""B4: run the unchanged B3 probe with only a generic answer-format reminder."""
from scripts.milestone5br3_reward_variance_probe import main

CONCISE_REMINDER = (
    'When answering, output only the minimal final answer inside <answer>...</answer>.\n'
    'Do not include explanations, equations, derivations, or full sentences.'
)

if __name__ == '__main__':
    raise SystemExit(main(answer_reminder=CONCISE_REMINDER,
                          artifact_root='artifacts/milestone5br3b4'))
