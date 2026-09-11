"""Explicit non-thinking tokenization and target-only loss boundaries."""


def tokenize_target(record, tokenizer):
    messages = record['messages']
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                           enable_thinking=False)
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False,
                                           enable_thinking=False, add_generation_prompt=True)
    if tokens[:len(prefix)] != prefix:
        raise ValueError('training tokens do not extend the non-thinking inference prefix')
    if len(tokens) <= len(prefix):
        raise ValueError('empty supervised target')
    return tokens, len(prefix)


def supervised_positions(offset, length, padded_length):
    """Target indices after autoregressive shift; EOS included, padding excluded."""
    return [i for i in range(1, padded_length) if offset <= i < length]


def target_only_loss(model, batch, lengths):
    import mlx.core as mx
    import mlx.nn as nn
    targets = batch[:, 1:]
    steps = mx.arange(1, batch.shape[1])
    mask = (steps >= lengths[:, 0:1]) & (steps < lengths[:, 1:])
    loss = nn.losses.cross_entropy(model(batch[:, :-1]), targets)
    ntokens = mask.sum()
    return (loss * mask).astype(mx.float32).sum() / ntokens, ntokens


class TargetDataset:
    def __init__(self, rows, tokenizer):
        self.items = [tokenize_target(row, tokenizer) for row in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
