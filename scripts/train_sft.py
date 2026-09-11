"""Quantized LoRA using installed MLX-LM trainer and audited target-only loss."""
import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agentrl.sft_dataset import read_jsonl
from src.agentrl.sft_training import TargetDataset, target_only_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path)
    args=parser.parse_args()
    import yaml
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import get_total_parameters
    from mlx_lm.tuner.trainer import train, evaluate, TrainingArgs
    cfg=yaml.safe_load(args.config.read_text())
    if cfg['fine_tune_type']!='lora':
        parser.error('Only quantized LoRA is authorized')
    if cfg['iters'] % cfg['grad_accumulation_steps']:
        parser.error('Do not discard unfinished gradient accumulation')
    out=Path(cfg['adapter_path'])
    if out.exists():
        parser.error('Adapter directory exists: refusing another run/overwrite')
    audit=json.loads(Path('artifacts/milestone4_audit/audit.json').read_text())
    assert audit['result']=='PASS'
    np.random.seed(cfg['seed']); mx.random.seed(cfg['seed'])
    model,tokenizer=load(cfg['model'],revision=cfg['revision'])
    quantized=sum(isinstance(m,nn.QuantizedLinear) for _,m in model.named_modules())
    assert quantized>0
    model.freeze()
    linear_to_lora_layers(model,cfg['num_layers'],cfg['lora_parameters'])
    params=tree_flatten(model.trainable_parameters())
    assert all('lora_a' in name or 'lora_b' in name for name,_ in params)
    trainable=sum(v.size for _,v in params); total=get_total_parameters(model)
    print(f'Trainable {trainable}/{total} = {trainable/total:.6%}; quantized linear modules {quantized}',flush=True)
    data=Path(cfg['data'])
    rows=read_jsonl(data/'train.jsonl'); valid_rows=read_jsonl(data/'valid.jsonl')
    allowed={r['id'] for r in read_jsonl('data/sft/train.jsonl')}
    assert all(r['canonical_id'] in allowed for r in rows)
    trainset=TargetDataset(rows,tokenizer); validset=TargetDataset(valid_rows,tokenizer)
    assert max(len(x[0]) for x in trainset.items+validset.items)<=cfg['max_seq_length']
    probe=TargetDataset(random.Random(42).sample(rows,min(32,len(rows))),tokenizer)
    out.mkdir(parents=True)
    (out/'adapter_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'training_config.yaml').write_text(args.config.read_text())
    started=time.perf_counter()
    summary={'config':cfg,'environment':{'Python':platform.python_version(), 'Mac':platform.platform(),
             **{name:version(name) for name in ['mlx','mlx-lm','huggingface-hub']}},
             'trainable_parameters':trainable,'total_parameters':total,'trainable_ratio':trainable/total,
             'quantized_linear_modules':quantized,'train_targets':len(rows),'valid_targets':len(valid_rows),
             'sample_exposure':cfg['iters']*cfg['batch_size'],'optimizer_updates':cfg['iters']//cfg['grad_accumulation_steps'],
             'passes':cfg['iters']*cfg['batch_size']/len(rows),
             'dataset_sha256':{n:hashlib.sha256((data/f'{n}.jsonl').read_bytes()).hexdigest() for n in ['train','valid']},
             'train_reports':[],'validation_reports':[]}
    def measured(dataset):
        return evaluate(model,dataset,batch_size=1,num_batches=-1,max_seq_length=cfg['max_seq_length'],loss=target_only_loss)
    summary['initial_train_probe_loss']=measured(probe)
    class Callback:
        def record(self,kind,info):
            record=dict(info,elapsed_seconds=time.perf_counter()-started)
            value=info['train_loss'] if kind=='train' else info['val_loss']
            if not math.isfinite(value):
                raise RuntimeError('Non-finite loss; stopping')
            if kind=='validation' and summary['validation_reports'] and value>3*summary['validation_reports'][0]['val_loss']:
                raise RuntimeError('Validation loss exploded; stopping')
            summary['train_reports' if kind=='train' else 'validation_reports'].append(record)
            with (out/'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(kind=kind,**record))+'\n')
            (out/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        def on_train_loss_report(self,info): self.record('train',info)
        def on_val_loss_report(self,info): self.record('validation',info)
    ta=TrainingArgs(batch_size=cfg['batch_size'],iters=cfg['iters'],val_batches=cfg['val_batches'],
        steps_per_report=cfg['steps_per_report'],steps_per_eval=cfg['steps_per_eval'],
        steps_per_save=cfg['save_every'],adapter_file=str(out/'adapters.safetensors'),
        max_seq_length=cfg['max_seq_length'],grad_checkpoint=cfg['grad_checkpoint'],
        grad_accumulation_steps=cfg['grad_accumulation_steps'])
    train(model,optim.Adam(learning_rate=cfg['learning_rate']),trainset,validset,args=ta,
          loss=target_only_loss,training_callback=Callback())
    summary['final_train_probe_loss']=measured(probe)
    summary['final_validation_loss']=measured(validset)
    summary['best_validation_loss']=min([r['val_loss'] for r in summary['validation_reports']]+[summary['final_validation_loss']])
    summary['elapsed_seconds']=time.perf_counter()-started
    summary['peak_memory_gb']=mx.get_peak_memory()/1e9
    summary['adapter_sha256']=hashlib.sha256((out/'adapters.safetensors').read_bytes()).hexdigest()
    summary['result']='PASS'
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETE',json.dumps({k:v for k,v in summary.items() if k not in ['config','train_reports','validation_reports']}),flush=True)


if __name__=='__main__':
    main()
