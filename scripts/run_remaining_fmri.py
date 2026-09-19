"""Official baseline adapters for the shared fMRI data and evaluation protocol."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def timegan(x, cfg, out, smoke):
    import tensorflow.compat.v1 as tf
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    print('TensorFlow GPU devices:', tf.config.list_physical_devices('GPU'), flush=True)
    if os.environ.get('REQUIRE_TF_GPU') == '1' and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Requested TensorFlow GPU runtime has no visible GPU')
    tf.disable_v2_behavior()
    sys.path.insert(0, str(ROOT / 'baselines/TimeGAN'))
    from timegan import timegan as fit
    if smoke:
        cfg['iterations'] = cfg.pop('smoke_steps', 1)
        cfg['batch_size'] = min(cfg['batch_size'], len(x))
    cfg['checkpoint_prefix'] = str(out / 'model')
    return fit(x, cfg)


def timevae(x, cfg, out, smoke):
    import tensorflow as tf
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    print('TensorFlow GPU devices:', tf.config.list_physical_devices('GPU'), flush=True)
    if os.environ.get('REQUIRE_TF_GPU') == '1' and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Requested TensorFlow GPU runtime has no visible GPU')
    tf.keras.utils.set_random_seed(cfg.pop('seed'))
    sys.path.insert(0, str(ROOT / 'baselines/TimeVAE/src'))
    from vae.timevae import TimeVAE
    epochs = cfg.pop('max_epochs')
    model = TimeVAE(seq_len=x.shape[1], feat_dim=x.shape[2], **cfg)
    model.fit_on_data(x, max_epochs=1 if smoke else epochs, verbose=2)
    model.save(str(out))
    return model.get_prior_samples(len(x))


def timevqvae(x, cfg, out, smoke):
    import torch
    import yaml
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    sys.path.insert(0, str(ROOT / 'baselines/TimeVQVAE/src'))
    from timevqvae.vqvae import VQVAE
    from timevqvae.maskgit import MaskGIT, PriorModelConfig
    official = yaml.safe_load((ROOT / 'baselines/TimeVQVAE/configs/config.yaml').read_text())
    ec, vc, mc = official['encoder'], official['VQ-VAE'], official['MaskGIT']
    # Standardize using this same training set; invert before shared evaluation.
    mean, std = x.mean(axis=(0, 1), keepdims=True), x.std(axis=(0, 1), keepdims=True)
    std = std.clip(1e-6)
    data = torch.tensor(((x - mean) / std).transpose(0, 2, 1), dtype=torch.float32, device='cuda')
    model = VQVAE(in_channels=x.shape[2], input_length=x.shape[1], n_fft=vc['n_fft'],
                  init_dim=ec['init_dim'], hid_dim=ec['hid_dim'],
                  downsampled_width_l=ec['downsampled_width']['lf'],
                  downsampled_width_h=ec['downsampled_width']['hf'],
                  encoder_n_resnet_blocks=ec['n_resnet_blocks'],
                  decoder_n_resnet_blocks=official['decoder']['n_resnet_blocks'],
                  codebook_size_l=vc['codebook_sizes']['lf'], codebook_size_h=vc['codebook_sizes']['hf'],
                  kmeans_init=vc['kmeans_init'], codebook_dim=vc['codebook_dim']).cuda()

    def train(module, stage, total, lossfn):
        optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=cfg['lr'])
        module.train()
        count = 1 if smoke else total
        warmup = max(1, int(count * cfg['warmup_fraction']))
        for step in range(count):
            progress = max(0, step-warmup) / max(1, count-warmup)
            lr = cfg['lr']*(step+1)/warmup if step < warmup else cfg['min_lr'] + (cfg['lr']-cfg['min_lr'])*(1+math.cos(math.pi*progress))/2
            for group in optimizer.param_groups:
                group['lr'] = lr
            indices = torch.randperm(len(data), device='cuda')[:cfg['batch_size']]
            optimizer.zero_grad()
            loss = lossfn(module, data[indices])
            if not torch.isfinite(loss):
                raise ValueError(f'{stage}: nonfinite loss at step {step}')
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f'{stage} step={step+1}/{count} loss={loss.item():.6f}', flush=True)
            if (step+1) % 1000 == 0 or step+1 == count:
                torch.save({'model': module.state_dict(), 'optimizer': optimizer.state_dict(),
                            'step': step+1, 'mean': mean, 'std': std, 'config': cfg}, out / f'{stage}.pt')

    def reconstruction(module, batch):
        value = module(batch)
        return sum(value.recons_loss.values()) + sum(v['loss'].mean() for v in value.vq_losses.values())
    train(model, 'stage1', cfg['stage1_steps'], reconstruction)
    model.eval()
    prior = MaskGIT(vqvae=model,
                   lf_choice_temperature=mc['choice_temperatures']['lf'],
                   hf_choice_temperature=mc['choice_temperatures']['hf'],
                   lf_num_sampling_steps=mc['T']['lf'], hf_num_sampling_steps=mc['T']['hf'],
                   lf_codebook_size=vc['codebook_sizes']['lf'], hf_codebook_size=vc['codebook_sizes']['hf'],
                   transformer_embedding_dim=ec['hid_dim'],
                   lf_prior_model_config=PriorModelConfig(**mc['prior_model_l']),
                   hf_prior_model_config=PriorModelConfig(**mc['prior_model_h']),
                   classifier_free_guidance_scale=mc['cfg_scale'], n_classes=1).cuda()
    train(prior, 'stage2', cfg['stage2_steps'], lambda m, b: m(b, torch.zeros(len(b),1,dtype=torch.long,device='cuda')).total_mask_prediction_loss)
    prior.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(data), cfg['batch_size']):
            count = min(cfg['batch_size'], len(data)-start)
            lo, hi = prior.iterative_decoding(num_samples=count, class_condition=None, device='cuda')
            result = prior.decode_token_ind_to_timeseries(lo, 'lf') + prior.decode_token_ind_to_timeseries(hi, 'hf')
            pieces.append(result.transpose(1, 2).cpu().numpy())
    import numpy as np
    return np.concatenate(pieces) * std + mean


def evaluate(method, seed, fake, real, out, dataset='fMRI', dataset_key='fmri'):
    import numpy as np
    py = str(ROOT / '.venv/bin/python')
    evaluation_env = os.environ.copy()
    for key in ['TF_USE_LEGACY_KERAS', 'LD_LIBRARY_PATH', 'REQUIRE_TF_GPU', 'XLA_FLAGS', 'TF_XLA_FLAGS']:
        evaluation_env.pop(key, None)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([py, '-u', str(ROOT / 'scripts/eval_seeded.py'), '--root', str(out),
                    '--ori_path', str(real), '--fake_path', str(fake), '--seed', str(seed),
                    '--n_iter', '1'], cwd=ROOT, env=evaluation_env, check=True)
    values = dict(line.split(':',1) for line in (out/'metrics.txt').read_text().splitlines())
    mapping = {'Context-FID':'C-FID', 'KL':'KL', 'DS_mean':'DS', 'PS_mean':'PS',
               **{f'Segment-DTW-L{n}':f'Seg-DTW-L{n}' for n in [6,8,10]}}
    names = {'timegan':'TimeGAN', 'timevae':'TimeVAE', 'timevqvae':'TimeVQVAE', 'gtgan':'GT-GAN',
             'diffusion_ts':'Diffusion-TS', 'pad_ts':'PaD-TS', 'k_protodiff':'K-ProtoDiff'}
    record = {'dataset':dataset, 'method':names[method], 'seed':seed,
              'provenance':'local_reproduction', 'evaluator':'eval_seeded.py',
              'metrics':{key:float(values[src]) for src,key in mapping.items()}}
    metadata = fake.parent / 'run_config.json'
    if metadata.exists():
        record['run_config'] = json.loads(metadata.read_text())
    assert all(np.isfinite(v) for v in record['metrics'].values())
    with (ROOT/'OUTPUT/.main-results.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT/'OUTPUT/main_results_records'/f'{dataset_key}__{method}__seed{seed}.json'
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(record, indent=2)+'\n')
        temp.replace(path)
        for builder in ['build_main_results_table.py', 'build_segment_dtw_table.py']:
            subprocess.run([py, str(ROOT/'scripts'/builder)], cwd=ROOT, check=True)


def gtgan(x, cfg, out, smoke):
    import numpy as np
    source = out/'training_data.npy'
    np.save(source, x)
    command = [sys.executable, '-u', 'GTGAN_energy.py', '--data', 'fmri',
               '--fmri-array', str(source), '--train', '--skip-eval', '--seed', str(cfg['seed']),
               '--save_dir', str(out), '--batch-size', str(min(len(x),cfg['batch_size'])),
               '--first_epoch', str(1 if smoke else cfg['first_epoch']),
               '--max-steps', str(5 if smoke else cfg['max_steps']),
               '--atol', '1e-3', '--rtol', '1e-3', '--log_time', '2',
               '--reconstruction', '0.01', '--kinetic-energy', '0.5', '--jacobian-norm2', '0.1']
    subprocess.run(command, cwd=ROOT/'baselines/GT-GAN', check=True)
    return np.load(out/'samples.npy')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['timegan', 'timevae', 'timevqvae', 'gtgan'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--smoke-steps', type=int, default=1)
    p.add_argument('--smoke-tag', default='default')
    p.add_argument('--smoke-data-limit', type=int, default=32)
    p.add_argument('--tf-intra-threads', type=int, default=4)
    p.add_argument('--tf-inter-threads', type=int, default=2)
    p.add_argument('--tf-xla', action='store_true')
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      TF_NUM_INTRAOP_THREADS=str(a.tf_intra_threads),
                      TF_NUM_INTEROP_THREADS=str(a.tf_inter_threads), TF_CPP_MIN_LOG_LEVEL='2')
    if a.tf_xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
    import numpy as np
    import yaml
    config = yaml.safe_load((ROOT/'Config/baselines/fmri.yaml').read_text())
    real = ROOT/config['real_path']
    ckpt = ROOT/('OUTPUT/smoke_remaining' if a.smoke else 'checkpoints')/a.method/f'seed{a.seed}'
    if a.smoke:
        ckpt = ckpt / Path(sys.prefix).name / a.smoke_tag
    ckpt.mkdir(parents=True, exist_ok=True)
    with (ckpt/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = dict(config[a.method], seed=a.seed)
        if a.batch_size is not None:
            if a.batch_size <= 0:
                raise ValueError('batch size must be positive')
            cfg['batch_size'] = a.batch_size
        random.seed(a.seed)
        np.random.seed(a.seed)
        x = np.load(real).astype(np.float32)
        if a.smoke:
            x = x[:max(a.smoke_data_limit, cfg['batch_size'])]
            if a.method == 'timegan':
                cfg['smoke_steps'] = a.smoke_steps
        fake = ckpt/'samples.npy'
        if not fake.exists():
            (ckpt/'run_config.json').write_text(json.dumps({
                'method': a.method, 'seed': a.seed, 'gpu': a.gpu,
                'parameters': cfg, 'smoke': a.smoke,
                'tf_intra_threads': a.tf_intra_threads, 'tf_inter_threads': a.tf_inter_threads,
                'tf_xla': a.tf_xla,
                'launched_at': datetime.now(timezone.utc).isoformat(),
            }, indent=2)+'\n')
            print(a.method, 'seed', a.seed, 'data', x.shape, flush=True)
            result = globals()[a.method](x, cfg, ckpt, a.smoke)
            result = np.asarray(result)
            assert result.shape == x.shape and np.isfinite(result).all(), result.shape
            np.save(fake, result)
        if not a.smoke:
            evaluate(a.method, a.seed, fake, real, ROOT/'OUTPUT/fmri_baselines'/f'metrics_{a.method}_seed{a.seed}')
        print('COMPLETED', a.method, a.seed, 'smoke', a.smoke, flush=True)


if __name__ == '__main__':
    main()
