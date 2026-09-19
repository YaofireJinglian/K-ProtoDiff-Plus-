"""Baseline diffusion throughput improvements with unchanged training budgets."""
import random
import time
import numpy as np
import torch


def train(trainer, x, out):
    # Keep the existing DataLoader ordering and batch boundaries.
    checkpoint = out/'fast_training.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        trainer.model.load_state_dict(state['model'])
        trainer.ema.load_state_dict(state['ema'])
        trainer.opt.load_state_dict(state['opt'])
        trainer.sch.load_state_dict(state['scheduler'])
        trainer.step = state['step']
        trainer.milestone = state['milestone']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        print('RESUME optimized baseline step',trainer.step,flush=True)
    started = time.monotonic()
    start = trainer.step
    trainer.model.train()
    while trainer.step < trainer.train_num_steps:
        trainer.opt.zero_grad(set_to_none=True)
        losses = []
        for _ in range(trainer.gradient_accumulate_every):
            batch = next(trainer.dl).to(trainer.device, non_blocking=True)
            loss = trainer.model(batch, target=batch) / trainer.gradient_accumulate_every
            loss.backward()
            losses.append(loss.detach())
        # One device synchronization per optimizer step, rather than per microbatch.
        value = torch.stack(losses).sum().item()
        if not np.isfinite(value):
            raise ValueError('Nonfinite baseline loss')
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),1.0)
        trainer.opt.step()
        trainer.sch.step(value)
        trainer.step += 1
        trainer.ema.update()
        if trainer.step % 100 == 0 or trainer.step == trainer.train_num_steps:
            print(f'TRAIN step={trainer.step}/{trainer.train_num_steps} loss={value:.6f} '
                  f'seconds_per_step={(time.monotonic()-started)/max(1,trainer.step-start):.4f}',flush=True)
        if trainer.step % trainer.save_cycle == 0:
            trainer.milestone += 1
            trainer.save(trainer.milestone)
        if trainer.step % 500 == 0 or trainer.step == trainer.train_num_steps:
            # Native DataLoader cursor is not serializable: restart ordering on
            # resume is explicitly recorded, not represented as bitwise replay.
            state = dict(model=trainer.model.state_dict(),ema=trainer.ema.state_dict(),
                         opt=trainer.opt.state_dict(),scheduler=trainer.sch.state_dict(),
                         step=trainer.step,milestone=trainer.milestone,
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                         numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                         resume_data_order='new DataLoader iterator')
            temp = checkpoint.with_suffix('.tmp')
            torch.save(state,temp)
            temp.replace(checkpoint)


def sample(trainer, count, batch_size, shape, out):
    """Save independent sample chunks so an interrupted run keeps its work."""
    folder = out/'sample_chunks'
    folder.mkdir(exist_ok=True)
    chunks = []
    started = time.monotonic()
    trainer.ema.ema_model.eval()
    with torch.no_grad():
        for start in range(0,count,batch_size):
            path=folder/f'{start:08d}.npy'
            size=min(batch_size,count-start)
            if path.exists():
                value=np.load(path)
            else:
                # Each chunk gets a fixed stream, including after recovery.
                seed=int(out.name.removeprefix('seed')) + 100000 + start
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    value=trainer.ema.ema_model.generate_mts(batch_size=size).cpu().numpy()
                with path.with_suffix('.tmp').open('wb') as handle:
                    np.save(handle,value)
                path.with_suffix('.tmp').replace(path)
            if value.shape != (size,*shape) or not np.isfinite(value).all():
                raise ValueError(f'Invalid generated chunk {path}')
            chunks.append(value)
            print(f'SAMPLE {start+size}/{count} elapsed={time.monotonic()-started:.1f}',flush=True)
    return (np.concatenate(chunks)+1)/2
