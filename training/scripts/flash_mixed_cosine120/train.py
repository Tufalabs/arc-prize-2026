"""Qualify mixed CE/KL on four B200s, then train the two-pass cosine recipe."""
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path('/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/artifacts')
PUBLIC=Path('/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030')
ATTEMPT=PUBLIC/'runs'/('job-'+os.environ['SLURM_JOB_ID'])
SCRATCH=Path('/scratch/jobs')/os.environ['SLURM_JOB_ID']/'mixed120-training'
SOURCE=SCRATCH/'source'
PRODUCTION=PUBLIC/'training'


def read(path):
    """Read a complete JSON artifact."""
    return json.loads(Path(path).read_text())


def digest(path):
    """Hash a capsule or dataset without retaining its bytes."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    """Publish progress without exposing a partial JSON document."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run(command,phase,env):
    """Run one gated phase and retain its console log."""
    print(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'phase':phase,'status':'started'}),flush=True)
    with (ATTEMPT/(phase+'.log')).open('w') as f:
        subprocess.run([str(x) for x in command],cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'phase':phase,'status':'passed'}),flush=True)


def accept(directory,config_path,expected=None):
    """Require valid CE/KL accounting, frozen routers, synchronized adapters and a full checkpoint."""
    state=read(directory/'RUN_STATE.json');config=read(config_path)
    assert state['world_size']==4
    assert expected is None or state['update']==expected
    assert state['config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    records=[json.loads(line) for line in (directory/'metrics.jsonl').open()]
    updates=[r for r in records if 'train' in r]
    assert updates and updates[-1]['update']==state['update']
    for record in updates:
        metrics=record['train']
        assert all(math.isfinite(metrics[k]) for k in ['loss_per_token','ce_loss','teacher_kl','grad_norm'])
        assert metrics['teacher_kl']>=-1e-6
        assert math.isclose(metrics['loss_per_token'],metrics['ce_loss']+0.3*metrics['teacher_kl'],rel_tol=2e-5,abs_tol=2e-6)
    validation=updates[-1]['validation']
    assert set(validation['by_game'])=={'pathfinder-0001','taps-0001','cd82-fb555c5d','dc22-fdcac232','escape_velocity-0001'}
    assert math.isclose(validation['ce_plus_kl_loss'],validation['ce_loss']+0.3*validation['teacher_kl'],rel_tol=2e-5,abs_tol=2e-6)
    parameters=read(directory/'PARAMETERS_PASSED.json')
    assert parameters['status']=='passed' and parameters['update']==state['update']
    assert parameters['updated_adapter_groups']==['attention','routed','shared']
    assert parameters['router_tensors']==48
    assert all(parameters[k] for k in ['trainable_replicas_equal','frozen_routers_unchanged','optimizer_contains_only_adapters','head_frozen'])
    latest=read(directory/'checkpoints/latest.json');checkpoint=directory/'checkpoints'/latest['path']
    assert latest['update']==state['update']
    assert all((checkpoint/name).is_file() for name in ['adapters.pt','optimizer.pt','state.json',
                   *[f'optimizer-rank{i}.pt' for i in range(4)],*[f'rng-rank{i}.pt' for i in range(4)]])
    return {'status':'passed',**state,'validation':validation,'checkpoint':str(checkpoint),'parameters':parameters}


def main():
    """Start fresh only after CPU data preparation and an isolated real mixed-loss update pass."""
    ATTEMPT.mkdir(parents=True,exist_ok=False);SCRATCH.mkdir(parents=True)
    ready=read(PUBLIC/'DATA_READY.json')
    assert ready['status']=='passed' and ready['holdout_disjoint'] and ready['gpus']==4
    official=read(PUBLIC/'OFFICIAL_EVALUATION_BUILD_PASSED.json')
    assert official['status']=='passed' and official['student_profile'] and official['fp8_kv']
    assert official['holdout_passes']==20 and official['official_passes']==3
    assert int(os.environ['SLURM_GPUS_ON_NODE'])==4
    config_path=PUBLIC/'train-config.json'
    assert digest(config_path)==ready['config_sha256']
    identity=read(ROOT/'training-identity.json')
    assert digest(ROOT/'source.tar')==identity['source_sha256']
    assert digest(ROOT/'harness-source.tar.gz')==identity['harness_archive_sha256']
    for p in [SOURCE,SCRATCH/'runtime',SCRATCH/'harness']:p.mkdir()
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             PYTHONPATH=str(SOURCE)+':'+str(ROOT/'test-deps'),
             PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',GLOO_SOCKET_IFNAME='lo',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',
             WANDB_DIR=str(PUBLIC/'wandb'),TRITON_CACHE_DIR=str(SCRATCH/'triton'),
             TORCHINDUCTOR_CACHE_DIR=str(SCRATCH/'inductor'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
             FLASH_SFT_PINNED_TELEMETRY='1')
    # The first extraction must precede the helper's cwd-based phase runner.
    subprocess.run(['tar','-xf',str(ROOT/'source.tar'),'-C',str(SOURCE)],check=True)
    assert read(config_path)==read(SOURCE/'configs/flash_sft/reap307-kimi-mixed-cosine.json')
    subprocess.run([sys.executable,str(SOURCE/'scripts/flash_mixed_cosine120/stage_checks.py'),str(ROOT)],check=True)
    runtime={'tarball':str(ROOT/'trainer-runtime.tar.zst'),'tarball_sha256':read(SOURCE/'scripts/flash_mixed_cosine120/artifacts.json')['files']['trainer-runtime.tar.zst']}
    assert digest(runtime['tarball'])==runtime['tarball_sha256']
    run(['tar','--zstd','-xf',runtime['tarball'],'-C',SCRATCH/'runtime'],'unpack-runtime',env)
    run(['tar','-xzf',ROOT/'harness-source.tar.gz','-C',SCRATCH/'harness'],'unpack-harness',env)
    run([sys.executable,SOURCE/'scripts/flash_mixed_cosine120/bootstrap.py',SCRATCH/'runtime',SCRATCH/'harness',SOURCE/'scripts/flash_mixed_cosine120/artifacts.json'],'install-current-harness',env)
    python=SCRATCH/'runtime/bin/python'
    env['FLASH_MIXED_HARNESS_IDENTITY']=str(SCRATCH/'runtime/HARNESS_IMPORTED.json')
    save(ATTEMPT/'SOURCE.json',identity)
    (PUBLIC/'wandb').mkdir(exist_ok=True)
    run([python,'-m','pytest','tests/test_flash_sft.py','tests/test_flash_sft_logprobs.py',
         'tests/test_flash_sft_logprobs_sharded.py','tests/test_flash_sft_cosine120.py',
         'tests/test_kimi_conversion_diagnostics.py','tests/test_mixed_sft_provenance.py','tests/logprob',
         '-q','--tb=short'],'trainer-environment-tests',dict(env,WANDB_MODE='disabled'))
    run([python,'-m','sft.distillation.staging','--source',read(config_path)['model'],'--destination',SCRATCH/'model'],'stage-model',env)
    env['FLASH_SFT_STAGED_MODEL']=str(SCRATCH/'model')
    command=[python,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4','--module','training.flash_sft']
    resume=bool(os.environ.get('MIXED120_RESUME'))
    if not resume:
        assert not PRODUCTION.exists(),'A fresh experiment must not overwrite an earlier training directory'
        sys.path.insert(0,str(SOURCE))
        from training.flash_sft import RowIndex
        config=read(config_path)
        index=RowIndex(config['rows']['data'],config['rows']['data_sha256'],max_length=config['max_length'])
        longest=sorted(range(len(index)),key=lambda i:index.lengths[i],reverse=True)
        chosen=[]
        for want_kl in [False,True]:
            chosen.append(next(i for i in longest if bool(index.row(i)['teacher_logprob_targets']['events'])==want_kl))
        for i in longest:
            if i not in chosen:chosen.append(i)
            if len(chosen)==4:break
        assert len(chosen)==4 and len(set(chosen))==4
        qualification_rows=ATTEMPT/'qualification-rows.jsonl'
        with qualification_rows.open('w') as f:
            for i in chosen:f.write(json.dumps(index.row(i),separators=(',',':'))+'\n')
        config['rows']={'data':str(qualification_rows),'data_sha256':digest(qualification_rows)}
        # Four rows make one update per pass. Retain the production schedule's
        # length so PR122's decay-range guard accepts its unchanged decay;
        # --max-updates=1 still executes only the isolated first update.
        config['passes']=ready['updates']
        config['wandb']=None
        qualification_config=ATTEMPT/'qualification-config.json';save(qualification_config,config)
        run([*command,'--config',qualification_config,'--run-dir',ATTEMPT/'qualification',
             '--max-updates','1','--verify-parameters'],'mixed-gpu-qualification',dict(env,WANDB_MODE='disabled'))
        qualified=accept(ATTEMPT/'qualification',qualification_config,1)
        assert qualified['validation']['kl_coverage']>0
        save(ATTEMPT/'QUALIFICATION_PASSED.json',qualified)
    else:
        assert (PRODUCTION/'checkpoints/latest.json').exists()
    save(ATTEMPT/'RUNNING.json',{'status':'running','fresh_adapters':not resume,'target_updates':ready['updates']})
    run([*command,'--config',config_path,'--run-dir',PRODUCTION,
         *([] if resume else ['--max-updates','20']),'--time-budget-seconds','72000',
         '--stop-file',PUBLIC/'STOP','--verify-parameters'],'production',env)
    result=accept(PRODUCTION,config_path)
    assert result['total_updates']==ready['updates']
    save(ATTEMPT/'SEGMENT_PASSED.json',result);(ATTEMPT/'RUNNING.json').unlink()
    if result['complete']:
        save(PUBLIC/'TRAINING_PASSED.json',dict(result,source=identity))
    elif (PUBLIC/'STOP').exists():
        save(PUBLIC/'STOPPED.json',result)
    else:
        response=subprocess.check_output(['sbatch','--parsable','--dependency=afterok:'+os.environ['SLURM_JOB_ID'],
                '--export=ALL,MIXED120_RESUME=1',str(ROOT/'train.sbatch')],text=True).strip()
        job=response.split(';')[0];assert job.isdigit()
        save(ATTEMPT/'CONTINUATION.json',{'job_id':job,'completed_updates':result['update']})
        save(PUBLIC/'LATEST_TRAINING_JOB.json',{'job_id':job,'completed_updates':result['update']})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if ATTEMPT.exists():
            (ATTEMPT/'RUNNING.json').unlink(missing_ok=True)
            save(ATTEMPT/'FAILED.json',{'type':type(error).__name__,'message':str(error)})
        raise
