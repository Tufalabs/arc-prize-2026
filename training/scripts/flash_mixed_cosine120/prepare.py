"""Build the committed Kimi corpus and check its hashes against the committed training recipe."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
BASE = Path('/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc3-reap307')
PUBLIC = Path('/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030')
HOLDOUT = {'pathfinder-0001','taps-0001','cd82-fb555c5d','dc22-fdcac232','escape_velocity-0001'}
EXCLUDE = {'crane_dipper-0001','sliders-0001'}
RECIPE = SOURCE/'configs/flash_sft/reap307-kimi-mixed-cosine.json'
MANIFEST = SOURCE/'configs/flash_sft/reap307-kimi-mixed-sources.json'
REQUIRED_ADDITIONS = {'arc2__arc2-train-hard+arc2-eval-20261030-110002',
                      'arc3__seal-re-arc-unlisted-s20261006-20261030-071018'}


def digest(path):
    """Hash a potentially large immutable input without loading it in memory."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    """Publish a complete JSON artifact atomically."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run(command,phase,env):
    """Run a preparation phase with a persistent, separately readable log."""
    print(json.dumps({'phase':phase,'status':'started'}),flush=True)
    with (PUBLIC/'preparation'/f'{phase}.log').open('w') as f:
        subprocess.run([str(x) for x in command],env=env,cwd=SOURCE,check=True,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps({'phase':phase,'status':'passed'}),flush=True)


def publish(config, audit):
    """Publish readiness only for the exact committed config and immutable data."""
    from training.flash_sft import load_config

    expected=json.loads(RECIPE.read_text())
    if config != expected:
        raise ValueError('Built data/config differs from the committed recipe; review and freeze the candidate first')
    for data in [config['rows'],config['validation']]:
        assert digest(data['data'])==data['data_sha256'],data['data']
    assert audit['manifest_sha256']==digest(MANIFEST)
    load_config(RECIPE)
    save(PUBLIC/'train-config.json',expected)
    save(PUBLIC/'DATA_READY.json',dict(audit,config_sha256=digest(PUBLIC/'train-config.json')))


def main(build_only=False, verify_only=False):
    """Keep CE on every usable completion and attach only audited original logprobs."""
    (PUBLIC/'preparation').mkdir(parents=True,exist_ok=True)
    failure=PUBLIC/'PREPARATION_FAILED.json'
    if failure.exists():
        stamp=dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%S%fZ')
        failure.replace(PUBLIC/'preparation'/('previous-failure-'+stamp+'.json'))
    env=dict(os.environ,PYTHONPATH=str(SOURCE)+':'+str(HERE/'conversion'),TOKENIZERS_PARALLELISM='false',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path[:0]=[str(SOURCE),str(HERE/'conversion')]
    from sft.logprob.preparation_cache import cache_matches, cache_record
    from provenance import imported_harness_identity
    harness=imported_harness_identity()
    if verify_only:
        candidate=json.loads((PUBLIC/'DATA_CANDIDATE.json').read_text())
        publish(candidate['config'],candidate['audit'])
        return
    from seal import SealAgent
    agent=SealAgent.build(environment='slurm-b200',kimi_student=True,checkpoint=str(BASE))
    agent.prepare_for_backend()
    factory=agent.llm_config.backend_factory
    kwargs=dict(factory.backend_kwargs.get('chat_template_kwargs') or {})
    kwargs['reasoning_effort']=agent.reasoning_effort
    assert agent.world_model_default and agent.summary_thinking and agent.reasoning_effort=='xhigh'
    serving=PUBLIC/'preparation/serving.jsonl'
    serving.write_text(json.dumps({'provider_validation':{'request_args':{'chat_template_kwargs':kwargs}}})+'\n')
    run_config=PUBLIC/'preparation/serving-run-config.json'
    save(run_config,{'engine_args':{'mm_processor_kwargs':factory.engine_args.mm_processor_kwargs}})
    save(PUBLIC/'RENDER_CONTRACT.json',{'harness_commit':harness['harness_commit'],'harness_identity':harness,
         'kimi_student':True,'summary_thinking':agent.summary_thinking,'world_model_default':agent.world_model_default,'chat_template_kwargs':kwargs,
         'teacher_contexts':'Preserved as recorded; reserialized with current tokenizer, tool results and continuation',
         'old_collection_behavior_is_not_retroactively_regenerated':True})
    inventory=PUBLIC/'SOURCE_INVENTORY.json'
    frozen=json.loads(MANIFEST.read_text())
    if inventory.exists() and json.loads(inventory.read_text()) != frozen:
        save(PUBLIC/'SOURCE_INVENTORY.before-reviewed-recipe.json',json.loads(inventory.read_text()))
    save(inventory,frozen)
    assert REQUIRED_ADDITIONS <= {item['name'] for item in frozen['collections']}, 'Refresh inventory with the two requested new datasets'
    render_files={name:digest(BASE/name) for name in json.loads(RECIPE.read_text())['rows_render_files']}
    conversion_flags=['--summary-thinking' if agent.summary_thinking else '--no-summary-thinking',
                      '--workers','4','--skip-unparseable-calls']
    renderer={'files':render_files,'chat_template_kwargs':kwargs,
              'mm_processor_kwargs':factory.engine_args.mm_processor_kwargs,
              'harness_archive_sha256':harness['harness_archive_sha256'],
              'runtime_sha256':harness['files']['harness-runtime.tar.zst'],
              'packages':{name:package['sha256'] for name,package in harness['packages'].items()}}
    conversion_records={}
    raw_roots={};converted_roots={};seen={};duplicates=[];logprob_steps={}
    for item in frozen['collections']:
        name=item['name'];archive=Path(item['archive'])
        assert digest(archive)==item['archive_sha256'],archive
        raw=PUBLIC/'raw'/name
        if not (raw/'EXTRACTED.json').exists():
            raw.mkdir(parents=True,exist_ok=True)
            proc=subprocess.Popen(['zstd','-dc',str(archive)],stdout=subprocess.PIPE)
            with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:tar.extractall(raw,filter='data')
            assert proc.wait()==0
            save(raw/'EXTRACTED.json',item)
        traces=raw/'traces'
        assert traces.is_dir(),raw
        raw_roots[name]=str(traces)
        for trace in sorted(traces.iterdir()):
            file=trace/'steps.jsonl'
            if not file.exists() or not (trace/'_SUCCESS').exists():continue
            # A run ID identifies one trajectory. Prefer its longest published snapshot.
            indices=set();count=0
            with file.open() as stream:
                for line in stream:
                    step=json.loads(line);count+=1
                    if (step.get('extra') or {}).get('teacher_logprobs'):indices.add(step['step_index'])
            logprob_steps[(name,trace.name)]=indices
            if trace.name in seen:
                old=seen[trace.name]
                if count>old['steps']:
                    (old['trace']/'_SUCCESS').rename(old['trace']/'_DUPLICATE')
                else:
                    (trace/'_SUCCESS').rename(trace/'_DUPLICATE')
                    duplicates.append({'run_id':trace.name,'discarded':name,'steps':count});continue
                duplicates.append({'run_id':trace.name,'discarded':old['source'],'steps':old['steps']})
            seen[trace.name]={'trace':trace,'source':name,'steps':count}
    for item in frozen['collections']:
        name=item['name'];traces=Path(raw_roots[name]);converted=PUBLIC/'converted'/name
        active=sorted(p.name for p in traces.iterdir() if (p/'_SUCCESS').exists())
        if not active:continue
        fingerprint=cache_record('conversion',
             inputs={'active_traces':active,'archive_sha256':item['archive_sha256'],
                     'benchmark_sha256':digest(Path(item['root'])/'benchmark.json')},
             options={'argv':conversion_flags,'checkpoint':str(BASE),'renderer':renderer},
             code_files={'conversion/kimi_steps.py':HERE/'conversion/kimi_steps.py'})
        fingerprint_path=converted/'INPUT_TRACES.json'
        if not (converted/'STEPS_READY.json').exists() or not cache_matches(fingerprint_path,fingerprint):
            if converted.exists():shutil.rmtree(converted)
            run([sys.executable,HERE/'conversion/kimi_steps.py','kimi','--traces',traces,
                 '--run-dir',item['root'],'--serving',serving,'--run-config',run_config,'--checkpoint',BASE,
                 *conversion_flags,'--out',converted],'convert-'+name,env)
            ready=json.loads((converted/'STEPS_READY.json').read_text())
            ready['conversion_cache']=fingerprint
            save(converted/'STEPS_READY.json',ready)
            save(fingerprint_path,fingerprint)
        ready=json.loads((converted/'STEPS_READY.json').read_text())
        assert ready['parse_back_fail']==0,converted
        assert ready['summary_thinking']==agent.summary_thinking,converted
        converted_roots[name]=str(converted)
        conversion_records[name]=fingerprint
    packed=PUBLIC/'packed-reviewed19'
    packing_flags=['--holdout',','.join(sorted(HOLDOUT)),'--exclude',','.join(sorted(EXCLUDE)),
                   '--max-completions','32','--max-length','192000','--probe-per-game','4']
    source_fingerprint=cache_record('packing',
         inputs={'manifest_sha256':digest(MANIFEST),
                 'converted_receipts':{name:digest(Path(path)/'STEPS_READY.json') for name,path in converted_roots.items()}},
         options={'argv':packing_flags},
         code_files={name:HERE/'conversion'/name for name in ['kimi_rows.py','grpo_rows.py']})
    packed_fingerprint=packed/'SOURCES_SHA256.json'
    if not (packed/'ROWS_READY.json').exists() or not cache_matches(packed_fingerprint,source_fingerprint):
        if packed.exists():shutil.rmtree(packed)
        run([sys.executable,HERE/'conversion/kimi_rows.py','--steps',*[k+'='+v for k,v in converted_roots.items()],
             *packing_flags,'--out',packed],'pack',env)
        ready=json.loads((packed/'ROWS_READY.json').read_text())
        ready['packing_cache']=source_fingerprint
        save(packed/'ROWS_READY.json',ready)
        save(packed_fingerprint,source_fingerprint)
    # Exact duplicate rows cannot acquire additional sampling weight from repeated source snapshots.
    for filename in ['rows.jsonl','validation.jsonl','validation-probe.jsonl']:
        source=packed/filename;best={};order=[]
        with source.open() as stream:
            for line in stream:
                row=json.loads(line)
                key=hashlib.sha256(json.dumps([row['input_token_ids'],row['output_token_ids'],
                     row.get('packed_completions',[]),[i['sha256'] for i in row.get('images',[])]],separators=(',',':')).encode()).hexdigest()
                quality=sum(i in logprob_steps[(row['source_set'],row['run_id'])] for i in row['steps'])
                if key not in best:order.append(key)
                if key not in best or quality>best[key][0]:best[key]=(quality,row)
        selected=PUBLIC/'deduplicated-reviewed19'/filename;selected.parent.mkdir(exist_ok=True)
        with selected.open('w') as output:
            for key in order:
                row=best[key][1];row['summary_thinking']=agent.summary_thinking
                output.write(json.dumps(row,separators=(',',':'))+'\n')
    save(PUBLIC/'sources.json',raw_roots)
    for filename in ['rows.jsonl','validation-probe.jsonl']:
        target=PUBLIC/'data-reviewed19'/filename
        run([sys.executable,'-m','sft.logprob.targets','--rows',PUBLIC/'deduplicated-reviewed19'/filename,
             '--sources',PUBLIC/'sources.json','--allow-missing-logprobs','--tokenizer',BASE/'tokenizer.json',
             '--out',target],'targets-'+filename,env)
    from training.flash_sft import RowIndex
    config=json.loads(RECIPE.read_text())
    from training.flash_sft import RENDER_FILES
    config['rows_render_files']={name:digest(BASE/name) for name in RENDER_FILES}
    config['rows']['data_sha256']=digest(config['rows']['data'])
    config['validation']['data_sha256']=digest(config['validation']['data'])
    train=RowIndex(config['rows']['data'],config['rows']['data_sha256'],max_length=config['max_length'],
                   recorded_logprobs=config['recorded_logprobs'],fields=['source_set'])
    valid=RowIndex(config['validation']['data'],config['validation']['data_sha256'],max_length=config['max_length'],
                   recorded_logprobs=config['recorded_logprobs'],fields=['source_set'])
    train_games={r['game_id'] for r in train.fields};valid_games={r['game_id'] for r in valid.fields}
    assert valid_games==HOLDOUT and not train_games&(HOLDOUT|EXCLUDE)
    assert any(r['source_set'].startswith('arc2__') for r in train.fields)
    assert any(r['source_set'].startswith('arc3__') for r in train.fields)
    decay_updates=math.ceil(len(train)/4)
    config['optimizer']['decay']={'decay_from':5,'decay_until':decay_updates-1,'final_fraction':0.1}
    coverage=json.loads((PUBLIC/'data-reviewed19/rows.jsonl.report.json').read_text())
    audit={'status':'passed','cutoff_utc':frozen['cutoff_utc'],'gpus':4,'gpu_limit':8,
         'batch_rows':4,'passes':2,'updates':2*decay_updates,
         'train_rows':len(train),'target_tokens_per_pass':sum(train.tokens),'kl_positions_per_pass':train.aligned_positions,
         'validation_rows':len(valid),'validation_games':sorted(valid_games),'train_games':len(train_games),
         'source_collections':[item['name'] for item in frozen['collections']],
         'excluded_collections':frozen['excluded_collections'],'manifest_sha256':digest(MANIFEST),
         'holdout_disjoint':True,'holdout_aliases_excluded':sorted(EXCLUDE),'coverage':coverage,
         'duplicate_trajectories':duplicates,
         'preparation_provenance':{'conversions':conversion_records,'packing':source_fingerprint,
                                   'targets_sha256':digest(SOURCE/'sft/logprob/targets.py')},
         'schedule':'PR122 native cosine over first pass after 5-update warmup; flat 1e-5 second pass',
         'training_stack':'PR122 native Flash SFT on main + recorded-logprob extension',
         'evaluation':'PR131: heldout 5 x 20; official+hard 36 x 3; hard baselines multiplied by 3'}
    save(PUBLIC/'DATA_CANDIDATE.json',{'config':config,'audit':audit})
    if not build_only:publish(config,audit)
    print(json.dumps({'status':'candidate-built' if build_only else 'data-ready','rows':len(train),'updates':2*decay_updates}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--build-only',action='store_true',help='Build a candidate for review without authorizing GPU training')
    mode.add_argument('--verify-only',action='store_true',help='Check a built candidate against the committed frozen recipe')
    args=parser.parse_args()
    try:main(args.build_only,args.verify_only)
    except BaseException as error:
        save(PUBLIC/'PREPARATION_FAILED.json',{'type':type(error).__name__,'message':str(error)})
        raise
