"""Install pinned current harness sources into a job-private Python runtime."""
import json
import shutil
import sys
from pathlib import Path

from provenance import tree_digest


def main():
    """Relocate the sealed Python and replace only its pure Python harness packages."""
    runtime=Path(sys.argv[1]).resolve();source=Path(sys.argv[2]).resolve()
    identity=json.loads(Path(sys.argv[3]).read_text())
    assert str(runtime).startswith('/scratch/jobs/')
    assert (source/'platypus/packages').is_dir(), 'Harness capsule needs platypus/packages'
    assert (source/'taaf_source/src/taaf').is_dir(), 'Harness capsule needs taaf_source/src/taaf'
    cfg=runtime/'pyvenv.cfg'
    cfg.write_text('\n'.join('home = '+str(runtime/'.python/bin') if s.startswith('home = ') else
                   'executable = '+str(runtime/'.python/bin/python3.12') if s.startswith('executable = ') else s
                   for s in cfg.read_text().splitlines())+'\n')
    site=runtime/'lib/python3.12/site-packages'
    allowed={'duck','framediff','platypus','reference_agents','seal','sweeper','ullm'}
    installed=[]
    for package in sorted((source/'platypus/packages').glob('*/src/*')):
        if not package.is_dir() or package.name=='__pycache__':continue
        assert package.name in allowed,package
        target=site/package.name
        if target.exists():shutil.rmtree(target)
        shutil.copytree(package,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        installed.append(package.name)
    taaf=site/'taaf'
    if taaf.exists():shutil.rmtree(taaf)
    shutil.copytree(source/'taaf_source/src/taaf',taaf,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    identity['packages']={name:{'path':str(site/name),'sha256':tree_digest(site/name)} for name in installed+['taaf']}
    (runtime/'HARNESS_IMPORTED.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps({'installed':installed+['taaf']}))


if __name__=='__main__':main()
