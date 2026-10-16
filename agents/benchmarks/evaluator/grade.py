"""Grade a completed consumer against pre-frozen fixtures; never load in an agent run."""
import importlib.util,itertools,json,os,subprocess,sys,traceback
from pathlib import Path
try:
 import numpy as np
except ImportError:
 np = None

def close(a,b):
 if np is not None and (isinstance(a,(list,tuple,np.ndarray)) or isinstance(b,(list,tuple,np.ndarray))):
  a=np.asarray(a); b=np.asarray(b)
  assert a.shape==b.shape,('shape',a.shape,b.shape)
  assert np.isfinite(a).all(),'nonfinite'
  assert np.all(np.abs(a-b)<=1e-12+1e-10*np.abs(b)), 'signed numerical mismatch'
 else:
  import math
  assert math.isfinite(a) and math.isfinite(b),'nonfinite'
  assert abs(a-b)<=1e-12+1e-10*abs(b), 'signed numerical mismatch'
def array(a,ndim):
 assert np is not None,'numpy required'
 assert isinstance(a,np.ndarray) and a.dtype==np.float64 and a.ndim==ndim,'array contract'
 assert a.flags.owndata,'output must own its buffer'
def cli(path,args=(),stdin=''):
 q=subprocess.run([str(path),*map(str,args)],input=stdin,text=True,capture_output=True,timeout=120)
 assert q.returncode==0,('process failure',q.returncode,q.stderr[-1000:])
 return [x.split() for x in q.stdout.splitlines() if x.strip()]
def load(consumer,name):
 spec=importlib.util.spec_from_file_location(name,consumer/(name+'.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def submission_files(consumer):
 assert (consumer/'README.md').is_file(),'missing consumer README.md'
 tests=[p for p in consumer.glob('test*') if p.is_file() and p.suffix in ('.py','.cpp','.cc','.sh','')]
 tests += [p for d in (consumer/'tests', consumer/'test') if d.is_dir() for p in d.rglob('*') if p.is_file()]
 assert tests,'missing local test file'

def grade(task,consumer,fixtures):
 submission_files(consumer)
 cases=fixtures[str(task)]; checks=0
 if task in (2,3):
  p=subprocess.run(['make','-B','-C',str(consumer)],capture_output=True,text=True,timeout=180)
  assert p.returncode==0,('build',p.stderr[-3000:],p.stdout[-1000:])
 if task==1:
  f=load(consumer,'recouple').recoupling
  previous=None
  for c in cases:
   kind,*labels=c['request']; left,right,v=f(kind,*(tuple(x) for x in labels))
   assert [list(x) for x in left]==c['left']; assert [list(x) for x in right]==c['right']
   array(v,2); close(v,c['value'])
   assert np.linalg.norm(v@v.T-np.eye(len(left)))<=1e-9
   assert np.linalg.norm(v.T@v-np.eye(len(right)))<=1e-9
   if previous: close(previous[0],previous[1])
   previous=v,v.copy(); checks+=1
 elif task==2:
  exe=consumer/'u9_batch'; assert cli(exe)==[]
  stdin=''.join(' '.join(str(x) for pair in c['request'] for x in pair)+'\n' for c in cases)
  lines=iter(cli(exe,stdin=stdin))
  for q,c in enumerate(cases):
   assert next(lines)==['block',str(q),*map(str,c['shape'])]
   for row in c['entries']:
    line=next(lines); assert line[:8]==['entry',str(q),*map(str,row[:6])]; assert len(line)==9
    close(float(line[8]),row[6]); checks+=1
  assert next(lines,None) is None
 elif task==3:
  exe=consumer/'coupling_batch'; assert cli(exe)==[]
  stdin='\n'.join(c['request'] for c in cases)+'\n'
  lines=iter(cli(exe,stdin=stdin))
  for q,c in enumerate(cases):
   expected_header=[c['tag'],str(q),*map(str,c['shape'])]
   actual_header=next(lines,None)
   assert actual_header==expected_header,('header mismatch',q,actual_header,expected_header)
   tag='c' if c['tag']=='canon' else 'p'
   for row in c['entries']:
    line=next(lines,None)
    assert line is not None,('premature EOF',q,row)
    prefix=[tag,str(q),*map(str,row[:-1])]
    assert line[:-1]==prefix,('entry prefix mismatch',line,prefix)
    close(float(line[-1]),row[-1]); checks+=1
  assert next(lines,None) is None,'trailing output'
 else: raise ValueError(task)
 return checks

if __name__=='__main__':
 try:
  n=grade(int(sys.argv[1]),Path(sys.argv[2]).resolve(),json.loads(Path(sys.argv[3]).read_text()))
  print(json.dumps(dict(success=True,checks=n)))
 except Exception as e:
  print(json.dumps(dict(success=False,error=str(e),traceback=traceback.format_exc()[-5000:])))
  sys.exit(1)
