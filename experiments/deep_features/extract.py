"""Download official CIFAR/weights and cache frozen normalized image features."""
from pathlib import Path
import os,time,json,hashlib,argparse
R=Path(__file__).resolve().parent
os.environ['TORCH_HOME']=str(R/'cache/torch')
import numpy as np
import torch,torchvision
from torchvision import models,datasets

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def main():
 (R/'cache').mkdir(exist_ok=True);(R/'raw').mkdir(exist_ok=True)
 torch.set_num_threads(4)
 assert torch.cuda.is_available()
 a=torch.randn(256,256,device='cuda');_=(a@a).sum().item()
 print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'cuda':torch.version.cuda}),flush=True)
 from download_cifar import main as ensure_cifar
 ensure_cifar()
 data=[datasets.CIFAR10(str(R/'cache/data'),train=t,download=True) for t in [True,False]]
 rng=np.random.default_rng(9142026)
 ids=[]
 for ds,count in zip(data,[300,100]):
  y=np.asarray(ds.targets);ids.append(np.sort(np.concatenate([rng.choice(np.flatnonzero(y==c),count,replace=False) for c in range(10)])))
 np.savez(R/'raw/data_indices.npz',train=ids[0],test=ids[1],train_labels=np.asarray(data[0].targets)[ids[0]],test_labels=np.asarray(data[1].targets)[ids[1]])
 for name in ['resnet18','vit_b_16']:
  target=R/'raw'/f'{name}_features.npz'
  if target.exists():print('CACHE',name,flush=True);continue
  weights=models.ResNet18_Weights.IMAGENET1K_V1 if name=='resnet18' else models.ViT_B_16_Weights.IMAGENET1K_V1
  model=getattr(models,name)(weights=weights)
  if name=='resnet18':model.fc=torch.nn.Identity()
  else:model.heads=torch.nn.Identity()
  model.eval().cuda();trans=weights.transforms();torch.cuda.reset_peak_memory_stats()
  start=time.perf_counter();outputs=[]
  for ds,index in zip(data,ids):
   features=[]
   for k in range(0,len(index),32):
    batch=torch.stack([trans(ds[int(i)][0]) for i in index[k:k+32]]).cuda()
    with torch.inference_mode():out=model(batch)
    features.append(out.cpu().numpy())
    if k%320==0:print(name,'train' if ds.train else 'test',k,'/',len(index),'seconds',round(time.perf_counter()-start,1),flush=True)
   feat=np.concatenate(features);outputs.append(feat)
  np.savez_compressed(target,train_raw=outputs[0],test_raw=outputs[1],train=outputs[0]/np.linalg.norm(outputs[0],axis=1,keepdims=True),test=outputs[1]/np.linalg.norm(outputs[1],axis=1,keepdims=True))
  record={'name':name,'weights':str(weights),'url':weights.url,'transforms':str(trans),'seconds':time.perf_counter()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'feature_sha256':sha(target),'indices_sha256':sha(R/'raw/data_indices.npz'),'weights_sha256':{p.name:sha(p) for p in (R/'cache/torch/hub/checkpoints').glob('*')},'train_shape':list(outputs[0].shape),'test_shape':list(outputs[1].shape)}
  (R/'raw'/f'{name}_extraction.json').write_text(json.dumps(record,indent=2))
  print('FINISHED',name,record['seconds'],flush=True)
  del model;torch.cuda.empty_cache()
if __name__=='__main__':main()
