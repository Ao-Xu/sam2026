"""Download unchanged CIFAR archive and verify official MD5."""
from pathlib import Path
import hashlib,subprocess
R=Path(__file__).resolve().parent/'cache/data'
URL='https://scidata.sjtu.edu.cn/records/p4t8m-rbe26/files/cifar-10-python.tar.gz?download=1'
MD5='c58f30108f718f92721af3b95e74349a'
def main():
 R.mkdir(parents=True,exist_ok=True);p=R/'cifar-10-python.tar.gz'
 if p.exists() and hashlib.md5(p.read_bytes()).hexdigest()==MD5:
  print('CIFAR archive checksum verified');return
 subprocess.run(['curl.exe','-fL','--retry','2','--connect-timeout','20','-o',str(p),URL],check=True)
 assert hashlib.md5(p.read_bytes()).hexdigest()==MD5,'CIFAR checksum mismatch'
 print('CIFAR archive checksum verified')
if __name__=='__main__':main()
