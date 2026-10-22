"""Mirror the live experiment ledger and small metrics; leave model files remote."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
HOST='hongzhijun@192.168.10.208'


def main():
    command=['rsync','-a','--timeout=25','-e','ssh -o BatchMode=yes -o ConnectTimeout=10',
             '--include=/EXPERIMENTS.md','--include=/status.json','--include=/registry.json',
             '--include=/records/***','--include=/proofs/***','--include=/outputs/',
             '--include=/outputs/*/','--include=/outputs/*/metrics.json',
             '--include=/outputs/*/adaptation_state/',
             '--include=/outputs/*/adaptation_state/*.json','--exclude=*',
             HOST+':'+str(ROOT)+'/',str(ROOT)+'/']
    while True:
        try:
            subprocess.run(command,check=True,timeout=50)
            state=json.loads((ROOT/'status.json').read_text())['state']
            print(time.strftime('%Y-%m-%d %H:%M:%S'),state,flush=True)
            if state in ('complete','failed'):return
        except (OSError,subprocess.SubprocessError,ValueError) as error:
            print('SYNC_RETRY',type(error).__name__,str(error),flush=True)
        time.sleep(30)


if __name__=='__main__':main()
