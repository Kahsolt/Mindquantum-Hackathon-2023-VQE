import json
from pathlib import Path
from typing import List, Dict, Any
from traceback import print_exc

BASE_PATH = Path(__file__).parent.absolute()
DB_FILE = BASE_PATH / 'h4_best.json'

import sys
sys.path.append(str(BASE_PATH))
import solver; solver.TIMEOUT_LIMIT = 2**30
from solver import *

Record = {
  'fci': float,
  'geo': Geo,
}
DB = {
  'best': 'Record',
  'hist': List['Record'],
}


def load_db(fp:Path) -> DB:
  if not fp.exists():
    return {
      'best': {
        'fci': -2.2746174479004226,
        'geo': [
          -0.1739623688446747,  -1.9923301234570616,  1.7530631359825857,
          -1.074432362648722,    1.146662929665773,  -1.2656017035444636,
          -0.8475961795528334,   1.798823370902424,  -1.014063425986511,
          -0.20009506773487343, -1.3148984234265686,  1.4694457000064454,
        ],
      },
      'hist': [],
    }

  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)

def save_db(data:DB, fp:Path):
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)
  
def update_db(db:DB, fci:float, geo:List[float]) -> bool:
  rec = {
    'fci': fci,
    'geo': geo,
  }
  db['hist'].append(rec)
  new_best = False
  if fci < db['best']['fci']:
    db['best'] = rec
    new_best = True
    print(f'>> new best found: fci={fci}')
  return new_best


@timer
def exhaustive_search(args):
  name, init_x = read_csv(args.input_mol)
  db = load_db(DB_FILE)
  tmp_fp = Path(args.output_mol)

  try:
    for optim in OPTIM_METH:
      args.optim = optim
      for init in INIT_METH:
        args.init = init

        for i in range(args.n_times):
          print(f'>> run optim={optim}, init={init} [{i}/{args.n_times}]')
          try:
            name, best_x = run(args, name, init_x)
            best_geo = best_x.reshape(len(name), -1)
            write_csv(tmp_fp, name, best_geo)
            fci = get_fci_from_csv(args)
            new_best = update_db(db, fci, best_geo.tolist())
            if new_best: save_db(db, DB_FILE)
          except KeyboardInterrupt:
            raise
          except:
            print_exc()

  finally:
    save_db(db, DB_FILE)
    if tmp_fp.exists(): tmp_fp.unlink()


if __name__ == '__main__':
  args = get_args()
  
  args.input_mol  = BASE_PATH / 'h4.csv'
  args.output_mol = BASE_PATH / 'h4_best_tmp.csv'
  args.objective  = 'pyscf'
  args.track      = False
  args.n_times    = 10

  exhaustive_search(args)
