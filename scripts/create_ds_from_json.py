import argparse
import tarfile
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive_path', type=str, required=True)
    parser.add_argument('--ds_path', type=str, required=True)
    args = parser.parse_args()

    Path(args.ds_path).mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.archive_path, 'r') as tar:
        # iterate over all members
        member = tar.next()
        if "fischer" in args.archive_path:
            pbar = tqdm(total=23398)
        elif "swb" in args.archive_path:
            pbar = tqdm(total=4870)
        while member is not None:
            if member.isfile() and (not "swb" in args.archive_path or "dual" in member.name):
                save_path = Path(args.ds_path) / (member.name.split("/")[-1][:-5] + '.npy')
                if save_path.exists():
                    pbar.update(1)
                    member = tar.next()
                    continue
                # get member file object
                f = tar.extractfile(member)
                # read the file object
                content = f.read()
                content = json.loads(content)
                pitch = content['pitch']["dio_sm"]
                pitch = np.array(pitch)
                pitch = pitch.astype(np.float32)
                energy = content['energy']
                energy = np.array(energy)
                energy = energy.astype(np.float32)
                voiced = content['vad']["dio_sm"]
                voiced = np.array(voiced)
                voiced = voiced.astype(np.float32)
                # stack features
                features = np.stack([pitch, energy, voiced], axis=0)
                # save features
                np.save(Path(args.ds_path) / (member.name.split("/")[-1][:-5] + '.npy'), features)
            pbar.update(1)
            member = tar.next()