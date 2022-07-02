import os
import re
import json
import pprint
import subprocess
from google.cloud import storage

PROJECT = "kb-encoder2"
BUCKET = "kb-encoder"
MODEL_DIR = "arc/"
OUTPUT_DIR = "arc/"

if __name__ == "__main__":
    client = storage.Client(PROJECT)
    checkpoint_steps = []
    for blob in client.list_blobs(BUCKET, prefix=MODEL_DIR):
        result = re.search(r"model\.ckpt-([0-9]+)\.data", blob.name)
        if result is not None:
            step = int(result[1])
            if step not in checkpoint_steps:
                checkpoint_steps.append(step)

    checkpoint_steps = sorted(checkpoint_steps)
    print(f"Checkpoint steps: {checkpoint_steps}")
    checkpoint_precision = {}
    for step in checkpoint_steps:
        print(f"Evaluating checkpoint {step}")
        process = subprocess.Popen(
            f'bash -c "./eval_tpu.sh {step}"', shell=True, stderr=subprocess.PIPE,
        )
        found = False
        for line in iter(process.stderr.readline, b""):
            line = line.decode("utf-8")
            print(line, end=" ")
            result = re.search(
                r"precision at step [0-9]+: ([+-]?([0-9]*[.])?[0-9]+)", line
            )
            if result is not None:
                checkpoint_precision[step] = float(result[1])
                found = True
        if not found:
            raise RuntimeError("An error occured during bash execution")

    with open("batch_eval_result.json", "w") as file:
        json.dump(checkpoint_precision, file)

    bucket = client.bucket(BUCKET)
    blob = bucket.blob(os.path.join(OUTPUT_DIR, "batch_eval_result.json"))
    blob.upload_from_filename("batch_eval_result.json")
    pprint.pprint(checkpoint_precision)
    print("Batch evaluation result saved")
