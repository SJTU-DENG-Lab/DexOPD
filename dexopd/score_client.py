"""Persistent score workers allow benchmark dependencies to use separate environments."""
import json
from pathlib import Path
import subprocess


class Scorer:
    def __init__(self, task, upstream, assets, device, python):
        script = Path(__file__).with_name("score_worker.py")
        self.process = subprocess.Popen(
            [python, "-u", str(script), task, str(upstream), str(assets.resolve()), str(device)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Score worker closed its output (exit {self.process.poll()}).")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def score(self, pixels, prompts, metadata):
        if len(pixels) != len(prompts) or len(prompts) != len(metadata):
            raise ValueError("Scoring rows and pixels must be aligned.")
        header = json.dumps({"shape": pixels.shape, "prompts": prompts, "metadata": metadata})
        stream = self.process.stdin
        stream.write(header.encode("utf-8") + b"\n")
        stream.write(pixels.tobytes(order="C"))
        stream.flush()
        scores = self.read()["scores"]
        if len(scores) != len(prompts):
            raise ValueError("Score worker returned incomplete results.")
        return scores

    def __enter__(self):
        try:
            if self.read().get("ready") is not True:
                raise RuntimeError("Score worker failed to initialize.")
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            self.process.stdout.close()
