"""Measure the local real-photo workflow. Run from the repository root.

Uses psutil from the training environment for host/process memory sampling.
Writes only the requested evidence directory and a new reference library.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import socket
import sys
from threading import Event, Thread
from time import perf_counter
from unittest.mock import patch

import psutil

from rank_homes.catalog import load_catalog, representative_photos
from rank_homes.cli import main
from rank_homes.extraction_configuration import extraction_configuration, fingerprint
from rank_homes.vision import create_extractor
import rank_homes.workflows as workflows


def inventory(root):
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def run(output, library):
    import torch

    references, candidates = Path("data/references"), Path("data/candidates")
    before = {str(root): inventory(root) for root in (references, candidates)}
    metrics = {"python": sys.version, "platform": platform.platform(),
               "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
               "physical_cpus": psutil.cpu_count(logical=False),
               "ram_bytes": psutil.virtual_memory().total, "psutil_version": version("psutil"),
               "torch_cuda": torch.version.cuda, "phases": []}
    process = psutil.Process()
    stop = Event()
    samples = []

    def sample_memory():
        while not stop.is_set():
            samples.append(process.memory_info().rss)
            stop.wait(0.01)

    sampler = Thread(target=sample_memory)
    sampler.start()
    started = perf_counter()
    try:
        start = perf_counter()
        extractor = create_extractor()
        if extractor.configuration["device"] == "cuda":
            torch.cuda.synchronize()
        metrics["first_load_seconds"] = perf_counter() - start
        metrics["device"] = extractor.configuration["device"]
        cuda = metrics["device"] == "cuda"
        metrics["gpu"] = torch.cuda.get_device_name() if cuda else None
        metrics["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory if cuda else None
        # Release models while retaining imported modules and operating-system caches.
        del extractor
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        start = perf_counter()
        extractor = create_extractor()
        if cuda:
            torch.cuda.synchronize()
        metrics["warm_load_seconds"] = perf_counter() - start
        configuration = extraction_configuration(extractor.configuration, extractor.dimension)
        metrics["configuration"] = configuration
        metrics["fingerprint"] = fingerprint(configuration)
        for root, side in ((references, "reference"), (candidates, "candidate")):
            photos = load_catalog(root, side)
            metrics[side + "_catalog"] = {"listings": len({p.apartment for p in photos}),
                "eligible_photos": len(photos), "representatives": len(representative_photos(photos)),
                "inventory": before[str(root)]}

        phase = None

        def timed(call, records, details):
            if cuda:
                torch.cuda.synchronize()
            start = perf_counter()
            result = call()
            if cuda:
                torch.cuda.synchronize()
            records.append({**details, "seconds": perf_counter() - start})
            return result

        analyze = extractor.analyze
        embed = extractor.embedder
        score = workflows.score_candidate

        def analyze_photo(photo, image):
            result = timed(lambda: analyze(photo, image), phase["photos"],
                           {"path": photo.relative_path, "side": photo.side})
            phase["photos"][-1]["crops"] = len(result.regions)
            return result

        def embed_crop(image):
            return timed(lambda: embed(image), phase["crops"], {"size": image.size})

        def score_listing(apartment, *args):
            return timed(lambda: score(apartment, *args), phase["listings"], {"apartment": apartment})

        extractor.analyze = analyze_photo
        extractor.embedder = embed_crop
        commands = [("build", ["references", "build", "--images", str(references), "--output", str(library)])]
        for label, threshold in (("inspect", "0"), ("default", "0.7")):
            commands.append((label, ["compare", "--references", str(library), "--references-images",
                                    str(references), "--listings", str(candidates), "--threshold", threshold]))
        published = None
        with patch.object(workflows, "score_candidate", score_listing):
            for label, command in commands:
                phase = {"name": label, "argv": command, "photos": [], "crops": [], "listings": []}
                start = perf_counter()
                with (output / (label + ".txt")).open("w", encoding="utf-8") as transcript:
                    with redirect_stdout(transcript), redirect_stderr(transcript):
                        status = main(command, extractor_factory=lambda: extractor)
                phase["seconds"] = perf_counter() - start
                phase["status"] = status
                metrics["phases"].append(phase)
                if status:
                    raise RuntimeError(f"{label} failed, inspect transcript")
                if label == "build":
                    published = inventory(library)
                else:
                    assert published == inventory(library), "reference library changed"
        metrics["inputs_intact_and_no_candidate_artifacts"] = before == {
            str(root): inventory(root) for root in (references, candidates)}
        assert metrics["inputs_intact_and_no_candidate_artifacts"]
        metrics["library_bytes"] = sum(p.stat().st_size for p in library.rglob("*") if p.is_file())
        metrics["library_inventory"] = published
        metrics["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if cuda else None
        metrics["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if cuda else None
    finally:
        stop.set()
        sampler.join()
        metrics["sampled_peak_rss_bytes"] = max(samples, default=0)
        metrics["total_seconds"] = perf_counter() - started
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    if args.library.exists() or args.output.exists():
        parser.error("use new output and library paths to preserve earlier evidence")
    args.output.mkdir(parents=True)
    with patch.object(socket.socket, "connect", side_effect=RuntimeError("network disabled")), \
            patch.object(socket.socket, "connect_ex", side_effect=RuntimeError("network disabled")):
        run(args.output, args.library)
