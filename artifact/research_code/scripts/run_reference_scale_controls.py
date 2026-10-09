"""Controls for the reference-scale concern raised in the manuscript appendix.

Three things are measured here. None of them reads an A/B outcome from the
corpus experiments.

1. A synthetic population whose conditional rows are dense. Ground truth is
   available because we can draw as many independent corpus pairs as we like.
   This separates errors of the first-order law from errors of the estimate
   that a finite reference bank supplies.
2. The same construction with a Zipfian vocabulary, so the conditional rows
   are sparse. The predicted/observed ratio is swept over reference bank size.
3. Row occupancy on the real corpora, counted at exactly the conditional rows
   the reconstructor queries under the saved evaluation masks, and the exact
   transpose identity between opposite offsets.

The population is known by construction in the synthetic parts, so a ratio
away from one is an error of the prediction, not an unmeasured property of
text. The synthetic populations are deliberately simple. They are controls
for one mechanism, not models of natural language.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from diffusion_lm_rmt import categorical_lag as dense
from diffusion_lm_rmt import sparse_categorical_lag as sparse
from diffusion_lm_rmt import sparse_categorical_influence as influence
from diffusion_lm_rmt.categorical_lag import offset_list, per_sequence_squared_error


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def stream(config: dict, *tags: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([config["seed"], *tags]))


class Population:
    """A source-clustered population we can sample from without limit.

    Each source draws a latent topic and one to four chunks. Tokens come from
    a topic-tilted local distribution mixed with a global one. The dense regime
    uses a small vocabulary and a first-order chain. The sparse regime uses a
    Zipfian vocabulary, so most conditional rows stay nearly empty at the
    corpus sizes under test.
    """

    def __init__(self, spec: dict, rng: np.random.Generator):
        self.classes = int(spec["classes"])
        self.length = int(spec["chunk_length"])
        self.zipf = spec.get("zipf_exponent")
        self.topic_weight = float(spec["topic_weight"])
        if self.zipf is None:
            base = rng.dirichlet(np.full(self.classes, .6), size=self.classes)
            topics = rng.dirichlet(np.full(self.classes, .35), size=int(spec["topics"]))
            self.kernels = []
            for topic in topics:
                mixed = (1 - self.topic_weight) * base + self.topic_weight * topic[None, :]
                self.kernels.append(np.cumsum(mixed / mixed.sum(axis=1, keepdims=True), axis=1))
        else:
            weight = 1. / np.arange(1, self.classes + 1) ** float(self.zipf)
            self.unigram = weight / weight.sum()
            self.topic_size = int(spec["topic_size"])
            self.repeat = float(spec["repeat_rate"])
            self.topics = [rng.choice(self.classes, size=self.topic_size,
                                      replace=False, p=self.unigram)
                           for _ in range(int(spec["topics"]))]

    def sources(self, count: int, rng: np.random.Generator):
        blocks, groups, cursor = [], [], 0
        for _ in range(count):
            chunks = int(rng.integers(1, 5))
            blocks.append(self._source(chunks, rng))
            groups.append(np.arange(cursor, cursor + chunks))
            cursor += chunks
        return np.concatenate(blocks), groups

    def _source(self, chunks: int, rng: np.random.Generator) -> np.ndarray:
        if self.zipf is None:
            table = self.kernels[rng.integers(len(self.kernels))]
            ids = np.empty((chunks, self.length), dtype=np.int64)
            state = rng.integers(self.classes, size=chunks)
            draws = rng.random((chunks, self.length))
            for position in range(self.length):
                ids[:, position] = state
                state = (draws[:, position][:, None] > table[state]).sum(axis=1)
            return ids
        topic = self.topics[rng.integers(len(self.topics))]
        shape = (chunks, self.length)
        local = topic[rng.integers(self.topic_size, size=shape)]
        globally = rng.choice(self.classes, size=shape, p=self.unigram)
        ids = np.where(rng.random(shape) < self.topic_weight, local, globally)
        carry = rng.random(shape) < self.repeat
        for position in range(1, self.length):
            ids[:, position] = np.where(carry[:, position],
                                        (ids[:, position - 1] + 1) % self.classes,
                                        ids[:, position])
        return ids

    def bank(self, chunks: int, rng: np.random.Generator):
        ids, groups = self.sources(int(chunks / 2.5) + 80, rng)
        while len(ids) < chunks:
            more, extra = self.sources(40, rng)
            groups += [g + len(ids) for g in extra]
            ids = np.concatenate([ids, more])
        kept = [g[g < chunks] for g in groups]
        return ids[:chunks], [g for g in kept if len(g)]

    def corpus(self, chunks: int, rng: np.random.Generator) -> np.ndarray:
        return self.bank(chunks, rng)[0]


def regime(name: str, config: dict, index: int) -> dict:
    spec = dict(config["regimes"][name])
    spec.setdefault("chunk_length", config["chunk_length"])
    alpha, ridge = float(config["smoothing"]), float(config["stacking_ridge"])
    offsets = offset_list(int(config["lag_radius"]))
    setup = stream(config, index, 0)
    population = Population(spec, setup)
    classes = population.classes

    prior_counts = np.bincount(population.corpus(int(config["prior_chunks"]), setup).ravel(),
                               minlength=classes).astype(float)
    prior = prior_counts / prior_counts.sum()
    gram, cross = sparse.fitting_equations(
        population.corpus(int(config["auxiliary_chunks"]), setup),
        population.corpus(int(config["auxiliary_chunks"]), setup),
        prior, offsets, alpha, .5, setup)
    evaluation = population.corpus(int(config["evaluation_chunks"]), setup)
    mask = setup.random(evaluation.shape) < float(config["mask_rate"])
    mask[~mask.any(axis=1), 0] = True
    rows = np.nonzero(mask)[0]

    banks = {}
    for position, chunks in enumerate(config["reference_banks"]):
        banks[chunks] = population.bank(chunks, stream(config, index, 1, position))

    skill_ids, skill_groups = banks[max(banks)]
    tables = sparse.conditional_tables(sparse.pair_counts(skill_ids, offsets, classes), prior, alpha)
    _, targets, output = sparse.masked_prediction(evaluation, mask, tables, prior,
                                                  offsets, gram, cross, ridge)
    baseline = 1 - 2 * prior[targets] + float(prior @ prior)
    skill = float(1 - output.loss(targets).mean() / baseline.mean())

    occupancy = {}
    for chunks, (ids, _) in banks.items():
        totals = np.asarray(sparse.pair_counts(ids, offsets, classes)[1].sum(axis=1)).ravel()
        queried = totals[evaluation.ravel()]
        occupancy[str(chunks)] = dict(empty_fraction=float(np.mean(queried == 0)),
                                      median_row_total=float(np.median(queried)))

    results = []
    for size_index, fitted in enumerate(config["fitted_sizes"]):
        draws = stream(config, index, 2, size_index)
        observed = []
        for _ in range(int(spec["pairs"])):
            sides = []
            for _ in range(2):
                counts = sparse.pair_counts(population.corpus(fitted, draws), offsets, classes)
                sides.append(sparse.masked_prediction(
                    evaluation, mask, sparse.conditional_tables(counts, prior, alpha),
                    prior, offsets, gram, cross, ridge)[2])
            point = sparse.squared_difference(sides[0], sides[1])
            observed.append(float(np.mean(per_sequence_squared_error(rows, point, len(evaluation)))))
        observed = np.asarray(observed)
        error = float(observed.std(ddof=1) / np.sqrt(len(observed)))
        record = dict(regime=name, classes=classes, fitted_chunks=int(fitted),
                      pairs=int(spec["pairs"]), skill=skill,
                      observed_mean=float(observed.mean()), observed_standard_error=error,
                      banks=[])
        for chunks, (ids, groups) in banks.items():
            prediction = float(np.mean(influence.source_influence_disagreement(
                ids, groups, evaluation, mask, prior, offsets, gram, cross,
                n=int(fitted), smoothing=alpha, ridge=ridge)))
            record["banks"].append(dict(
                reference_chunks=int(chunks), reference_sources=len(groups),
                bank_over_fitted=float(chunks / fitted), prediction=prediction,
                ratio=prediction / float(observed.mean()),
                ratio_lower=prediction / float(observed.mean() + 1.96 * error),
                ratio_upper=prediction / float(observed.mean() - 1.96 * error),
                **occupancy[str(chunks)]))
            print(f"  {name} n={fitted} bank={chunks}: ratio="
                  f"{record['banks'][-1]['ratio']:.3f}", flush=True)
        results.append(record)
    first, last = results[0], results[-1]
    for record in results:
        record["observed_ratio_to_smallest_fit"] = first["observed_mean"] / record["observed_mean"]
    print(f"  {name}: skill={100*skill:.2f}%, observed falls by "
          f"{first['observed_mean']/last['observed_mean']:.2f}x over a "
          f"{last['fitted_chunks']//first['fitted_chunks']}x size increase", flush=True)
    return dict(regime=name, specification=spec, results=results)


def real_corpus_rows(config: dict) -> list[dict]:
    """Row occupancy at queried contexts, and the exact opposite-offset identity."""
    offsets = offset_list(int(config["lag_radius"]))
    alpha = float(config["smoothing"])
    prepared = Path(config["prepared"])
    output = []
    for corpus, root in config["corpora"].items():
        payload = torch.load(Path(root) / f"{config['reference_split']}.pt",
                             map_location="cpu", weights_only=True)
        reference = payload["input_ids"].numpy()
        with np.load(prepared / corpus / f"{config['sparsity_case']}.npz") as saved:
            evaluation, mask = saved["eval_ids"], saved["mask"]
        classes = 50257
        transpose_gap = 0.
        counts = sparse.pair_counts(reference[:max(config["fitted_sizes"])], offsets, classes)
        for offset in offsets:
            if offset > 0:
                difference = counts[offset] - counts[-offset].T
                transpose_gap = max(transpose_gap,
                                    float(abs(difference).max()) if difference.nnz else 0.)
        for fitted in config["fitted_sizes"]:
            counts = sparse.pair_counts(reference[:fitted], offsets, classes)
            totals = {d: np.asarray(counts[d].sum(axis=1)).ravel() for d in offsets}
            rows, positions = np.nonzero(mask)
            queried = []
            for offset in offsets:
                source = positions - offset
                inside = (source >= 0) & (source < evaluation.shape[1])
                visible = np.zeros(len(rows), bool)
                visible[inside] = ~mask[rows[inside], source[inside]]
                queried.append(totals[offset][evaluation[rows[visible], source[visible]]])
            queried = np.concatenate(queried)
            output.append(dict(
                corpus=corpus, fitted_chunks=int(fitted), queried_rows=int(len(queried)),
                empty_fraction=float(np.mean(queried == 0)),
                median_row_total=float(np.median(queried)),
                below_smoothing_fraction=float(np.mean(queried < alpha)),
                mean_smoothing_share=float(np.mean(alpha / (queried + alpha))),
                opposite_offset_max_gap=transpose_gap))
            print(f"  {corpus} n={fitted}: empty={100*output[-1]['empty_fraction']:.1f}% "
                  f"median={output[-1]['median_row_total']:.0f} "
                  f"smoothing share={100*output[-1]['mean_smoothing_share']:.1f}%", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/reference_scale_controls_v1.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.perf_counter()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "code").mkdir()
    code = {}
    for path in (Path(__file__), Path(dense.__file__), Path(sparse.__file__),
                 Path(influence.__file__)):
        (output / "code" / path.name).write_bytes(path.read_bytes())
        code[str(path)] = digest(path)
    write_json(output / "config.json", config)

    print("real-corpus queried rows", flush=True)
    rows = real_corpus_rows(config)
    write_json(output / "queried_rows.json", rows)
    synthetic = []
    for index, name in enumerate(("dense", "sparse")):
        print(f"synthetic {name} regime", flush=True)
        synthetic.append(regime(name, config, index))
        write_json(output / "synthetic.json", synthetic)

    inputs = {str(args.config): digest(args.config)}
    for corpus, root in config["corpora"].items():
        inputs[str(Path(root) / f"{config['reference_split']}.pt")] = digest(
            Path(root) / f"{config['reference_split']}.pt")
        case = Path(config["prepared"]) / corpus / f"{config['sparsity_case']}.npz"
        inputs[str(case)] = digest(case)
    write_json(output / "metadata.json", dict(
        complete=True, command=" ".join(sys.argv), duration_seconds=time.perf_counter() - started,
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        reads_ab_outcomes=False, code_sha256=code, inputs_sha256=inputs,
        outputs_sha256={name: digest(output / name)
                        for name in ("queried_rows.json", "synthetic.json")},
        threads={key: os.environ.get(key) for key in
                 ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}))
    print(f"results: {output} ({time.perf_counter() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
