#!/usr/bin/env python3
"""Convert a Matrix Market directory into a DNBCScope project."""

import argparse
import csv
import gzip
import json
import math
import re
import struct
from pathlib import Path


def open_text(path: Path):
    # 10x text files are UTF-8. Explicitly set the encoding so Windows does
    # not decode them with the active GBK code page.
    return (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8", errors="replace")
    )


def find_file(directory: Path, names: list[str]) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing one of: {', '.join(names)}")


def rows(path: Path):
    with open_text(path) as handle:
        yield from csv.reader(handle, delimiter="\t")


def read_embedding(path: Path | None, cells: int) -> list[float]:
    if path is None:
        # A deterministic fallback layout. Real analyses should provide UMAP coordinates.
        output = []
        for index in range(cells):
            angle = index * 2.399963229728653
            radius = math.sqrt((index + 1) / cells)
            output.extend((math.cos(angle) * radius, math.sin(angle) * radius))
        return output
    values = []
    for row in rows(path):
        try:
            values.extend((float(row[-2]), float(row[-1])))
        except ValueError:
            continue
    if len(values) != cells * 2:
        raise ValueError(f"Embedding has {len(values) // 2} rows, expected {cells}")
    return values


def read_clusters(path: Path | None, cells: int) -> tuple[list[int], list[dict]]:
    if path is None:
        return [0] * cells, [{"id": 0, "name": "All cells"}]
    labels = [row[-1] for row in rows(path) if row]
    if len(labels) != cells:
        raise ValueError(f"Cluster file has {len(labels)} rows, expected {cells}")
    names = list(dict.fromkeys(labels))
    if len(names) > 255:
        raise ValueError("DNBCScope v1 supports at most 255 clusters")
    mapping = {name: index for index, name in enumerate(names)}
    return [mapping[label] for label in labels], [{"id": i, "name": name} for i, name in enumerate(names)]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Matrix Market directory containing matrix.mtx[.gz]")
    parser.add_argument("output", type=Path, help="Output .dnbc directory")
    parser.add_argument("--name", default="Matrix Market dataset")
    parser.add_argument("--genes", default="", help="Comma-separated genes to include; default: first 100 genes")
    parser.add_argument("--embedding", type=Path, help="TSV/CSV with x,y in its final two columns")
    parser.add_argument("--clusters", type=Path, help="TSV with cluster label in its final column")
    args = parser.parse_args()

    matrix_path = find_file(args.input, ["matrix.mtx.gz", "matrix.mtx"])
    feature_path = find_file(args.input, ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"])
    barcode_path = find_file(args.input, ["barcodes.tsv.gz", "barcodes.tsv"])
    features = list(rows(feature_path))
    cell_count = sum(1 for _ in rows(barcode_path))
    requested = {item.strip().upper() for item in args.genes.split(",") if item.strip()}
    selected = {}
    for index, row in enumerate(features, 1):
        name = row[1] if len(row) > 1 else row[0]
        if (requested and name.upper() in requested) or (not requested and len(selected) < 100):
            selected[index] = name

    expressions = {index: [0.0] * cell_count for index in selected}
    with open_text(matrix_path) as handle:
        dimensions_read = False
        for line in handle:
            if line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            if not dimensions_read:
                feature_rows, matrix_cells, _ = map(int, parts)
                if feature_rows != len(features) or matrix_cells != cell_count:
                    raise ValueError("Matrix dimensions do not match features/barcodes")
                dimensions_read = True
                continue
            feature_index, cell_index, value = int(parts[0]), int(parts[1]), float(parts[2])
            if feature_index in expressions:
                expressions[feature_index][cell_index - 1] = math.log1p(value)

    output = args.output
    (output / "expression").mkdir(parents=True, exist_ok=True)
    positions = read_embedding(args.embedding, cell_count)
    clusters, cluster_info = read_clusters(args.clusters, cell_count)
    (output / "positions.f32").write_bytes(struct.pack(f"<{len(positions)}f", *positions))
    (output / "clusters.u8").write_bytes(bytes(clusters))

    genes = []
    for index, name in selected.items():
        values = expressions[index]
        maximum = max(values) or 1.0
        values = [value / maximum for value in values]
        relative = f"expression/{safe_name(name)}.f32"
        (output / relative).write_bytes(struct.pack(f"<{len(values)}f", *values))
        genes.append({"name": name, "file": relative})

    manifest = {
        "format": "dnbcscope-project",
        "version": 1,
        "name": args.name,
        "assay": "Single Cell Gene Expression",
        "cells": cell_count,
        "features": len(features),
        "files": {"positions": "positions.f32", "clusters": "clusters.u8"},
        "clusters": cluster_info,
        "genes": genes,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Created {output}: {cell_count} cells, {len(features)} features, {len(genes)} browsable genes")


if __name__ == "__main__":
    main()
