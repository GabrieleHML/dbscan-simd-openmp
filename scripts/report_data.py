#!/usr/bin/env python3
"""Turns the benchmark results (results/*.csv) into the tables and chart data of the report.

Writes docs/relazione/dati/*.csv, read by pgfplots, and docs/relazione/tabelle/*.tex,
included by the chapters, so that `make bench report` updates every number in the PDF.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "docs" / "relazione" / "dati"
TABLES = ROOT / "docs" / "relazione" / "tabelle"


def load(name: str) -> list[dict]:
    with open(RESULTS / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def pick(rows: list[dict], **conditions) -> dict:
    found = [r for r in rows if all(str(r[k]) == str(v) for k, v in conditions.items())]
    if len(found) != 1:
        raise LookupError(f"{len(found)} rows match {conditions}")
    return found[0]


def write_csv(name: str, header: list[str], rows: list[list], delimiter: str = ",") -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(header)
        w.writerows(rows)


def num(x: float, digits: int = 2) -> str:
    """Italian decimal comma, for the tables."""
    return f"{x:.{digits}f}".replace(".", ",")


def thousands(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def variants() -> None:
    rows = load("variants")
    steps = [
        ("Versione d'esame", dict(program="exam-2024")),
        ("Ricerca esaustiva, C", dict(program="dbscan64", index="linear", simd="False")),
        ("Ricerca esaustiva, AVX", dict(program="dbscan64", index="linear", simd="True")),
        ("Ball Tree, C", dict(program="dbscan64", index="balltree", simd="False")),
        ("Ball Tree, AVX", dict(program="dbscan64", index="balltree", simd="True")),
        ("Ball Tree, AVX, 4 thread", dict(program="dbscan64-omp", threads=4)),
        ("Ball Tree, AVX, 8 thread", dict(program="dbscan64-omp", threads=8)),
        ("scikit-learn, 1 thread", dict(program="scikit-learn", threads=1)),
        ("scikit-learn, 8 thread", dict(program="scikit-learn", threads=8)),
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    for dataset, table in [("blobs", "varianti_blobs"), ("exam", "varianti_esame")]:
        base = float(pick(rows, dataset=dataset, program="exam-2024")["time"])
        lines, chart = [], []
        for label, cond in steps:
            r = pick(rows, dataset=dataset, **cond)
            t = float(r["time"])
            evals = thousands(int(r["distance_evals"])) if r["distance_evals"] else "--"
            lines.append(f"{label} & {num(t, 3)} & {num(base / t, 1)}$\\times$ & {evals} \\\\")
            chart.append([label, f"{t:.4f}", f"{base / t:.2f}"])
        (TABLES / f"{table}.tex").write_text("\n".join(lines) + "\n")
        # Semicolons: the labels contain commas, and pgfplots does not understand quoted fields
        write_csv(table, ["variante", "tempo", "speedup"], chart[::-1], delimiter=";")


def architectures() -> None:
    """The same steps with the 32-bit SSE build and the 64-bit AVX build."""
    rows = [r for r in load("variants") if r["dataset"] == "blobs"]
    if not any(r["program"] == "dbscan32" for r in rows):
        return
    steps = [
        ("Ricerca esaustiva, C", dict(index="linear", simd="False"), ""),
        ("Ricerca esaustiva, SIMD", dict(index="linear", simd="True"), ""),
        ("Ball Tree, C", dict(index="balltree", simd="False"), ""),
        ("Ball Tree, SIMD", dict(index="balltree", simd="True"), ""),
        ("Ball Tree, SIMD, 4 thread", dict(threads=4), "-omp"),
        ("Ball Tree, SIMD, 8 thread", dict(threads=8), "-omp"),
    ]
    lines = []
    for label, cond, suffix in steps:
        t32 = float(pick(rows, program="dbscan32" + suffix, **cond)["time"])
        t64 = float(pick(rows, program="dbscan64" + suffix, **cond)["time"])
        lines.append(f"{label} & {num(t32, 3)} & {num(t64, 3)} & {num(t64 / t32, 2)} \\\\")
    (TABLES / "architetture.tex").write_text("\n".join(lines) + "\n")


def threads() -> None:
    rows = load("threads")
    out = []
    counts = sorted({int(r["threads"]) for r in rows})
    t1 = {i: float(pick(rows, index=i, threads=1)["time"]) for i in ["balltree", "linear"]}
    for t in counts:
        bt = float(pick(rows, index="balltree", threads=t)["time"])
        li = float(pick(rows, index="linear", threads=t)["time"])
        out.append([t, f"{t1['balltree'] / bt:.3f}", f"{t1['linear'] / li:.3f}", t])
    write_csv("threads", ["threads", "balltree", "linear", "ideale"], out)


def scaling() -> None:
    rows = load("scaling")
    out = []
    for n in sorted({int(r["n"]) for r in rows}):
        def t(**c):
            try:
                return f"{float(pick(rows, n=n, **c)['time']):.4f}"
            except LookupError:
                return "nan"
        out.append([n, t(program="dbscan64", index="linear"), t(program="dbscan64", index="balltree"),
                    t(program="dbscan64-omp"), t(program="scikit-learn")])
    write_csv("scaling", ["n", "linear", "balltree", "omp8", "sklearn"], out)


def leaf_size() -> None:
    rows = load("leaf_size")
    out = []
    dims = ["2", "16", "64"]
    series = {d: [r for r in rows if r["d"] == d] for d in dims}
    best = {d: min(float(r["time"]) for r in s) for d, s in series.items()}
    for leaf in sorted({int(r["leaf_size"]) for r in rows}):
        out.append([leaf] + [f"{float(pick(series[d], leaf_size=leaf)['time']) / best[d]:.3f}" for d in dims])
    write_csv("leaf_size", ["leaf"] + [f"d{d}" for d in dims], out)


def dimensions() -> None:
    rows = load("dimensions")
    out, lines = [], []
    for d in sorted({int(r["d"]) for r in rows}):
        li, bt = pick(rows, d=d, index="linear"), pick(rows, d=d, index="balltree")
        frac = int(bt["distance_evals"]) / int(li["distance_evals"])
        speedup = float(li["time"]) / float(bt["time"])
        out.append([d, f"{100 * frac:.2f}", f"{speedup:.2f}"])
        lines.append(f"{d} & {num(float(li['time']), 3)} & {num(float(bt['time']), 3)} & {num(speedup, 1)}$\\times$ & "
                     f"{num(100 * frac, 1)}\\% & {bt['tree_depth']} \\\\")
    write_csv("dimensions", ["d", "percentuale", "speedup"], out)
    (TABLES / "dimensioni.tex").write_text("\n".join(lines) + "\n")


def optimization() -> None:
    rows = load("optimization")
    lines = []
    for index, name in [("linear", "Ricerca esaustiva"), ("balltree", "Ball Tree")]:
        for opt in ["-O0", "-O2"]:
            c = float(pick(rows, index=index, opt=opt, simd="False")["time"])
            s = float(pick(rows, index=index, opt=opt, simd="True")["time"])
            lines.append(f"{name} & \\codice{{{opt}}} & {num(c, 3)} & {num(s, 3)} & {num(c / s, 2)}$\\times$ \\\\")
    (TABLES / "compilatore.tex").write_text("\n".join(lines) + "\n")


def environment() -> None:
    env = json.loads((RESULTS / "environment.json").read_text())
    rows = [
        ("Processore", f"{env['cpu']} ({env['cores']} core, {env['logical_cpus']} thread)"),
        ("Sistema operativo", env["os"].replace("_", "\\_")),
        ("Compilatore", env["gcc"]),
        ("Assemblatore", env["nasm"].split(" compiled")[0]),
        ("Riferimento", f"Python {env['python']}, NumPy {env['numpy']}, scikit-learn {env['scikit-learn']}"),
        ("Data delle misure", env["date"]),
    ]
    (TABLES / "ambiente.tex").write_text("\n".join(f"{k} & {v} \\\\" for k, v in rows) + "\n")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for step in [variants, architectures, threads, scaling, leaf_size, dimensions, optimization, environment]:
        try:
            step()
        except FileNotFoundError as e:
            print(f"skipped {step.__name__}: {e.filename} missing")
    print(f"written {DATA.relative_to(ROOT)} and {TABLES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
