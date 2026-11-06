"""ROCm Windows fork: a faster ``importlib.metadata.packages_distributions``.

transformers calls it at import time in every process that loads it. The stdlib version infers a
package's top-level names from ``dist.files``, which stats every file listed in RECORD to drop
missing ones: ~26k stats on this venv, ~2.5-4 s per process on Windows. This one reads the same
RECORD / installed-files.txt / SOURCES.txt lines without the existence check. A RECORD that lists
files that are gone can add an extra top-level name for that distribution; nothing else changes.
"""

import collections
import importlib.metadata as md
import inspect


def _record_paths(dist):
    lines = dist._read_files_distinfo() or dist._read_files_egginfo_installed() or dist._read_files_egginfo_sources()
    if not lines:
        return ()
    import csv

    return (md.PackagePath(row[0]) for row in csv.reader(lines) if row)


def _top_level_inferred(dist):
    names = {p.parts[0] if len(p.parts) > 1 else inspect.getmodulename(p) for p in _record_paths(dist)}
    return [n for n in names if n is not None and "." not in n]


def packages_distributions():
    pkg_to_dist = collections.defaultdict(list)
    for dist in md.distributions():
        for pkg in md._top_level_declared(dist) or _top_level_inferred(dist):
            pkg_to_dist[pkg].append(dist.metadata["Name"])
    return dict(pkg_to_dist)


def install():
    md.packages_distributions = packages_distributions
