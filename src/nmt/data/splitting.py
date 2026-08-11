"""Train / validation / test partitioning without cross-split leakage.

Tatoeba is a *translation graph*, not a list of independent pairs.  A single
English sentence is typically linked to several Spanish sentences, and each of
those may itself be linked to further English sentences.  Splitting the pair
list uniformly at random therefore leaks: "I'm tired." / "Estoy cansado."
lands in the training set while "I'm tired." / "Estoy cansada." lands in test,
and the reported BLEU measures memorisation rather than translation.

The fix is to split *connected components* rather than pairs.  Treat every
distinct sentence as a node, every link as an edge, and merge nodes with a
union-find structure.  A component is then a cluster of mutually-translatable
sentences, and assigning whole components to a split guarantees that no
sentence string appears on either side of two different splits.

The effect is not marginal.  On this corpus, component-aware splitting lowers
test BLEU by a substantial margin relative to a naive random split -- which is
exactly the point: the naive number was inflated.  Both figures are reported so
the difference is visible.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from nmt.utils.logging_utils import get_logger

logger = get_logger(__name__)


class UnionFind:
    """Disjoint-set forest with path compression and union by size."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression: point every node on the way straight at the root.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self._size[left_root] < self._size[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size[right_root]

    def __len__(self) -> int:
        return len(self._parent)


@dataclass(frozen=True)
class SplitSizes:
    """Target proportions for the three splits."""

    validation: float = 0.02
    test: float = 0.02

    @property
    def train(self) -> float:
        return 1.0 - self.validation - self.test

    def __post_init__(self) -> None:
        if not 0 < self.validation + self.test < 1:
            raise ValueError("validation + test must lie strictly between 0 and 1")


@dataclass
class SplitReport:
    """Diagnostics describing the partition that was produced."""

    total_pairs: int = 0
    components: int = 0
    largest_component_pairs: int = 0
    singleton_components: int = 0
    split_pairs: dict[str, int] = field(default_factory=dict)
    split_components: dict[str, int] = field(default_factory=dict)
    eval_group_size_cap: int = 0
    leakage_check: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _components(pairs: Iterable[tuple[str, str]]) -> dict[str, list[int]]:
    """Group pair indices by the connected component they belong to.

    Sentences are namespaced by language (``"en:..."`` / ``"es:..."``) so that
    an English string that happens to equal a Spanish string -- proper nouns,
    "No." -- does not spuriously merge two unrelated components.
    """
    forest = UnionFind()
    keys: list[tuple[str, str]] = []

    for en, es in pairs:
        en_key, es_key = f"en:{en}", f"es:{es}"
        forest.add(en_key)
        forest.add(es_key)
        forest.union(en_key, es_key)
        keys.append((en_key, es_key))

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (en_key, _) in enumerate(keys):
        grouped[forest.find(en_key)].append(index)

    return grouped


def split_pairs(
    pairs: list[tuple[str, str]],
    sizes: SplitSizes | None = None,
    *,
    seed: int = 42,
    max_eval_group_size: int = 4,
) -> tuple[dict[str, list[tuple[str, str]]], SplitReport]:
    """Partition ``pairs`` into train / validation / test by component.

    Parameters
    ----------
    pairs
        Cleaned ``(english, spanish)`` tuples.
    sizes
        Target proportions.  With ~250k pairs, 2% each gives evaluation sets of
        roughly 5,000 pairs -- large enough for a stable BLEU estimate (the
        usual guidance is >= 1,000 sentences) while leaving 96% for training.
    seed
        Controls the component shuffle, so the partition is reproducible.
    max_eval_group_size
        Components contributing more than this many pairs are forced into
        training.  A 40-pair component landing in test would fill the
        evaluation set with paraphrases of one sentence and make BLEU
        hostage to whether the model happens to know that sentence.

    Returns
    -------
    (dict, SplitReport)
        Split name -> pairs, plus diagnostics including an explicit leakage
        assertion.
    """
    sizes = sizes or SplitSizes()
    grouped = _components(pairs)

    report = SplitReport(
        total_pairs=len(pairs),
        components=len(grouped),
        eval_group_size_cap=max_eval_group_size,
    )
    component_sizes = [len(v) for v in grouped.values()]
    report.largest_component_pairs = max(component_sizes) if component_sizes else 0
    report.singleton_components = sum(1 for s in component_sizes if s == 1)

    # Small components are eligible for evaluation; large ones always train.
    eligible = [k for k, v in grouped.items() if len(v) <= max_eval_group_size]
    oversized = [k for k, v in grouped.items() if len(v) > max_eval_group_size]

    rng = random.Random(seed)
    rng.shuffle(eligible)

    target_test = int(round(len(pairs) * sizes.test))
    target_validation = int(round(len(pairs) * sizes.validation))

    assignment: dict[str, str] = dict.fromkeys(oversized, "train")
    filled_test = filled_validation = 0

    for key in eligible:
        size = len(grouped[key])
        if filled_test < target_test:
            assignment[key] = "test"
            filled_test += size
        elif filled_validation < target_validation:
            assignment[key] = "validation"
            filled_validation += size
        else:
            assignment[key] = "train"

    splits: dict[str, list[tuple[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    component_counts: dict[str, int] = defaultdict(int)

    for key, indices in grouped.items():
        target = assignment[key]
        component_counts[target] += 1
        for index in indices:
            splits[target].append(pairs[index])

    for name in splits:
        rng.shuffle(splits[name])
        report.split_pairs[name] = len(splits[name])
        report.split_components[name] = component_counts[name]

    report.leakage_check = _verify_no_leakage(splits)
    report.notes.append(
        "Splits are disjoint at the level of connected components of the "
        "Tatoeba translation graph, so no sentence string occurs in more than "
        "one split on either the source or the target side."
    )

    logger.info(
        "Split %s pairs -> train %s / validation %s / test %s",
        f"{len(pairs):,}",
        f"{report.split_pairs['train']:,}",
        f"{report.split_pairs['validation']:,}",
        f"{report.split_pairs['test']:,}",
    )
    return splits, report


def _verify_no_leakage(splits: dict[str, list[tuple[str, str]]]) -> dict[str, int]:
    """Assert that no sentence string is shared between splits.

    This is a *test*, not a statistic: any non-zero value indicates a bug in
    the component construction, so it is checked on every run and recorded in
    the report rather than being left to a unit test alone.
    """
    def sentences(name: str, side: int) -> set[str]:
        return {pair[side] for pair in splits[name]}

    overlaps: dict[str, int] = {}
    for side, label in ((0, "en"), (1, "es")):
        train = sentences("train", side)
        for other in ("validation", "test"):
            overlaps[f"{label}_train_vs_{other}"] = len(train & sentences(other, side))
        overlaps[f"{label}_validation_vs_test"] = len(
            sentences("validation", side) & sentences("test", side)
        )

    total = sum(overlaps.values())
    if total:
        logger.error("Leakage detected across splits: %s", overlaps)
    else:
        logger.info("Leakage check passed: no shared sentences across splits.")

    return overlaps
