from __future__ import annotations

import collections
import functools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MobilityContext:
    user_id: int
    history: tuple[int, ...]


class TrainOnlyMobilityPrior:
    """A generic candidate-list calibrator fitted only from training windows.

    The configured proposal strength is fixed. Test labels are never passed
    to this object.
    """

    def __init__(self, windows: Iterable[Sequence[int]], users: Iterable[int]):
        self.global_count = collections.Counter()
        self.user_count: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.transition_count = collections.Counter()
        self.user_transition: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.outgoing: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.user_outgoing: dict[
            tuple[int, int], collections.Counter
        ] = collections.defaultdict(collections.Counter)
        for window, user in zip(windows, users):
            ids = [int(value) for value in window]
            if len(ids) < 2:
                continue
            for poi in ids:
                self.global_count[poi] += 1
                self.user_count[int(user)][poi] += 1
            for source, target in zip(ids, ids[1:]):
                self.transition_count[(source, target)] += 1
                self.user_transition[int(user)][(source, target)] += 1
                self.outgoing[source][target] += 1
                self.user_outgoing[(int(user), source)][target] += 1

    @staticmethod
    def _log1p(value: int) -> float:
        return math.log1p(max(0, value))

    def support_score(self, candidate: int, context: MobilityContext) -> float:
        last = context.history[-1] if context.history else -1
        history_count = context.history.count(candidate)
        recency = 0.0
        if candidate in context.history:
            reverse_rank = list(reversed(context.history)).index(candidate)
            recency = 1.0 / (reverse_rank + 1.0)
        return (
            0.25 * self._log1p(self.global_count[candidate])
            + 0.50 * self._log1p(self.user_count[context.user_id][candidate])
            + 0.75 * self._log1p(self.transition_count[(last, candidate)])
            + 1.00 * self._log1p(self.user_transition[context.user_id][(last, candidate)])
            + 0.40 * self._log1p(history_count)
            + 0.35 * recency
        )

    @functools.lru_cache(maxsize=None)
    def proposals(self, context: MobilityContext, limit: int) -> tuple[int, ...]:
        """Return a bounded train-only proposal list for a mobility context."""

        if limit <= 0:
            return ()
        last = context.history[-1] if context.history else -1
        pool = set(context.history)
        pool.update(self.outgoing[last])
        pool.update(self.user_outgoing[(context.user_id, last)])
        pool.update(value for value, _ in self.user_count[context.user_id].most_common(64))
        pool.update(value for value, _ in self.global_count.most_common(64))
        ranked = sorted(
            pool,
            key=lambda candidate: (self.support_score(candidate, context), candidate),
            reverse=True,
        )
        return tuple(ranked[:limit])

    def rerank(
        self,
        candidates: Sequence[int],
        context: MobilityContext,
        alpha: float,
        *,
        augment_train_proposals: bool = False,
        proposal_limit: int = 10,
    ) -> list[int]:
        unique = list(dict.fromkeys(int(value) for value in candidates if int(value) >= 0))
        original_count = len(unique)
        if augment_train_proposals:
            unique.extend(
                candidate
                for candidate in self.proposals(context, proposal_limit)
                if candidate not in unique
            )
        scored = []
        for rank, candidate in enumerate(unique):
            # A train-only proposal starts one rank below the LLM list. The
            # validation-selected alpha must overcome that fixed penalty.
            base_rank = rank if rank < original_count else original_count + 1
            score = -float(base_rank) + float(alpha) * self.support_score(candidate, context)
            scored.append((score, -base_rank, candidate))
        return [candidate for _, _, candidate in sorted(scored, reverse=True)]



def _build_prior(train) -> TrainOnlyMobilityPrior:
    return TrainOnlyMobilityPrior(
        ([int(value) for value in frame["poi_id"]] for frame in train),
        (int(frame["user_id"].iloc[0]) for frame in train),
    )


def apply_ranking(
    records: Sequence[Mapping[str, Any]],
    *,
    train,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if contract["action"] == "model_order":
        return [dict(record) for record in records]
    if contract["action"] != "train_only_mobility_proposal_augmentation":
        raise ValueError(f"Unknown ranking action: {contract['action']}")
    if contract.get("target_use") != "none":
        raise ValueError("Ranking contract must explicitly forbid target use")
    prior = _build_prior(train)
    alphas = {int(key): float(value) for key, value in contract["alphas"].items()}
    augment = {int(value) for value in contract["augment_beams"]}
    result = []
    for source in records:
        row = dict(source)
        context = MobilityContext(
            user_id=int(source["user_id"]),
            history=tuple(int(value) for value in source["history_pois"]),
        )
        generation = {}
        for beam in (1, 3, 5, 10):
            source_generation = source["generation"][f"beam{beam}"]
            generation[f"beam{beam}"] = {
                **source_generation,
                **{
                    parser: prior.rerank(
                        [int(value) for value in source_generation[parser]],
                        context,
                        alphas[beam],
                        augment_train_proposals=beam in augment,
                        proposal_limit=int(contract["proposal_limit"]),
                    )
                    for parser in ("new_ids", "full_ids")
                },
            }
        row["generation"] = generation
        result.append(row)
    return result
