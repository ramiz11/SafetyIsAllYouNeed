"""Configured candidate-list processing for Original LLM4POI."""
import copy


def shared_training_augmentation(records, train, contract):
    """Apply the tested zero-strength training proposals without changing top1.

    Training answers may enter the training prior; evaluation answers do not.
    This is the same proposal contract previously used by NYC original. Saved
    already-augmented lists are not described as raw generation outputs.
    """
    expected = {
        "action": "train_only_mobility_proposal_augmentation",
        "alphas": {str(k): 0.0 for k in (1, 3, 5, 10)},
        "augment_beams": [3, 5, 10], "proposal_limit": 10, "target_use": "none",
    }
    if contract != expected:
        raise ValueError("Unsupported shared augmentation contract")
    from .ranking import apply_ranking

    ranked = apply_ranking(records, train=train, contract=contract)
    for source, result in zip(records, ranked):
        for parser in ("new_ids", "full_ids"):
            if source["generation"]["beam1"][parser][:1] != result["generation"]["beam1"][parser][:1]:
                raise ValueError("Changed top-one prediction requires fresh Safety scoring")
    return ranked


def history_supported_alternatives(records):
    """Preserve each beam head, retaining lower candidates only if observed.

    Beam1 is untouched. Beam3/5/10 keep their current order and first candidate,
    even when it proposes a new place; alternatives must occur in that example's
    observed POI history. No target, correctness, coordinate, or Safety is read.
    The input is not mutated. This transforms already-ranked lists and makes no
    claim to recover raw model candidates from augmented artifacts.
    """
    result = copy.deepcopy(records)
    for record in result:
        observed = set(record["history_pois"])
        for beam in (3, 5, 10):
            generation = record["generation"][f"beam{beam}"]
            for parser in ("new_ids", "full_ids"):
                ids = generation[parser]
                generation[parser] = ids[:1] + [poi for poi in ids[1:] if poi in observed]
    return result
