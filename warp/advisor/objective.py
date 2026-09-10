"""Shared design utility; evaluation always retains ER and CE separately."""


def utility(results, gold_doc_ids, objective="mixed", complete_weight=0.5):
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Design utility requires gold evidence")
    found = {r.doc_id for r in results}
    recall = len(found & gold) / len(gold)
    complete = float(gold <= found)
    if objective == "evidence_recall":
        return recall
    if objective == "complete_evidence":
        return complete
    if objective != "mixed" or not 0 <= complete_weight <= 1:
        raise ValueError("Invalid design objective or complete weight")
    return (1 - complete_weight) * recall + complete_weight * complete
