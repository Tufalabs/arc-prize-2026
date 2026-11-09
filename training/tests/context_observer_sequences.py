"""Public hook sequences shared by dependency-free and Duck-backed tests."""

from types import SimpleNamespace


def compaction_stamps(observer, token_count, *, served):
    params = SimpleNamespace(request_context={})
    observer.prepare_request(params)
    observer.prompt_measured(2, epoch=0, trigger=8, compacting=False)
    observer.request_prepared(params, epoch=0, prompt_tokens=2, target=3)
    stamps = [dict(params.request_context["arc3_lineage"])]
    observer.prepare_request(params)
    proof = token_count(9, list(range(9)))
    observer.prompt_measured(proof, epoch=0, trigger=8, compacting=True)
    if served:
        observer.prompt_measured(proof, epoch=0, trigger=8, compacting=True)
        observer.prepare_request(params)
        observer.prompt_measured(proof, epoch=0, trigger=8, compacting=True)
        observer.request_prepared(params, epoch=0, prompt_tokens=proof, target=8)
        stamps.append(dict(params.request_context["arc3_lineage"]))
    observer.history_rewritten(epoch=0)
    if served:
        observer.prepare_request(params)
        observer.prompt_measured(3, epoch=1, trigger=8, compacting=True)
        observer.history_rewritten(epoch=1)
    epoch = 2 if served else 1
    observer.prompt_measured(3, epoch=epoch, trigger=8, compacting=False)
    observer.request_prepared(
        params, epoch=epoch, prompt_tokens=3, target=8 if served else 3
    )
    stamps.append(dict(params.request_context["arc3_lineage"]))
    return stamps


def expected_stamps(*, served):
    stamps = [dict(version=1, epoch=0, kind="start", compaction_prompt_token_ids=None)]
    if served:
        stamps.append(
            dict(version=1, epoch=0, kind="continue", compaction_prompt_token_ids=None)
        )
    stamps.append(
        dict(
            version=1,
            epoch=2 if served else 1,
            kind="reset",
            compaction_prompt_token_ids=None if served else list(range(9)),
        )
    )
    return stamps
