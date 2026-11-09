"""CI has no FLA and the trainer process has no vLLM; the Flash modules import anyway."""

from tests.distillation.import_probes import assert_never_imported, block_modules


def main():
    block_modules("fla", "vllm")
    import training.flash.decision_replay  # noqa: F401
    import training.flash.dense_lora  # noqa: F401
    import training.flash.model  # noqa: F401
    import training.flash.selector_scoring  # noqa: F401
    import training.flash_ppo  # noqa: F401

    assert_never_imported("vllm")


if __name__ == "__main__":
    main()
