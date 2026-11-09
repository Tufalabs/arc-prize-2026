"""The RL trainer process for Flash never imports vLLM; servers are separate processes."""

import sys

from tests.distillation.import_probes import assert_never_imported, block_modules


def main():
    block_modules("vllm")
    import training.flash_backend  # noqa: F401
    import training.flash_entrypoint  # noqa: F401
    import training.inference  # noqa: F401
    import training.official_eval  # noqa: F401

    assert "training.native_vllm_server" not in sys.modules
    assert_never_imported("vllm")


if __name__ == "__main__":
    main()
