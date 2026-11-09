"""Flash torchrun entrypoint; runtime provisioning is external."""

from training.full_weight_entrypoint import main


if __name__ == "__main__":
    main(weight_mode="flash")
