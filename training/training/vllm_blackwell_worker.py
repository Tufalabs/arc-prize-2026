"""Install the GDN guard in each serving process before model load or warmup."""

from vllm.v1.worker.gpu_worker import Worker

from training.blackwell_gdn import configure_blackwell_gdn


class BlackwellGdnWorker(Worker):
    def init_device(self):
        # vLLM 0.25.1 selects the device and constructs the runner here.
        # Model loading, profile_run and graph capture happen afterward.
        super().init_device()
        configure_blackwell_gdn("vllm", device=self.device)

    # Flash merged policy weights (training/flash_policy_weights.py), reached
    # through engine.collective_rpc by name: no pickled callables cross the
    # boundary. /weights/start prepares, /weights/update receives over the
    # NCCL group vLLM's init_weight_transfer_engine opened with the learner,
    # /weights/finish verifies. The receiver keeps the checkpoint census and
    # the first publication's tensor set.
    def _flash_policy_receiver(self):
        from training.flash_policy_weights import PolicyWeightReceiver

        receiver = getattr(self, "_flash_policy_weights", None)
        if receiver is None:
            receiver = PolicyWeightReceiver(self.vllm_config.model_config.model)
            self._flash_policy_weights = receiver
        return receiver

    def prepare_flash_policy_weights(self, manifest):
        return self._flash_policy_receiver().prepare(self.get_model(), manifest)

    def receive_flash_policy_weights(self, sha256):
        import torch

        group = getattr(getattr(self, "weight_transfer_engine", None), "model_update_group", None)
        if group is None:
            raise RuntimeError("the NCCL weight-transfer group with the learner is not initialized")

        def receive(buffer):
            # The learner is rank 0 of the group; one broadcast per tensor.
            group.broadcast(buffer, src=0, stream=torch.cuda.current_stream())

        return self._flash_policy_receiver().receive(self.get_model(), sha256, receive)

    def verify_flash_policy_weights(self, sha256):
        return self._flash_policy_receiver().verify(sha256)
