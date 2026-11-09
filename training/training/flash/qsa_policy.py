"""Explicit QSA execution policy, independent of selector semantics."""

QSA_KERNELS = ("flex", "indexed_backward", "physical_sparse", "group_sparse")


def validate_qsa_kernel(value=None, attention_backend="flex_qsa", *, packed=False):
    if value is None:
        value = "physical_sparse" if attention_backend == "flex_qsa" else "flex"
    if value not in QSA_KERNELS:
        raise ValueError(f"qsa_kernel={value!r}; allowed values are {QSA_KERNELS} or None")
    if value != "flex" and attention_backend != "flex_qsa":
        raise ValueError("Sparse qsa_kernel requires attention_backend=flex_qsa")
    if value not in ("flex", "group_sparse") and packed:
        raise ValueError("Sparse qsa_kernel supports one unpacked document only")
    return value
