"""Every workspace member imports together — the check that the monorepo is
installed as one environment rather than three that happen to coexist."""


def test_workspace_members_import() -> None:
    import duck
    import ullm

    import platypus
    import taaf

    # Naming them IS the test: the imports above are the assertion, and an
    # unused-import warning would be the checker missing the point.
    assert all(module.__name__ for module in (ullm, duck, platypus, taaf))
