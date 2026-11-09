"""Every workspace member imports together — the check that the monorepo is
installed as one environment rather than five that happen to coexist."""


def test_workspace_members_import() -> None:
    # Proves all members are installed together; real cross-member tests
    # live beside this file.
    import ullm

    import duck
    import platypus
    import reference_agents

    # Naming them IS the test: the imports above are the assertion, and an
    # unused-import warning would be the checker missing the point.
    assert all(m.__name__ for m in (ullm, duck, platypus, reference_agents))
