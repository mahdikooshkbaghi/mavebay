from mavebay import utils


def test_load_dataset():
    """
    test loading dataset for Amyloid Beta returns correct shapes.
    """
    x, y, L, C, alphabet, cons_seq = utils.load_dataset(
        filename="https://github.com/jbkinney/mavenn/raw/master/mavenn/examples/datasets/amyloid_data.csv.gz",  # noqa
        alphabet="protein*",
        verbose=False,
    )
    assert x.shape == (16066, 42, 21)
    assert y.shape == (16066, 1)
    assert L == 42
    assert C == 21
    assert alphabet == "protein*"
    assert cons_seq == "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
