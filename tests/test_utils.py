from mavebay import utils


def test_load_dataset():
    """
    test loading dataset for Amyloid Beta returns correct shapes.
    """
    x, y, L, C = utils.load_dataset(
        filename="examples/datasets/amyloid_data.csv.gz",
        alphabet="protein*",
        verbose=False,
    )
    assert x.shape == (16066, 42, 21)
    assert y.shape == (16066, 1)
    assert L == 42
    assert C == 20
