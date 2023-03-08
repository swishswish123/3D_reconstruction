


def test_arrays_equal(ground_truth, acual):
    assert len(ground_truth) == len(acual)

    for normalised_row, un_normalised_row in zip(ground_truth, acual):
        for normalised_item, un_normalised_item in zip(normalised_row, un_normalised_row):
            assert round(normalised_item) ==  round(un_normalised_item)