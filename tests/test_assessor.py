from catnap import CatnapIO
from catnap.assess import Assessor


def test_false_merges(catnap_io: CatnapIO):
    assessor = Assessor(catnap_io)
    merges = list(assessor.false_merges())
    assert len(merges) == 1


def test_false_splits(catnap_io: CatnapIO):
    assessor = Assessor(catnap_io)
    splits = list(assessor.false_splits())
    assert len(splits) == 1
