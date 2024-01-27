import pytest

from src.services.dao import DAO


@pytest.fixture
def dao() -> DAO:
    return DAO()


def test_load_data(dao: DAO) -> None:
    assert dao.train_features.shape == (1168, 4)
    assert dao.train_label.shape == (1168,)
    assert dao.test_features.shape == (292, 4)
    assert dao.test_label.shape == (292,)


def test_get_row(dao: DAO) -> None:
    features, price = dao.get_row(0, False)

    assert features.to_dict() == {
        "OverallQual": 6,
        "YearBuilt": 1963,
        "TotalBsmtSF": 1059,
        "GrLivArea": 1068,
    }
    assert price == 154500
