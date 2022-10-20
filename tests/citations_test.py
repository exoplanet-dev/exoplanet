__all__ = ["test_basic"]

from exoplanet.citations import CITATIONS, get_citations_for_model
from exoplanet.compat import pm
from exoplanet.light_curves import LimbDarkLightCurve


def test_basic():
    with pm.Model() as model:
        txt, bib = get_citations_for_model()
    assert txt == ""
    assert bib == ""

    with pm.Model() as model:
        LimbDarkLightCurve(0.5, 0.2)
        txt, bib = get_citations_for_model()
    for k in ["exoplanet", "starry"]:
        assert all(v in bib for v in CITATIONS[k][0])
        assert CITATIONS[k][1] in bib

    txt1, bib1 = get_citations_for_model(model=model)
    assert txt == txt1
    assert bib == bib1
