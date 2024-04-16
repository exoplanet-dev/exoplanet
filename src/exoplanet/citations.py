__all__ = ["add_citations_to_model", "CITATIONS"]

import logging
import textwrap

from exoplanet.compat import USING_PYMC3, pm


def add_citations_to_model(citations, model=None):
    try:
        model = pm.modelcontext(model)
        if not hasattr(model, "__citations__"):
            model.__citations__ = dict()
        for k in citations:
            model.__citations__[k] = CITATIONS[k]

    except TypeError:
        pass


def get_citations_for_model(model=None, width=79):
    """Get the citations for the components used an exoplanet PyMC model

    Returns: The acknowledgement text for exoplanet and its dependencies and a
    string containing the BibTeX entries for the citations in the
    acknowledgement.

    """
    model = pm.modelcontext(model)
    if not hasattr(model, "__citations__"):
        logging.warning("no citations registered with model")
        return "", ""

    cite = list(CITATIONS["arviz"][0])
    bib = [
        CITATIONS["exoplanet"][1],
        CITATIONS["arviz"][1],
    ]
    if USING_PYMC3:
        cite += list(CITATIONS["pymc3"][0]) + list(CITATIONS["theano"][0])
        bib += [CITATIONS["pymc3"][1], CITATIONS["theano"][1]]
    for _, v in model.__citations__.items():
        cite += list(v[0])
        bib.append(v[1])

    txt = (
        r"This research made use of \textsf{{exoplanet}} "
        r"\citep{{exoplanet:joss, exoplanet:zenodo}} and its dependencies "
        r"\citep{{{0}}}."
    )
    txt = txt.format(", ".join(sorted(cite)))
    txt = textwrap.wrap(txt, width=width)

    return "\n".join(txt), "\n".join(bib)


CITATIONS = {
    "exoplanet": (
        ("exoplanet:joss", "exoplanet:zenodo"),
        r"""
@article{exoplanet:joss,
       author = {{Foreman-Mackey}, Daniel and {Luger}, Rodrigo and {Agol}, Eric
                and {Barclay}, Thomas and {Bouma}, Luke G. and {Brandt},
                Timothy D. and {Czekala}, Ian and {David}, Trevor J. and
                {Dong}, Jiayin and {Gilbert}, Emily A. and {Gordon}, Tyler A.
                and {Hedges}, Christina and {Hey}, Daniel R. and {Morris},
                Brett M. and {Price-Whelan}, Adrian M. and {Savel}, Arjun B.},
        title = "{exoplanet: Gradient-based probabilistic inference for
                  exoplanet data \& other astronomical time series}",
      journal = {arXiv e-prints},
         year = 2021,
        month = may,
          eid = {arXiv:2105.01994},
        pages = {arXiv:2105.01994},
archivePrefix = {arXiv},
       eprint = {2105.01994},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210501994F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@misc{exoplanet:zenodo,
  author = {Daniel Foreman-Mackey and Arjun Savel and Rodrigo Luger  and
            Eric Agol and Ian Czekala and Adrian Price-Whelan and
            Christina Hedges and Emily Gilbert and Luke Bouma
            and Timothy D. Brandt and Tom Barclay},
   title = {exoplanet-dev/exoplanet v0.5.1},
   month = jun,
    year = 2021,
     doi = {10.5281/zenodo.1998447},
     url = {https://doi.org/10.5281/zenodo.1998447}
}
""",
    ),
    "pymc3": (
        ("exoplanet:pymc3",),
        r"""
@article{exoplanet:pymc3,
    title={Probabilistic programming in Python using PyMC3},
   author={Salvatier, John and Wiecki, Thomas V and Fonnesbeck, Christopher},
  journal={PeerJ Computer Science},
   volume={2},
    pages={e55},
     year={2016},
publisher={PeerJ Inc.}
}
""",
    ),
    "theano": (
        ("exoplanet:theano",),
        r"""
@article{exoplanet:theano,
    title="{Theano: A {Python} framework for fast computation of mathematical
            expressions}",
   author={{Theano Development Team}},
  journal={arXiv e-prints},
   volume={abs/1605.02688},
     year=2016,
    month=may,
      url={http://arxiv.org/abs/1605.02688}
}
""",
    ),
    "arviz": (
        ("exoplanet:arviz",),
        r"""
@article{exoplanet:arviz,
    title={{ArviZ} a unified library for exploratory analysis of {Bayesian}
           models in {Python}},
   author={Kumar, Ravin and Carroll, Colin and Hartikainen, Ari and Martin,
           Osvaldo A.},
  journal={The Journal of Open Source Software},
     year=2019,
      doi={10.21105/joss.01143},
      url={http://joss.theoj.org/papers/10.21105/joss.01143}
}
""",
    ),
    "astropy": (
        ("exoplanet:astropy13", "exoplanet:astropy18"),
        r"""
@article{exoplanet:astropy13,
   author = {{Astropy Collaboration} and {Robitaille}, T.~P. and {Tollerud},
             E.~J. and {Greenfield}, P. and {Droettboom}, M. and {Bray}, E. and
             {Aldcroft}, T. and {Davis}, M. and {Ginsburg}, A. and
             {Price-Whelan}, A.~M. and {Kerzendorf}, W.~E. and {Conley}, A. and
             {Crighton}, N. and {Barbary}, K. and {Muna}, D. and {Ferguson}, H.
             and {Grollier}, F. and {Parikh}, M.~M. and {Nair}, P.~H. and
             {Unther}, H.~M. and {Deil}, C. and {Woillez}, J. and {Conseil}, S.
             and {Kramer}, R. and {Turner}, J.~E.~H. and {Singer}, L. and
             {Fox}, R. and {Weaver}, B.~A. and {Zabalza}, V. and {Edwards},
             Z.~I. and {Azalee Bostroem}, K. and {Burke}, D.~J. and {Casey},
             A.~R. and {Crawford}, S.~M. and {Dencheva}, N. and {Ely}, J. and
             {Jenness}, T. and {Labrie}, K. and {Lim}, P.~L. and
             {Pierfederici}, F. and {Pontzen}, A. and {Ptak}, A. and {Refsdal},
             B. and {Servillat}, M. and {Streicher}, O.},
    title = "{Astropy: A community Python package for astronomy}",
  journal = {\aap},
     year = 2013,
    month = oct,
   volume = 558,
    pages = {A33},
      doi = {10.1051/0004-6361/201322068},
   adsurl = {http://adsabs.harvard.edu/abs/2013A%26A...558A..33A},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@article{exoplanet:astropy18,
   author = {{Astropy Collaboration} and {Price-Whelan}, A.~M. and
             {Sip{\H o}cz}, B.~M. and {G{\"u}nther}, H.~M. and {Lim}, P.~L. and
             {Crawford}, S.~M. and {Conseil}, S. and {Shupe}, D.~L. and
             {Craig}, M.~W. and {Dencheva}, N. and {Ginsburg}, A. and
             {VanderPlas}, J.~T. and {Bradley}, L.~D. and
             {P{\'e}rez-Su{\'a}rez}, D. and {de Val-Borro}, M.
             and {Aldcroft}, T.~L. and {Cruz}, K.~L. and {Robitaille}, T.~P.
             and {Tollerud}, E.~J. and {Ardelean}, C. and {Babej}, T. and
             {Bach}, Y.~P. and {Bachetti}, M. and {Bakanov}, A.~V. and
             {Bamford}, S.~P. and {Barentsen}, G. and {Barmby}, P. and
             {Baumbach}, A. and {Berry}, K.~L.  and {Biscani}, F. and
             {Boquien}, M. and {Bostroem}, K.~A. and {Bouma}, L.~G. and
             {Brammer}, G.~B. and {Bray}, E.~M. and {Breytenbach}, H. and
             {Buddelmeijer}, H. and {Burke}, D.~J. and {Calderone}, G. and
             {Cano Rodr{\'{\i}}guez}, J.~L. and {Cara}, M. and {Cardoso},
             J.~V.~M. and {Cheedella}, S. and {Copin}, Y. and {Corrales}, L.
             and {Crichton}, D. and {D'Avella}, D. and {Deil}, C. and
             {Depagne}, {\'E}. and {Dietrich}, J.~P. and {Donath}, A. and
             {Droettboom}, M. and {Earl}, N. and {Erben}, T. and {Fabbro}, S.
             and {Ferreira}, L.~A. and {Finethy}, T. and {Fox}, R.~T. and
             {Garrison}, L.~H. and {Gibbons}, S.~L.~J. and {Goldstein}, D.~A.
             and {Gommers}, R. and {Greco}, J.~P. and {Greenfield}, P. and
             {Groener}, A.~M. and {Grollier}, F. and {Hagen}, A. and {Hirst},
             P. and {Homeier}, D. and {Horton}, A.~J. and {Hosseinzadeh}, G.
             and {Hu}, L. and {Hunkeler}, J.~S. and {Ivezi{\'c}}, {\v Z}. and
             {Jain}, A. and {Jenness}, T. and {Kanarek}, G. and {Kendrew}, S.
             and {Kern}, N.~S. and {Kerzendorf}, W.~E. and {Khvalko}, A. and
             {King}, J. and {Kirkby}, D. and {Kulkarni}, A.~M. and {Kumar}, A.
             and {Lee}, A.  and {Lenz}, D.  and {Littlefair}, S.~P. and {Ma},
             Z. and {Macleod}, D.~M. and {Mastropietro}, M. and {McCully}, C.
             and {Montagnac}, S. and {Morris}, B.~M. and {Mueller}, M. and
             {Mumford}, S.~J. and {Muna}, D. and {Murphy}, N.~A. and {Nelson},
             S. and {Nguyen}, G.~H. and {Ninan}, J.~P. and {N{\"o}the}, M. and
             {Ogaz}, S. and {Oh}, S. and {Parejko}, J.~K.  and {Parley}, N. and
             {Pascual}, S. and {Patil}, R. and {Patil}, A.~A.  and {Plunkett},
             A.~L. and {Prochaska}, J.~X. and {Rastogi}, T. and {Reddy Janga},
             V. and {Sabater}, J.  and {Sakurikar}, P. and {Seifert}, M. and
             {Sherbert}, L.~E. and {Sherwood-Taylor}, H. and {Shih}, A.~Y. and
             {Sick}, J. and {Silbiger}, M.~T. and {Singanamalla}, S. and
             {Singer}, L.~P. and {Sladen}, P.~H. and {Sooley}, K.~A. and
             {Sornarajah}, S. and {Streicher}, O. and {Teuben}, P. and
             {Thomas}, S.~W. and {Tremblay}, G.~R. and {Turner}, J.~E.~H. and
             {Terr{\'o}n}, V.  and {van Kerkwijk}, M.~H. and {de la Vega}, A.
             and {Watkins}, L.~L. and {Weaver}, B.~A. and {Whitmore}, J.~B. and
             {Woillez}, J.  and {Zabalza}, V. and {Astropy Contributors}},
    title = "{The Astropy Project: Building an Open-science Project and Status
              of the v2.0 Core Package}",
  journal = {\aj},
     year = 2018,
    month = sep,
   volume = 156,
    pages = {123},
      doi = {10.3847/1538-3881/aabc4f},
   adsurl = {http://adsabs.harvard.edu/abs/2018AJ....156..123A},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    ),
    "espinoza18": (
        ("exoplanet:espinoza18",),
        r"""
@article{exoplanet:espinoza18,
   author = {{Espinoza}, N.},
    title = "{Efficient Joint Sampling of Impact Parameters and Transit Depths
              in Transiting Exoplanet Light Curves}",
  journal = {Research Notes of the American Astronomical Society},
     year = 2018,
    month = nov,
   volume = 2,
   number = 4,
    pages = {209},
      doi = {10.3847/2515-5172/aaef38},
   adsurl = {http://adsabs.harvard.edu/abs/2018RNAAS...2d.209E},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    ),
    "kipping13": (
        ("exoplanet:kipping13",),
        r"""
@ARTICLE{exoplanet:kipping13,
   author = {{Kipping}, D.~M.},
    title = "{Efficient, uninformative sampling of limb darkening coefficients
              for two-parameter laws}",
  journal = {\mnras},
     year = 2013,
    month = nov,
   volume = 435,
    pages = {2152-2160},
      doi = {10.1093/mnras/stt1435},
   adsurl = {http://adsabs.harvard.edu/abs/2013MNRAS.435.2152K},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    ),
    "kipping13b": (
        ("exoplanet:kipping13b",),
        r"""
@ARTICLE{exoplanet:kipping13b,
       author = {{Kipping}, D.~M.},
        title = "{Parametrizing the exoplanet eccentricity distribution with
                  the beta  distribution.}",
      journal = {\mnras},
         year = "2013",
        month = jul,
       volume = 434,
        pages = {L51-L55},
          doi = {10.1093/mnrasl/slt075},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2013MNRAS.434L..51K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    ),
    "starry": (
        ("exoplanet:luger18", "exoplanet:agol20"),
        r"""
@article{exoplanet:luger18,
   author = {{Luger}, R. and {Agol}, E. and {Foreman-Mackey}, D. and {Fleming},
             D.~P. and {Lustig-Yaeger}, J. and {Deitrick}, R.},
    title = "{starry: Analytic Occultation Light Curves}",
  journal = {\aj},
     year = 2019,
    month = feb,
   volume = 157,
    pages = {64},
      doi = {10.3847/1538-3881/aae8e5},
   adsurl = {http://adsabs.harvard.edu/abs/2019AJ....157...64L},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@article{exoplanet:agol20,
   author = {{Agol}, Eric and {Luger}, Rodrigo and {Foreman-Mackey}, Daniel},
    title = "{Analytic Planetary Transit Light Curves and Derivatives for
              Stars with Polynomial Limb Darkening}",
  journal = {\aj},
     year = 2020,
    month = mar,
   volume = {159},
   number = {3},
    pages = {123},
      doi = {10.3847/1538-3881/ab4fee},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2020AJ....159..123A},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    ),
    "celerite": (
        ("exoplanet:foremanmackey17", "exoplanet:foremanmackey18"),
        r"""
@article{exoplanet:foremanmackey17,
   author = {{Foreman-Mackey}, D. and {Agol}, E. and {Ambikasaran}, S. and
             {Angus}, R.},
    title = "{Fast and Scalable Gaussian Process Modeling with Applications to
              Astronomical Time Series}",
  journal = {\aj},
     year = 2017,
    month = dec,
   volume = 154,
    pages = {220},
      doi = {10.3847/1538-3881/aa9332},
   adsurl = {http://adsabs.harvard.edu/abs/2017AJ....154..220F},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@article{exoplanet:foremanmackey18,
   author = {{Foreman-Mackey}, D.},
    title = "{Scalable Backpropagation for Gaussian Processes using Celerite}",
  journal = {Research Notes of the American Astronomical Society},
     year = 2018,
    month = feb,
   volume = 2,
   number = 1,
    pages = {31},
      doi = {10.3847/2515-5172/aaaf6c},
   adsurl = {http://adsabs.harvard.edu/abs/2018RNAAS...2a..31F},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    ),
    "rebound": (
        ("exoplanet:rebound",),
        r"""
@ARTICLE{rebound,
       author = {{Rein}, H. and {Liu}, S. -F.},
        title = "{REBOUND: an open-source multi-purpose N-body code for
                 collisional dynamics}",
      journal = {\aap},
         year = 2012,
        month = jan,
       volume = {537},
          eid = {A128},
        pages = {A128},
          doi = {10.1051/0004-6361/201118085},
archivePrefix = {arXiv},
       eprint = {1110.4876},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2012A&A...537A.128R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    ),
    "reboundias15": (
        ("exoplanet:reboundias15",),
        r"""
@ARTICLE{reboundias15,
       author = {{Rein}, Hanno and {Spiegel}, David S.},
        title = "{IAS15: a fast, adaptive, high-order integrator for
                  gravitational dynamics, accurate to machine precision over a
                  billion orbits}",
      journal = {\mnras},
         year = 2015,
        month = jan,
       volume = {446},
       number = {2},
        pages = {1424-1437},
          doi = {10.1093/mnras/stu2164},
archivePrefix = {arXiv},
       eprint = {1409.4779},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1424R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    ),
    "vaneylen19": (
        ("exoplanet:vaneylen19",),
        r"""
@article{exoplanet:vaneylen19,
author = {{Van Eylen}, Vincent and {Albrecht}, Simon and {Huang}, Xu and
         {MacDonald}, Mariah G. and {Dawson}, Rebekah I. and {Cai}, Maxwell X.
         and {Foreman-Mackey}, Daniel and {Lundkvist}, Mia S. and
         {Silva Aguirre}, Victor and {Snellen}, Ignas and {Winn}, Joshua N.},
        title = "{The Orbital Eccentricity of Small Planet Systems}",
      journal = {\aj},
         year = 2019,
        month = feb,
       volume = 157,
       number = 2,
        pages = 61,
          doi = {10.3847/1538-3881/aaf22f},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019AJ....157...61V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
    ),
    "reboundx": (
        ("exoplanet:reboundx",),
        r"""
@article{tamayo2020reboundx,
   title  = {REBOUNDx: a library for adding conservative and dissipative forces
             to otherwise symplectic N-body integrations},
  author  = {Tamayo, Daniel and Rein, Hanno and Shi, Pengshuai and Hernandez,
             David M},
 journal  = {Monthly Notices of the Royal Astronomical Society},
  volume  = {491},
  number  = {2},
   pages  = {2885--2901},
    year  = {2020},
publisher = {Oxford University Press}
}
""",
    ),
}
