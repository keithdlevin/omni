# omnibus embedding
Methods as described in <a href='https://arxiv.org/abs/1705.09355'>A central limit theorem for an omnibus embedding of multiple random graphs and implications for multiscale network inference</a> by Levin, Athreya, Tang, Lyzinski, Park and Priebe.

<b>omni.py</b> implements the operations to construct the omnibus matrix and its associated embedding. It also includes an implementation of the "omnibar" embedding considered in the estimation experiments in Section 3.

<b>resample.py</b> implements the bootstrapping methods needed by our experiments. See also Levin and Levina (2025) in Electronic Journal of Statistics.

<b>networks.py</b> contains code for generating from a few simple latent variable network models, most notably the SBM and RDPG.

<b>embeddings.py</b> implements the ASE and the "Abar embedding" method used in some of the experiments in the paper.

<b>procrustes.py</b> implements basic functionality related to Procrustes alignment of point clouds, and is used in the Procrustes-based methods that serve as baseline comparisons in the paper.

The experiments considered in the paper are implemented in the files
<b>lp_estimation_expt.py</b>, <b>twograph_hyptest_drift_expt.py</b>, <b>twograph_hyptest_replace_expt.py</b>
<b>tti2v1_expt.py</b> and <b>varym.py</b>.
