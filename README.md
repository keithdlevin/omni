# omnibus embedding
Methods as described in <a href='https://arxiv.org/abs/1705.09355'>A central limit theorem for an omnibus embedding of multiple random graphs and implications for multiscale network inference</a> by Levin, Athreya, Tang, Lyzinski, Park and Priebe.

omni.py implements the operations to construct the omnibus matrix and its associated embedding. It also includes an implementation of the "omnibar" embedding considered in the estimation experiments in Section 3.

resample.py implements the bootstrapping methods needed by our experiments. See also Levin and Levina (2025) in Electronic Journal of Statistics.

networks.py contains code for generating from a few simple latent variable network models, most notably the SBM and RDPG.

embeddings.py implements the ASE and the "Abar embedding" method used in some of the experiments in the paper.

procrustes.py implements basic functionality related to Procrustes alignment of point clouds, and is used in the Procrustes-based methods that serve as baseline comparisons in the paper.

The experiments considered in the paper are implemented in the files
lp_estimation_expt.py, twograph_hyptest_drift_expt.py, twograph_hyptest_replace_expt.py
tti2v1_expt.py and varym.py
