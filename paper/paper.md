---
title: 'JIRT: Item Response Theory Modeling in Python and JAX'
tags:
  - Python
  - JAX
  - item response theory
  - psychometrics
  - political methodology
authors:
  - name: Junjie Liu
    orcid: 0000-0001-7052-9856
    affiliation: 1
affiliations:
  - name: Department of Political Science, Trinity College Dublin, Dublin, Ireland
    index: 1
date: 27 November 2026
bibliography: paper.bib
---

<!-- Before submission, confirm the date, GitHub release tag (v0.1.0), funding,
conflicts of interest, and AI usage disclosure after final human review. -->

# Summary

Raw sum scores conflate respondent traits with item difficulty: an identical
score reflects very different levels of proficiency or preference on an easy
instrument versus a demanding one. Item response theory (IRT) resolves this
ambiguity by jointly estimating item characteristics and latent respondent
traits from observed response patterns—whether binary correctness, ordered
survey ratings, unordered choices, longitudinal roll-call votes, network ties,
or term counts.

JIRT is an open-source Python library for IRT modeling built on JAX
[@bradbury2018jax]. It unifies two distinct modeling traditions within a single
differentiable array framework: iterative expectation-maximization (EM) and
variational ideal-point estimators widely used in political methodology, and
marginal maximum likelihood (MML) psychometric models used in educational and
psychological measurement. Beyond standard calibration, JIRT supports cognitive
diagnosis, multiple-group equating, latent-class mixtures, and amortized
content-based calibration that predicts the difficulty of unseen items directly
from item features.

# Statement of need

Applied researchers in psychometrics, education, and political methodology
routinely encounter response structures that violate the assumptions of basic
dichotomous IRT. Dichotomizing multiple-choice items discards diagnostic
information contained in distractor selections; treating Likert ratings as
continuous imposes arbitrary spacing between categories; and fitting a static
ideal point to multi-term legislative or judicial records obscures ideological
drift across a legislator's tenure. Similarly, comparing respondents across
cohorts requires shared anchor items to place group distributions on a common
scale, while formative assessment often seeks discrete skill-mastery profiles
rather than a single continuous score. JIRT provides dedicated estimators for
these settings inside Python, allowing data preparation, model fitting,
uncertainty quantification, and downstream analysis to remain in one coherent
workflow.

A second, increasingly common challenge arises when assessments evolve or when
new items must be evaluated before human response data accumulate. Classical
item-specific parameters cannot be estimated for items with zero responses.
When item-level covariates or text representations are available, an amortized
encoder can map item features to difficulty parameters and generalize to
previously unseen items. JIRT implements affine and nonlinear feature encoders
for this setting, bridging classical psychometrics with modern content-based
evaluation workflows such as benchmark and model evaluation
[@truong2025amortized].

# State of the field

Existing IRT software is largely divided by discipline and runtime. In R, `mirt`
is the standard library for unidimensional and multidimensional item factor
analysis [@chalmers2012mirt], whereas `emIRT` implements fast EM and variational
algorithms for binary, ordinal, dynamic, hierarchical, network, and text-scaling
ideal-point models [@imai2016ideal; @imai2025emirt]. In Python, `py-irt`
provides approximate Bayesian IRT via Pyro and PyTorch [@lalor2023pyirt],
`Girth` implements MML for common unidimensional psychometric models [@girth],
and the Python package `mirt` (distinct from the R package) offers a broad suite
of psychometric models backed by Rust [@lyons2026mirt].

JIRT occupies a distinct niche by bringing the `emIRT` political-measurement
family and the quadrature-based psychometric MML family together in pure Python
and JAX, alongside amortized content-based calibration. Extending R `mirt` or
`emIRT` would retain an R runtime dependency; extending `py-irt` would require
grafting deterministic quadrature and constrained MML onto a variational
Bayesian codebase; and while the Rust-backed Python `mirt` addresses
psychometric calibration, it does not cover the dynamic, hierarchical, network,
and Poisson ideal-point models or differentiable feature-encoded calibration.
Implementing these model families over a shared JAX backend allows them to share
automatic differentiation, compiled execution loops, and reproducible
random-number handling.

# Software design

JIRT preserves the response coding, identification constraints, and convergence
diagnostics appropriate to each model family while exposing a consistent Python
interface (Table 1).

| Model family | Implemented models | Estimation & execution |
| --- | --- | --- |
| Ideal-point & political measurement | Binary (EM/variational), ordinal, dynamic, hierarchical, network, Poisson | Compiled `lax.while_loop` updates; optional eager Python path and bootstrap SEs |
| Psychometric MML | Rasch/1PL–4PL, multidimensional, bifactor, testlet, GRM/PCM/GPCM/RSM, nominal/nested, sequential, noncompensatory, GGUM, multiple-group, mixture | Gauss–Hermite quadrature MML [@bock1981marginal] with JAX gradients; SciPy SLSQP default, optional compiled JAX projected-gradient solver |
| Cognitive diagnosis | DINA, DINO (Q-matrix mastery) | Compiled JAX EM loop returning posterior profile and attribute-mastery probabilities |
| Content-based IRT | Amortized 1PL (affine or tanh encoder) | Regularized MML via SciPy L-BFGS-B or compiled JAX BFGS; predicts unseen items from features |

For quadrature-based calibration, JIRT integrates over standardized latent
distributions using Gauss–Hermite quadrature, computes expected a posteriori
(EAP) respondent scores alongside conditional posterior variances, and surfaces
convergence flags and optimizer diagnostics. For dynamic ideal-point models,
subject-specific tenures delimit active estimation windows. All stochastic
routines—including initialization and bootstrap resampling—accept explicit JAX
PRNG keys for deterministic reproducibility.

Fitting requires 64-bit floating-point precision (`jax_enable_x64=True`) to
maintain numerical stability in truncated-normal tails, quadrature summations,
and tight optimization tolerances. Because JAX compiles computational graphs on
first invocation, warm repeated fits (such as bootstrap replicates or simulation
studies) benefit substantially from compilation, whereas changes in array shape
trigger recompilation. For quadrature-based models, SciPy remains the default
optimizer because weakly identified guessing and upper-asymptote parameters in
3PL and 4PL models can converge slowly or reach different local optima under
first-order projected-gradient updates; execution modes, stopping criteria, and
CPU benchmarks are documented in the
[performance guide](https://github.com/TerenceLiu98/jirt/blob/v0.1.0/PERFORMANCE.md).

# Research impact statement

JIRT supports reproducible empirical measurement and cross-software validation
across political science and educational assessment. Its six ideal-point
estimators have been verified against R `emIRT` on canonical political datasets,
including U.S. Senate roll-call votes, longitudinal judicial voting, large-scale
ordinal survey responses, Twitter follower networks, and party manifesto word
counts, matching parameter estimates, posterior variances, and iteration behavior.
For logistic psychometric calibration, independent fits on the LSAT7 benchmark
bundled with R `mirt` agree within $6.42 \times 10^{-5}$ in item parameters
across Rasch, 1PL, and 2PL specifications under matched quadrature and
identification constraints. In synthetic 3PL and 4PL comparisons, probability
and scoring kernels agree to machine precision at shared parameters, while
independent optimization exhibits expected sensitivity along flat likelihood
surfaces; full cross-validation records and fixture provenance are available in
the [validation report](https://github.com/TerenceLiu98/jirt/blob/v0.1.0/VALIDATION.md)
and [fixture documentation](https://github.com/TerenceLiu98/jirt/blob/v0.1.0/tests/fixtures/PROVENANCE.md).

To lower barriers to reproducible research, the test suite validates against
committed reference fixtures without requiring an R installation. In addition,
JIRT ships a self-contained
[whole-item holdout protocol](https://github.com/TerenceLiu98/jirt/blob/v0.1.0/examples/amortized.py)
for content-based calibration, enabling researchers to benchmark feature-driven
difficulty prediction against non-amortized baselines on their own item banks.

# AI usage disclosure

OpenAI GPT-5.6-Sol assisted with organizing and refactoring the codebase. GPT-5.6-Sol helped collecting references and validate the codebase performance in multiple platform. Gemini Hydrogen helped polishing manuscript's grammar and phase usage.  the The author is responsible for reviewing the polished text, verifying the evidence, and approving the final submission.

# Acknowledgements

JIRT's cross-software validation suite builds on the reference implementations
and benchmark datasets made publicly available by the `emIRT` and R `mirt`
projects.

# References
