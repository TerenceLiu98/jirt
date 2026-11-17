# Execution and optimizer choices

All estimators require JAX float64. The default SciPy optimizer is retained for
quadrature-based calibration; changing the optimizer is explicit:

```python
fit = jirt.logistic_irt(responses, model="2pl", optimizer="jax",
                        tolerance=1e-6, max_iter=3000)
```

`optimizer="jax"` requires `jit=True`. It preserves the likelihood, priors or
regularization, identification restrictions, and parameter bounds of the SciPy
path. It uses a different optimization algorithm and stopping rule: equal
`tolerance` values do not imply equal numerical accuracy. Compare final objective,
predictions, posterior moments, and identified parameters; check `converged` and
`message` before using a fit. There is no silent optimizer fallback.

## Execution audit

| Estimators | Compiled path | Reference path and remaining host work |
| --- | --- | --- |
| Binary EM/variational, ordinal, dynamic, hierarchical, network, Poisson | One `lax.while_loop` includes updates, finite checks, convergence, and model-specific finalization. Module-level kernels take data explicitly. | `Control(jit=False)` retains Python iteration and NumPy convergence checks. Input validation, initialization, result assembly and orientation anchoring remain on the host. |
| DINA/DINO | One compiled EM loop carries the posterior into the next M-step, including stopping and validity checks. | `jit=False` retains Python EM. Response-pattern aggregation and final result assembly remain on the host. |
| Logistic, multidimensional/bifactor/testlet, polytomous/sequential, nominal/nested, partially compensatory, GGUM, observed groups and mixtures | Optional `optimizer="jax"` compiles the optimizer, objective, gradient, constraint projection and line search. Objective/score kernels are reusable across calls. | Default `optimizer="scipy"` uses SLSQP with host evaluations and callbacks. Input validation, quadrature construction, pattern aggregation, parameter-layout construction, and result assembly remain on the host. Mixture multiple-start orchestration is sequential Python. |
| Amortized 1PL | Optional `optimizer="jax"` compiles JAX BFGS and its line search; objective kernels use explicit data and static encoder layout. | Default `optimizer="scipy"` uses L-BFGS-B. Feature normalization, validation, quadrature and result assembly remain on the host. |

Static structure, shapes and dtypes determine compilation reuse. Changing array
values alone does not require compilation. Changing the number of unique response
patterns, active-tenure entries, categories, free parameters, quadrature points,
maximum iterations or model structure can require compilation. The implementation
does not promise cross-process caching. An eager fit is a numerical reference,
not a performance baseline for the previous release's JIT behavior.

## Stopping and traces

The original iterative estimators retain their absolute/correlation criteria,
constant-vector fallback, minimum iteration counts, error-vector traces, and
binary final maximization using the preceding latent expectations. Correlation
reductions may differ at floating-point roundoff from NumPy near a stopping
threshold. Invalid device updates raise `FloatingPointError` after the compiled
loop returns. Diagnostic EM retains its mean-log-likelihood change criterion and
initial-plus-iteration likelihood trace.

The constrained JAX optimizer uses spectral projected gradient with a ten-value
nonmonotone Armijo line search. Its stopping residual is
`max(abs(project(x - grad) - x))`, with a unit gradient step. Box bounds are exact;
independent difference constraints (including 4PL asymptote gaps) use an analytic
box-intersection projection. Overlapping linear constraints use Dykstra projection
with a 1e-10 absolute feasibility/change threshold and a 1000-sweep cap. A failed
projection or line search returns `converged=False` with a message. Nonfinite
initial objectives/gradients raise `FloatingPointError`. The feasible set includes
its boundary: positive slope lower bounds, zero testlet loadings, asymptote bounds,
ordered threshold gaps, shared coefficients, anchors and structural zeros are not
replaced with stricter open-interval transformations. The trace contains the
initial likelihood and accepted iterates, which need not increase monotonically.

Amortized JAX BFGS stops on the gradient infinity norm and records only initial and
final objective values. SciPy L-BFGS-B also has a relative objective-change stopping
criterion and records callbacks. BFGS uses a dense inverse-Hessian approximation,
so its memory use grows quadratically with the encoder parameter count.

Device execution alone does not guarantee faster convergence. In particular,
weakly identified guessing/upper-asymptote models can exhaust their iteration
budget or reach a different local optimum. The SciPy default remains available
for these cases. Compare optimizers on the same data, starts, quadrature and
constraints, and report failures as well as successful fits.

## CPU measurements

Measured on an Apple M1 Pro, macOS arm64, JAX 0.12.0.1, CPU backend,
float64. JAX exposed only `cpu:0`; accelerator execution was unavailable.
The baseline is commit `0b7683df50dc2c58e8e0f245cdc96e349307f953`.
Each case used a fresh process and four fits, with synchronization before stopping
the timer. Warm values are medians of the final three fits. Baseline and changed
runs were executed sequentially. These are local measurements, not portable
speed guarantees. Cold totals include setup, tracing, compilation and execution;
compile time below is isolated for the main iterative kernel, excluding auxiliary
JAX setup/scoring compilations.

| Model | Subjects × items | Baseline cold (s) | Device cold (s) | Main compile (s) | Baseline warm (ms) | Device warm (ms) | Warm ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| binary | 200 × 6 | 1.176 | 0.606 | 0.271 | 233.16 | 9.36 | 24.9× |
| binary | 2000 × 20 | 1.423 | 0.608 | 0.255 | 272.85 | 32.72 | 8.3× |
| diagnostic | 200 × 6 | 0.559 | 0.325 | 0.132 | 67.87 | 1.83 | 37.1× |
| diagnostic | 2000 × 20 | 0.447 | 0.366 | 0.147 | 81.94 | 12.02 | 6.8× |
| logistic | 200 × 6 | 0.545 | 0.690 | 0.210 | 17.10 | 3.65 | 4.7× |
| logistic | 2000 × 20 | 0.763 | 0.870 | 0.250 | 75.95 | 20.32 | 3.7× |
| amortized | 200 × 6 | 1.019 | 1.315 | 0.361 | 128.97 | 2.44 | 52.9× |
| amortized | 2000 × 20 | 1.021 | 1.348 | 0.398 | 175.83 | 9.50 | 18.5× |

Binary and diagnostic fits used identical starts, tolerances (1e-6), limits and
stopping criteria. Iteration counts matched: binary 134/44 and diagnostic 98/36
for small/large cases; all converged. Diagnostic log likelihoods agreed within
4e-12. Logistic means 2PL; amortized means the affine encoder with L2=0.01.
For these different solvers, JAX used gradient tolerance 1e-6 and SciPy objective
tolerance 1e-11, with the same data, starts, quadrature and objective. Both
converged. The 2PL log-likelihood differences were below 1.3e-8; regularized
amortized objectives differed by less than 3e-13. Iterations were 33/30 versus
47/48 for 2PL, and 6/6 versus 7/6 for amortized. Cold device optimizer fits were
slower than the baseline despite faster warm fits. The amortized warm gain also
includes reuse of the objective kernel that the baseline recreated per fit.

A separate finite-sample 400 × 6 probe with guessing/upper asymptotes did **not**
converge under projected gradient within 5000 iterations (tolerance 1e-6).
Different local solutions had appreciably different predictions. This is why
SciPy remains the default; the successful 2PL benchmark does not establish
performance or interchangeability for 3PL/4PL.

`python scripts/benchmark.py` separately measures all six original model loops
on fixed 30-iteration fixtures, isolating compilation from synchronized kernel
execution and end-to-end fitting. `python scripts/benchmark_device.py` generates
the small/large report above. Select a previous checkout using
`--engine baseline --source /path/to/checkout/src --output outputs/baseline.json`.
Reports are generated under ignored `outputs/` and are not package artifacts.

`python scripts/profile_fitting.py` profiles a warmed binary fit with 5 and 50
iterations. In this run, the new path had 2,205 Python calls and 44 NumPy `asarray`
calls at both lengths, plus two JAX boolean conversions and one integer conversion.
The baseline had 117,393/132,466 Python calls and 292/1,057 NumPy `asarray` calls.
Baseline per-fit compilation contributes to these counts. NumPy counts include
ordinary CPU conversions: they are evidence about Python overhead and potential
synchronization entry points, **not** measurements of GPU transfer volume.

## Validation

The regression suite covers fixed reference updates, eager/compiled agreement,
convergence and iteration limits, nonfinite states, exact box boundaries and
intersecting ordering constraints, and reuse with changed data. Cross-solver
checks cover objectives, predictions and posterior moments for identified
calibration cases; they do not assert that nonconvex fits must find the same
optimum. Full-data checks for all six original estimators preserve reference
parameter tolerances, convergence flags and iteration counts. The examples,
format/lint checks, source/wheel builds and installation of the wheel in an
independent environment are also exercised.
