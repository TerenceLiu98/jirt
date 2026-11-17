"""Sequential ordered responses with logistic stage success probabilities."""

from .polytomous import polytomous_irt, polytomous_probabilities


def sequential_probabilities(theta, *, a, steps, category_counts, model="sequential"):
    """Return padded category probabilities for sequential or unit-slope Tutz.

    Reaching category k requires passing stages 0..k-1 and failing stage k;
    the last category requires passing every stage. Stage difficulties need
    not be ordered. Counts explicitly include unobserved categories.
    """
    if model not in ("sequential", "tutz"):
        raise ValueError("model must be sequential or tutz")
    return polytomous_probabilities(
        theta, a=a, steps=steps, category_counts=category_counts, model=model
    )


def sequential_irt(
    responses,
    *,
    category_counts,
    model="sequential",
    starts=None,
    quadrature_points=41,
    max_iter=1000,
    tolerance=1e-10,
    jit=True,
):
    """Fit sequential or Tutz responses with fixed N(0,1) ability.

    Responses are integer categories starting at zero, with NaN missing.
    Sequential estimates one positive slope per item shared across stages;
    Tutz fixes every slope to one. Each stage has a free difficulty. Starts
    accepts padded steps and, for sequential only, item slopes a. Returns a
    PolytomousResult with EAP scores and conditional posterior variances.
    """
    if model not in ("sequential", "tutz"):
        raise ValueError("model must be sequential or tutz")
    return polytomous_irt(
        responses,
        category_counts=category_counts,
        model=model,
        starts=starts,
        quadrature_points=quadrature_points,
        max_iter=max_iter,
        tolerance=tolerance,
        jit=jit,
    )
