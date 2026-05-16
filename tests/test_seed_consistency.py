"""Reproducibility hygiene: every helper that accepts ``random_state``
should default to the same seed that ``specs._default_hyperparams()``
uses. The canonical value is 42.

Background: the walk-forward + train pipelines pass ``spec.hyperparams``
explicitly, so production runs are unaffected by these defaults. But
ad-hoc scripts and unit tests that call helpers directly without the
spec used to get seed=0, while the same call routed through the spec
used 42. Two callers, two seeds, two different stochastic paths even
though they're "logically the same" run.

This pin lets a future contributor who adds a new helper with
``random_state=0`` see the regression in CI.
"""
from __future__ import annotations

import inspect

from blackheart_train.specs import _default_hyperparams


CANONICAL_SEED = 42


def test_specs_canonical_seed_is_42():
    """The canonical hyperparam factory must declare random_state=42 —
    every helper default in this test pins to this value, so flipping
    it here flips the whole project's reproducibility baseline (which
    is exactly what we want when we DO intend to change it).
    """
    assert _default_hyperparams()["random_state"] == CANONICAL_SEED


def _default_value(callable_, param_name: str):
    sig = inspect.signature(callable_)
    param = sig.parameters[param_name]
    assert param.default is not inspect.Parameter.empty, (
        f"{callable_.__name__}.{param_name} has no default; "
        f"this test can't enforce a value"
    )
    return param.default


def test_adversarial_auc_default_seed():
    from blackheart_train.adversarial import adversarial_auc
    assert _default_value(adversarial_auc, "random_state") == CANONICAL_SEED


def test_bootstrap_default_seed():
    from blackheart_train.bootstrap import bootstrap_macro_auc_ovr
    assert _default_value(bootstrap_macro_auc_ovr, "random_state") == CANONICAL_SEED


def test_feature_selection_default_seed():
    from blackheart_train.feature_selection import select_features
    assert _default_value(select_features, "random_state") == CANONICAL_SEED


def test_meta_label_default_seed():
    from blackheart_train.meta_label import fit_meta_label
    assert _default_value(fit_meta_label, "random_state") == CANONICAL_SEED


def test_no_helper_silently_defaults_to_zero():
    """Catches the original bug shape: a helper with random_state=0 in
    its signature. Iterates the four known helpers; a future helper
    added to the project should also appear here once the author notices
    this test.
    """
    from blackheart_train import adversarial, bootstrap, feature_selection, meta_label
    seen = []
    for mod in (adversarial, bootstrap, feature_selection, meta_label):
        for name, obj in vars(mod).items():
            if not callable(obj) or not hasattr(obj, "__module__"):
                continue
            if obj.__module__ != mod.__name__:
                continue
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            if "random_state" not in sig.parameters:
                continue
            default = sig.parameters["random_state"].default
            if default is inspect.Parameter.empty:
                continue
            seen.append((mod.__name__, name, default))
    # Every helper that has a random_state parameter with a default
    # MUST default to CANONICAL_SEED. Surface any deviation by name so
    # the failure message identifies the offender.
    offenders = [(m, n, d) for (m, n, d) in seen if d != CANONICAL_SEED]
    assert not offenders, (
        f"helpers with random_state != {CANONICAL_SEED}: {offenders}"
    )
    # Sanity: we did inspect at least the four we know about.
    assert len(seen) >= 4, f"expected to inspect >=4 helpers, saw {seen}"
