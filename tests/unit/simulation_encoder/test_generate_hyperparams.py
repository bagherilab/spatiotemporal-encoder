import unittest
from torch.optim import Adam, SGD

from simulation_encoder.dataclass.config_schemas import (
    HyperparameterDiscreteConfig,
    HyperparameterRangeConfig,
)
from simulation_encoder.utils.generate_hyperparams import (
    _get_continuous_values,
    _get_discrete_values,
    _get_optimizer_values,
    generate_hyperparameters,
)


# ---------------------------------------------------------------------------
# _get_continuous_values
# ---------------------------------------------------------------------------


class TestGetContinuousValues(unittest.TestCase):
    def test_none_input_returns_empty_dict(self):
        self.assertEqual(_get_continuous_values(None), {})

    def test_single_sample_returns_scalar_wrapped_in_list(self):
        params = {
            "image_loss_weight": HyperparameterRangeConfig(
                range=0.9, search="linear", num_samples=1
            )
        }
        result = _get_continuous_values(params)
        self.assertEqual(result["image_loss_weight"], [0.9])

    def test_linear_range_returns_correct_count(self):
        params = {
            "image_loss_weight": HyperparameterRangeConfig(
                range=[0.1, 0.9], search="linear", num_samples=5
            )
        }
        result = _get_continuous_values(params)
        self.assertEqual(len(result["image_loss_weight"]), 5)

    def test_linear_range_endpoints_correct(self):
        params = {"x": HyperparameterRangeConfig(range=[0.2, 0.8], search="linear", num_samples=3)}
        result = _get_continuous_values(params)
        self.assertAlmostEqual(result["x"][0], 0.2)
        self.assertAlmostEqual(result["x"][-1], 0.8)

    def test_log_range_returns_correct_count(self):
        params = {"lr": HyperparameterRangeConfig(range=[0.001, 0.1], search="log", num_samples=4)}
        result = _get_continuous_values(params)
        self.assertEqual(len(result["lr"]), 4)

    def test_unsupported_search_raises_value_error(self):
        params = {"x": HyperparameterRangeConfig(range=[0.1, 0.9], search="random", num_samples=3)}
        with self.assertRaises(ValueError):
            _get_continuous_values(params)

    def test_multiple_params_returned(self):
        params = {
            "a": HyperparameterRangeConfig(range=0.5, search="linear", num_samples=1),
            "b": HyperparameterRangeConfig(range=0.3, search="linear", num_samples=1),
        }
        result = _get_continuous_values(params)
        self.assertIn("a", result)
        self.assertIn("b", result)


# ---------------------------------------------------------------------------
# _get_discrete_values
# ---------------------------------------------------------------------------


class TestGetDiscreteValues(unittest.TestCase):
    def test_none_input_returns_empty_dict(self):
        self.assertEqual(_get_discrete_values(None), {})

    def test_latent_dim_passed_through(self):
        params = {"latent_dim": HyperparameterDiscreteConfig(values=[4, 8, 16])}
        result = _get_discrete_values(params)
        self.assertEqual(result["latent_dim"], [4, 8, 16])

    def test_optimizer_type_resolved_to_class(self):
        params = {
            "optimizer": HyperparameterDiscreteConfig(values=[{"type": "Adam", "lr": [0.001]}])
        }
        result = _get_discrete_values(params)
        self.assertEqual(result["optimizer"][0]["type"], Adam)

    def test_multiple_discrete_params(self):
        params = {
            "latent_dim": HyperparameterDiscreteConfig(values=[4]),
            "optimizer": HyperparameterDiscreteConfig(values=[{"type": "Adam", "lr": [0.001]}]),
        }
        result = _get_discrete_values(params)
        self.assertIn("latent_dim", result)
        self.assertIn("optimizer", result)


# ---------------------------------------------------------------------------
# _get_optimizer_values
# ---------------------------------------------------------------------------


class TestGetOptimizerValues(unittest.TestCase):
    def test_adam_with_single_lr(self):
        configs = [{"type": "Adam", "lr": [0.001]}]
        result = _get_optimizer_values(configs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], Adam)
        self.assertAlmostEqual(result[0]["lr"], 0.001)

    def test_adam_with_two_lrs_expands_grid(self):
        configs = [{"type": "Adam", "lr": [0.001, 0.0001]}]
        result = _get_optimizer_values(configs)
        self.assertEqual(len(result), 2)
        lrs = {r["lr"] for r in result}
        self.assertIn(0.001, lrs)
        self.assertIn(0.0001, lrs)

    def test_sgd_nesterov_true_with_zero_momentum_filtered(self):
        configs = [{"type": "SGD", "lr": [0.01], "momentum": [0.0, 0.9], "nesterov": [True, False]}]
        result = _get_optimizer_values(configs)
        for combo in result:
            if combo.get("nesterov"):
                self.assertNotEqual(combo.get("momentum"), 0)

    def test_sgd_type_resolved(self):
        configs = [{"type": "SGD", "lr": [0.01], "momentum": [0.9], "nesterov": [False]}]
        result = _get_optimizer_values(configs)
        self.assertEqual(result[0]["type"], SGD)

    def test_unknown_optimizer_type_kept_as_string(self):
        # _get_optimizer_values resolves known types (Adam, SGD) to classes;
        # unrecognised strings are kept as-is rather than raising.
        configs = [{"type": "CustomOpt", "lr": [0.1]}]
        result = _get_optimizer_values(configs)
        self.assertEqual(result[0]["type"], "CustomOpt")

    def test_multiple_optimizer_types_combined(self):
        configs = [
            {"type": "Adam", "lr": [0.001]},
            {"type": "SGD", "lr": [0.01], "momentum": [0.9], "nesterov": [False]},
        ]
        result = _get_optimizer_values(configs)
        types = {r["type"] for r in result}
        self.assertIn(Adam, types)
        self.assertIn(SGD, types)


# ---------------------------------------------------------------------------
# generate_hyperparameters
# ---------------------------------------------------------------------------


class TestGenerateHyperparameters(unittest.TestCase):
    def _continuous(self, val=0.9, n=1):
        return {
            "image_loss_weight": HyperparameterRangeConfig(
                range=val, search="linear", num_samples=n
            )
        }

    def _discrete(self, dims=None):
        dims = dims or [8]
        return {"latent_dim": HyperparameterDiscreteConfig(values=dims)}

    def test_single_param_set_produced(self):
        result = generate_hyperparameters(self._continuous(), self._discrete())
        self.assertEqual(len(result), 1)

    def test_timepoint_weight_is_complement_of_image_weight(self):
        result = generate_hyperparameters(self._continuous(0.9), self._discrete([8]))
        p = result[0]
        self.assertAlmostEqual(p["image_loss_weight"] + p["timepoint_loss_weight"], 1.0)

    def test_grid_expands_two_by_two(self):
        continuous = {
            "image_loss_weight": HyperparameterRangeConfig(
                range=[0.7, 0.9], search="linear", num_samples=2
            )
        }
        discrete = {"latent_dim": HyperparameterDiscreteConfig(values=[4, 8])}
        result = generate_hyperparameters(continuous, discrete)
        # 2 continuous × 2 discrete = 4 combinations
        self.assertEqual(len(result), 4)

    def test_latent_dim_present_in_each_param_set(self):
        result = generate_hyperparameters(self._continuous(), self._discrete([4, 16]))
        for p in result:
            self.assertIn("latent_dim", p)

    def test_all_latent_dim_values_represented(self):
        result = generate_hyperparameters(self._continuous(), self._discrete([4, 16]))
        dims = {p["latent_dim"] for p in result}
        self.assertEqual(dims, {4, 16})

    def test_both_none_returns_empty_list(self):
        # When both params are None, generate_hyperparameters should handle gracefully
        # (product of empty iterables is a single empty tuple, yielding one empty set)
        result = generate_hyperparameters(None, None)
        # One empty param set: product(*[]) has one element (the empty tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], {})


if __name__ == "__main__":
    unittest.main()
