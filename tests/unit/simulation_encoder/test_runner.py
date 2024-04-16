import unittest
from unittest.mock import patch, mock_open, MagicMock

import json

from simulation_encoder.runner import Runner


class TestRunner(unittest.TestCase):
    @patch("simulation_encoder.logger.ExperimentLogger")
    def test_runner(self, mock_logger):
        pass


if __name__ == "__main__":
    unittest.main()
