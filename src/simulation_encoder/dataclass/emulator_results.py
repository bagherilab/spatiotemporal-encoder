from dataclasses import dataclass, field


@dataclass
class ModelResult:
    model_params: dict[str, float]
    r2_score: float


@dataclass
class LabelResult:
    model_results: dict[str, ModelResult] = field(default_factory=dict)

    def add_model_result(
        self, model_type: str, model_params: dict[str, float], r2_score: float
    ) -> None:
        """Add result for a specific model type and label"""
        self.model_results[model_type] = ModelResult(model_params, r2_score)


@dataclass
class EncoderModelResult:
    label_results: dict[str, LabelResult] = field(default_factory=dict)

    def add_label_result(self, label: str) -> LabelResult:
        """Add a new label result and return it"""
        if label not in self.label_results:
            self.label_results[label] = LabelResult()
        return self.label_results[label]


@dataclass
class DatasetResult:
    encoder_models: dict[str, EncoderModelResult] = field(default_factory=dict)

    def add_encoder_model_result(self, encoder_model: str) -> EncoderModelResult:
        """Add a new encoder model result and return it"""
        if encoder_model not in self.encoder_models:
            self.encoder_models[encoder_model] = EncoderModelResult()
        return self.encoder_models[encoder_model]


@dataclass
class EmulationResults:
    datasets: dict[str, DatasetResult] = field(default_factory=dict)

    def add_dataset_result(self, dataset_name: str) -> DatasetResult:
        """Add a new dataset result and return it"""
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = DatasetResult()
        return self.datasets[dataset_name]

    def get_results(self) -> dict[str, DatasetResult]:
        """Retrieve all emulation results"""
        return self.datasets
