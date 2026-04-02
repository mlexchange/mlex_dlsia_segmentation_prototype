import mlflow
import numpy as np
import torch
from dlsia.core.networks.baggins import model_baggin

from mlex_dlsia.dataset import initialize_tiled_datasets
from mlex_dlsia.inference import _segment_single_frame, _segment_single_frame_ensemble
from mlex_dlsia.parameters import (
    IOParameters,
    MSDNetParameters,
    SMSNetEnsembleParameters,
    TUNet3PlusParameters,
    TUNetParameters,
)
from mlex_dlsia.utils.dataloaders import construct_inference_dataloaders

_NETWORK_PARAMS_MAP = {
    "DLSIA MSDNet": MSDNetParameters,
    "DLSIA TUNet": TUNetParameters,
    "DLSIA TUNet3+": TUNet3PlusParameters,
    "DLSIA SMSNetEnsemble": SMSNetEnsembleParameters,
}


class SegmentationWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Model params
        cfg = context.model_config
        network_name = cfg.get("network")
        params_cls = _NETWORK_PARAMS_MAP.get(network_name)
        if params_cls is None:
            raise ValueError(f"Unknown network type: {network_name!r}")
        self.model_parameters = params_cls(**cfg)

        # Pick ensemble vs single-net inference function
        self.network = network_name
        is_ensemble = self.network == "DLSIA SMSNetEnsemble"
        self.final_layer = None if is_ensemble else torch.nn.Softmax(dim=1)
        self._segment_fn = (
            _segment_single_frame_ensemble if is_ensemble else _segment_single_frame
        )

        # Load model(s) from artifacts and move to device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        n_nets = cfg.get("n_nets", 1)
        nets = [
            torch.load(context.artifacts[f"net_{i+1}"], map_location=self.device)
            for i in range(n_nets)
        ]

        if is_ensemble:
            self.net = model_baggin(models=nets, model_type="classification")
        else:
            self.net = nets[0]

    def predict(self, context, model_input):
        # MLflow pyfunc converts dataframe_records payloads to a pandas DataFrame.
        if hasattr(model_input, "iloc"):
            row = model_input.iloc[0]
            data_tiled_uri = row.get("data_tiled_uri")
            data_tiled_api_key = row.get("data_tiled_api_key")
        else:
            data_tiled_uri = model_input.get("data_tiled_uri")
            data_tiled_api_key = model_input.get("data_tiled_api_key")

        # Define IOParameters for dataset initialization
        io_parameters = IOParameters(
            data_tiled_uri=data_tiled_uri,
            data_tiled_api_key=data_tiled_api_key,
            uid_save=None,
            job_name=None,
            mlflow_model=None,
        )

        # Initialize dataset for the input frames
        dataset = initialize_tiled_datasets(
            io_parameters, self.model_parameters, is_training=False
        )

        assert (
            dataset.data_client.ndim >= 3
        ), f"Expected data_client to be at least 3D, got shape {dataset.data_client.shape}"
        results = np.empty(
            (len(dataset), *dataset.data_client.shape[1:3]), dtype=np.int8
        )

        for idx in range(len(dataset)):
            inference_loader = construct_inference_dataloaders(
                dataset[idx], self.model_parameters
            )

            prediction = self._segment_fn(
                network=self.net,
                dataloader=inference_loader,
                final_layer=self.final_layer,
                device=self.device,
            )
            stitched_prediction, _ = dataset.qlty_object.stitch(prediction)
            results[idx] = (
                torch.argmax(stitched_prediction, dim=1).numpy().astype(np.int8)
            )

        return results
