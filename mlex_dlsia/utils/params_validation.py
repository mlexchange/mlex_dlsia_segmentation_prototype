import logging

from mlex_dlsia.parameters import IOParameters, TrainingParameters

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def validate_parameters(parameters):
    """
    This function extracts parameters from the whole parameter dict
    and performs pydantic validation for both io-related and model-related parameters.
    Input:
        parameters: dict, parameters from the yaml file as a whole.
    Output:
        io_parameters: class, all io parameters in pydantic class
        network: str, name of the selected algorithm
        model_parameters: class, all model specific parameters in pydantic class
        run_parameters: class, all run specific parameters in pydantic class
    """
    # Validate and load I/O related parameters
    io_parameters = parameters["io_parameters"]
    io_parameters = IOParameters(**io_parameters)
    # Check whether mask_uri has been provided as this is a requirement for training.
    assert io_parameters.mask_tiled_uri, "Mask URI not provided for training."

    # Detect which model we have, then load corresponding parameters
    model_parameters = parameters["model_parameters"]
    model_parameters = TrainingParameters(**model_parameters)
    logging.info("Parameters loaded successfully.")
    return io_parameters, model_parameters
