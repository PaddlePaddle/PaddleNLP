import logging
# import
from requests.exceptions import ConnectionError

logger = logging.getLogger(__name__)


class StdoutLogger():
    """Minimal logger printing metrics and params to stdout.
    Useful for services like AWS SageMaker, where you parse metrics from the actual logs"""

    disable_logging = False

    def __init__(self, tracking_uri, **kwargs):
        self.tracking_uri = tracking_uri

    def init_experiment(self, experiment_name, run_name=None, nested=True):
        logger.info(
            f"\n **** Starting experiment '{experiment_name}' (Run: {run_name})  ****"
        )

    @classmethod
    def log_metrics(cls, metrics, step):
        logger.info(f"Logged metrics at step {step}: \n {metrics}")

    @classmethod
    def log_params(cls, params):
        logger.info(f"Logged parameters: \n {params}")

    @classmethod
    def log_artifacts(cls, dir_path, artifact_path=None):
        raise NotImplementedError

    @classmethod
    def end_run(cls):
        logger.info(f"**** End of Experiment **** ")
