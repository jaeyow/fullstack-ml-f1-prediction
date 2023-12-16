# pylint: disable=C0415,C0103,W0201,W0702,W0718
import os
from metaflow import FlowSpec, step
# from comet_ml.integration.metaflow import comet_flow

try:
    from dotenv import load_dotenv

    load_dotenv(verbose=True, dotenv_path=".env")
except Exception:
    print("No dotenv package")


# @comet_flow(project_name="<please-replace-me>")
class F1PredictionFeaturePipeline(FlowSpec):

    @step
    def start(self):
        """
        Initialization, place everything init related here, check that everything is
        in order like environment variables, connection strings, etc, and if there are
        any issues, quit now.
        """
        # assert os.environ["AWS_PROFILE_NAME"]
        # assert os.environ["AWS_DEFAULT_REGION"]
        # assert os.environ["BUCKET_NAME"]
        # assert os.environ["COMET_API_KEY"]

        # self.AWS_PROFILE_NAME = os.environ["AWS_PROFILE_NAME"]
        # self.AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
        # self.BUCKET_NAME = os.environ["BUCKET_NAME"]
        print("F1PredictionFeaturePipeline...")

        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """

if __name__ == "__main__":
    F1PredictionFeaturePipeline()