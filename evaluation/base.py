class EvalMetric:
    """Base class for evaluation metrics in unlearning benchmarks.

    Attributes:
        is_retrain_standard (bool): Whether retraining is needed for this metric.
        results (dict): The final results of the metric evaluation.
        buffer (dict): Temporary storage for intermediate results.

    """

    def get_name(self):
        """Get the name of the metric.

        Returns:
            str: Name of the metric.

        """

        if hasattr(self, "name"):
            return self.name
        return self.__class__.__name__

    def __init__(self, is_retrain_standard=True):
        """Initialize the evaluation metric.

        Args:
            is_retrain_standard (bool, optional): If retraining is needed. Defaults to True.

        """

        self.clear_results()
        self.is_retrain_standard = is_retrain_standard

    def clear_results(self):
        """Clear the results dictionary."""

        self.results = {}
        self.clear_buffer()

    def clear_buffer(self):
        """Clear the buffer dictionary."""
        self.buffer = {}

    def pre_process(self, *args, **kwargs):
        """Pre-processing steps before evaluation."""
        pass

    def evaluate(self, model, loaders, iteration):
        """Evaluate the model for the metric.

        Args:
            model (object): Model to evaluate.
            loaders (object): Data loaders.
            iteration (int): Current iteration.

        Raises:
            NotImplementedError: Method must be implemented in subclass.

        """

        raise NotImplementedError("eval() must be implemented")

    def finalize(self, buffer, name, retrain_buffer=None):
        """Finalize the metric calculations.

        Args:
            buffer (object): Temporary buffer storing intermediate results.
            name (str): Identifier for the model or experiment.
            retrain_buffer (object, optional): Buffer for retraining metrics.

        Returns:
            object: The finalized results.

        """

        return buffer

    def post_process(self, name, iteration, model, loaders):
        """Post-process after evaluation to update the buffer.

        Args:
            name (str): Identifier for the model or experiment.
            iteration (int): Current iteration.
            model (object): Model to evaluate.
            loaders (object): Data loaders.

        """

        self.buffer[(name, iteration)] = self.evaluate(model, loaders, iteration)

    def calculate(self):
        """Calculate the final results based on the buffer."""

        for name, iteration in self.buffer:
            if self.is_retrain_standard:
                self.results[(name, iteration)] = self.finalize(
                    self.buffer[(name, iteration)],
                    name + "_" + str(iteration),
                    retrain_buffer=self.buffer[("Retrain", iteration)],
                )
            else:
                self.results[(name, iteration)] = self.finalize(
                    self.buffer[(name, iteration)],
                    name + "_" + str(iteration),
                )

    def get_results(self):
        """Get the final results of the metric evaluation.

        Returns:
            dict: Final results of the metric evaluation.

        """

        return self.results