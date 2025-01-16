_metric_registry = {}


def register(name):
    """Decorator for registering a metric class with a name in the registry.

    Args:
        name (str): Identifier for the metric class.

    Raises:
        AssertionError: If `name` is not a string.

    Returns:
        class: The registered class.

    """

    assert isinstance(name, str), "Name must be a string"

    def decorator(cls):
        _metric_registry[name] = cls
        return cls

    return decorator


def get_metric(name):
    """Retrieve a metric class from the registry based on its name.

    Args:
        name (str): Identifier for the metric class.

    Raises:
        AssertionError: If `name` is not a string.
        ValueError: If a metric with the given `name` doesn't exist.

    Returns:
        class: The metric class corresponding to `name`.

    """

    assert isinstance(name, str), "Name must be a string"

    metric = _metric_registry.get(name)
    if metric is None:
        raise ValueError("No model named '{}'".format(name))

    return metric


def l2_distance(model1, model2):
    w1 = [param.view(-1) for param in model1.parameters()]
    w2 = [param.view(-1) for param in model2.parameters()]
    assert len(w1) == len(
        w2
    ), "The number of parameters in the two models are different."
    squared_sum = 0
    for i in range(len(w1)):
        squared_sum += (w1[i] - w2[i]).pow(2).sum()
    return squared_sum.sqrt()