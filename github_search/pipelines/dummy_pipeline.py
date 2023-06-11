#!/usr/bin/env python3

from clearml import Dataset, PipelineController, Task


def f1(x_in):
    x = x_in + 1
    print("running f1")
    return x


def f2(x_in):
    print("running f2")
    y = x_in * 2
    return y


if __name__ == "__main__":
    pipe = PipelineController(
        name="dummy pipeline",
        project="dummy",
        version="0.0.1",
        add_pipeline_tags=False,
    )

    pipe.add_parameter(name="x_in", default=1)
    pipe.add_function_step(
        name="function_f1",
        function=f1,
        function_kwargs=dict(x_in=1),
        function_return=["x"],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name="function_f2",
        function=f2,
        function_kwargs=dict(x_in="${function_f1.x}"),
        function_return=["y"],
        cache_executed_step=True,
    )
    pipe.set_default_execution_queue("default")
    pipe.start_locally(run_pipeline_steps_locally=True)
