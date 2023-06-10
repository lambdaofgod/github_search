from github_search.pipelines.steps import *
from github_search.pipelines.configs import *
from clearml import Dataset, PipelineController, Task


def add_generation_evaluation_step(pipe):
    pipe.add_function_step(
        name="evaluate_generated_texts",
        function=evaluate_generated_texts_step,
        function_kwargs=dict(
            generated_texts_df="${expand_documents.generated_texts_df}"
        ),
        function_return=["eval_df"],
        cache_executed_step=True,
    )
    return pipe


def add_information_retrieval_evaluation_step(pipe, ir_config):
    pipe.add_function_step(
        name="evaluate_information_retrieval",
        function=evaluate_information_retrieval_step,
        function_kwargs=dict(
            searched_df="${expand_documents.generated_texts_df}", ir_config=ir_config
        ),
        function_return=["ir_results"],
        cache_executed_step=True,
    )
    return pipe


def make_expansion_pipeline(config: PipelineConfig):
    sampling_config = config.sampling_config
    generation_config = config.generation_config
    pipe = PipelineController(
        name="sampled document expansion pipeline",
        project="examples",
        version="0.0.1",
        add_pipeline_tags=False,
    )
    pipe.add_function_step(
        name="sample_data",
        function=sample_data_step,
        function_kwargs=dict(sampling_config=dict(sampling_config)),
        function_return=["prompt_infos"],
        cache_executed_step=False,
    )
    pipe.add_function_step(
        name="expand_documents",
        function=expand_documents_step,
        function_kwargs=dict(
            text_generation_config=dict(generation_config),
            prompt_infos_df="${sample_data.prompt_infos}",
        ),
        function_return=["generated_texts_df"],
        cache_executed_step=True,
    )
    return pipe


def make_pipeline(
    config: PipelineConfig, ir_config: Optional[InformationRetrievalEvaluatorConfig]
):
    pipe = make_expansion_pipeline(config)
    if config.evaluate_generation:
        pipe = add_generation_evaluation_step(pipe)
    if ir_config is not None:
        pipe = add_information_retrieval_evaluation_step(pipe, ir_config)

    return pipe


def run_pipeline(cfg_path="conf/text_generation", ir_config_path="conf/ir_config.yaml"):
    cfg = PipelineConfig.load(cfg_path)
    if ir_config_path is not None:
        with open(ir_config_path) as f:
            ir_config = yaml.safe_load(f)
    else:
        ir_config = None
    pipe = make_pipeline(cfg, ir_config)
    print("running pipeline with config:", cfg)
    # cfg = PipelineConfig.parse_obj(cfg)
    # controller = Main().make_pipeline(cfg.sampling_config, cfg.text_generation_config)
    pipe.set_default_execution_queue("default")
    pipe.start_locally(run_pipeline_steps_locally=True)


if __name__ == "__main__":
    fire.Fire(run_pipeline)
