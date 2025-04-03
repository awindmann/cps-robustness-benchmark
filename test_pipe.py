from kfp import dsl
from typing import List, Dict
from kfp.client import Client


@dsl.component()
def create_list() -> List[Dict[str, str]]:
    return [{"a": "b"}, {"b": "c"}, {"c": "d"}, {"e": "f"}]

@dsl.component()
def sleep_comp(element: Dict) -> str:
    import time
    time.sleep(600)
    return str(element)

@dsl.component()
def print_list_entry(element: str) -> str:
    print(element)
    return str(element)


@dsl.pipeline
def sample_pipeline() -> None:
    get_list_task = create_list()
    with dsl.ParallelFor(items=get_list_task.output, parallelism=2) as element:
        sleep_task = sleep_comp(element=element)
        print_task = print_list_entry(element=sleep_task.output)


client = Client()
client.create_run_from_pipeline_func(sample_pipeline)
