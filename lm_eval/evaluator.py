import collections
import itertools
import random
import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
import numpy as np


def simple_evaluate(model, model_args, task_names,
                    num_fewshot=0, batch_size=None, device=None,
                    no_cache=False, limit=None, bootstrap_iters=100000):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: str
        Name of model, see lm_eval.models.get_model
    :param model_args: str
        String arguments for each model class, see LM.create_from_arg_string
    :param task_names: list[str]
        List of task names
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    lm = lm_eval.models.get_model(model).create_from_arg_string(model_args, {
        'batch_size': batch_size, 'device': device
    })

    if not no_cache:
        lm = lm_eval.base.CachingLM(
            lm, 'lm_cache/' + model + '_' + model_args.replace('=', '-').replace(',', '_').replace('/', '-') + '.db'
        )

    task_dict = lm_eval.tasks.get_task_dict(task_names)
    results = evaluate(lm, task_dict, False, num_fewshot, limit)

    # add info about the model and few shot config
    results["config"] = {
        "model": model,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters
    }

    return results


def evaluate(lm, task_dict, provide_description, num_fewshot, limit, bootstrap_iters=100000):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system
    assert not provide_description  # not implemented.

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if(task.has_validation_docs() or task.has_test_docs())
    ]

    results = collections.defaultdict(dict)
    lens_results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
        elif task.has_validation_docs():
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            docs[(task_name, doc_id)] = doc

            ctx = task.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
                rnd=rnd
            )

            reqs = task.construct_requests(doc, ctx)
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
        #       only in index. We could implement some kind of caching, but that would be more of a band-aid
        #       solution. we could also implement some kind of auto-grouping here;
        #       they should end up next to each other.

        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])

        for i in range(0, len(reqs)):
            resps[i].select_result(reqs[i].index)

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))

    vals = collections.defaultdict(list)
    lens_vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    layers = 0

    for (task_name, doc_id), requests in process_res_queue.items():
        requests.sort(key=lambda x: x[0])
        responses = [x[1].result for x in requests]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        metrics = task.process_results(doc, responses)
        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)

        if len(requests) > 0:
            ll_layer_count = len(requests[0][1].logit_lens)

            for layer_idx in range(0, ll_layer_count):
                resp = [x[1].logit_lens[layer_idx] for x in requests]
                lens_metrics = task.process_results(doc, resp)
                for metric, value in lens_metrics.items():
                    lens_vals[(task_name, layer_idx, metric)].append(value)

    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        results[task_name][metric] = task.aggregation()[metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this
        stderr = lm_eval.metrics.stderr_for_metric(
            metric=task.aggregation()[metric],
            bootstrap_iters=min(bootstrap_iters, 1000) if metric in ["bleu", "chrf", "ter"] else bootstrap_iters,
        )
        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)

    for (task_name, layer_idx, metric), items in lens_vals.items():
        task = task_dict[task_name]
        lens_results[task_name]["{}.layer{}".format(metric, layer_idx)] = task.aggregation()[metric](items)

    return {
        "results": dict(results),
        "lens-results": dict(lens_results),
        "versions": dict(versions)
    }


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, '%.4f' % v, '±', '%.4f' % se])
            else:
                values.append([k, version, m, '%.4f' % v, '', ''])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()
