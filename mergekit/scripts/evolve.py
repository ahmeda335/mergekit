# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.


import random
import numpy
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
#from scoop import futures
import hashlib
import os

import logging
import os
from typing import List, Optional, Tuple, Any

import click
import cma
import numpy as np
import pandas
import ray
import torch
import tqdm
import transformers
import yaml

try:
    import wandb
except ImportError:
    wandb = None


from mergekit.common import ModelReference
from mergekit.evo.config import (
    EvolMergeConfiguration,
    ModelGenomeDefinition,
    check_for_naughty_config,
)
from mergekit.evo.genome import ModelGenome
from mergekit.evo.strategy import (
    ActorPoolEvaluationStrategy,
    BufferedRayEvaluationStrategy,
    SerialEvaluationStrategy,
)
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


@click.command("mergekit-evolve")
@click.argument("genome-config-path", type=str)
@click.option("--max-fevals", type=int, default=100)
@click.option("--vllm/--no-vllm", is_flag=True, default=False, help="Use vLLM")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["pool", "buffered", "serial"]),
    default="pool",
    help="Evaluation scheduling strategy",
)
@click.option(
    "--in-memory/--no-in-memory",
    is_flag=True,
    default=False,
    help="Use in-memory merge & evaluation",
)
@click.option(
    "--storage-path",
    type=str,
    help="Path to storage accessible to all nodes for model storage",
    required=True,
)
@click.option("--num-gpus", type=int, help="Number of GPUs to use across all nodes")
@click.option("--merge-cuda/--no-merge-cuda", is_flag=True, default=True)
@click.option("--trust-remote-code/--no-trust-remote-code", is_flag=True, default=False)
@click.option("--allow-crimes/--no-allow-crimes", is_flag=True, default=False)
@click.option("--random-seed", type=int, default=0)
@click.option("--batch-size", type=int, default=None, help="Batch size for evaluation")
@click.option("--sigma0", type=float, default=1 / 6, help="Initial sigma for CMA-ES")
@click.option("use_wandb", "--wandb/--no-wandb", is_flag=True, default=False)
@click.option("--wandb-project", type=str, help="Wandb project name")
@click.option("--wandb-entity", type=str, help="Wandb entity name")
@click.option(
    "--task-search-path",
    type=str,
    multiple=True,
    help="Path to search for lmeval tasks",
)
@click.option(
    "--i-understand-the-depths-of-the-evils-i-am-unleashing",
    "allow_benchmark_tasks",
    is_flag=True,
    default=False,
    help="Allow benchmark tasks as objectives",
)
@click.option(
    "--save-final-model/--no-save-final-model",
    is_flag=True,
    default=True,
    help="Save the final merged model",
)
@click.option(
    "--reshard/--no-reshard",
    is_flag=True,
    default=True,
    help="Convert models to single-shard safetensors for faster merge",
)
def main(
    genome_config_path: str,
    max_fevals: int,
    vllm: bool,
    strategy: str,  # 'ActorPoolEvaluationStrategy', or 'BufferedRayEvaluationStrategy', or 'SerialEvaluationStrategy'.
    in_memory: bool,
    storage_path: Optional[str],  # Path to storage accessible to all nodes for model storage.
    num_gpus: Optional[int],  # Number of GPUs to use across all nodes.
    merge_cuda: bool,  # Performing matrix arithmetic on GPU.
    trust_remote_code: bool,
    allow_crimes: bool,
    random_seed: int,
    batch_size: Optional[int],
    sigma0: float,
    use_wandb: bool,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    task_search_path: List[str],
    allow_benchmark_tasks: bool,
    save_final_model: bool,
    reshard: bool,  # Convertig the data to be in a single shard so that accessing it will be faster.
):
    config = EvolMergeConfiguration.model_validate(
        yaml.safe_load(open(genome_config_path, "r", encoding="utf-8"))
    )

    check_for_naughty_config(config, allow=allow_benchmark_tasks)

    if use_wandb:
        if not wandb:
            raise RuntimeError("wandb is not installed")
        run = wandb.init(
            project=wandb_project or "mergekit-evolve",
            entity=wandb_entity,
            config=config.model_dump(mode="json"),
        )
    else:
        run = None

    merge_options = MergeOptions(
        transformers_cache=os.path.join(storage_path, "transformers_cache"),
        lora_merge_cache=os.path.join(storage_path, "lora_merge_cache"),
        cuda=merge_cuda,  # Performing matrix arithmetic on GPU.
        low_cpu_memory=merge_cuda and not in_memory,
        out_shard_size=1_000_000_000_000,  # one trillion bytes!
        trust_remote_code=trust_remote_code,
        allow_crimes=allow_crimes,  # Allow mixing architecture.
        random_seed=random_seed,
        quiet=True,
        read_to_gpu=merge_cuda and not in_memory,
        copy_tokenizer=True,
        safe_serialization=True,
    )

    # convert models to single-shard safetensors
    if reshard:  # If the models wants to be resharded 'be putten in a single block'.
        resharded_models = []
        resharded_base = None
        for model in tqdm.tqdm(config.genome.models, desc="Resharding models"):  # model here is the models user enters to be merged.
            resharded_models.append(
                _reshard_model(
                    model,
                    storage_path,
                    merge_options.lora_merge_cache,
                    trust_remote_code,
                )
            )
        if config.genome.base_model is not None:
            resharded_base = _reshard_model(
                config.genome.base_model,
                storage_path,
                merge_options.lora_merge_cache,
                trust_remote_code,
            )
    else:  # If the models do not want to be resharded.
        resharded_models = config.genome.models  # The resharded submodels.
        resharded_base = config.genome.base_model  # The resharded base model.

    genome = ModelGenome(
        ModelGenomeDefinition.model_validate(
            {
                **config.genome.model_dump(
                    exclude=[
                        "models",
                        "base_model",
                    ]
                ),
                "models": resharded_models,
                "base_model": resharded_base,
            }
        ),
        trust_remote_code=trust_remote_code,
    )

    if strategy == "pool":
        strat_cls = ActorPoolEvaluationStrategy
    elif strategy == "buffered":
        strat_cls = BufferedRayEvaluationStrategy
    elif strategy == "serial":
        strat_cls = SerialEvaluationStrategy
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    strat = strat_cls(
        config,
        genome,
        merge_options,
        num_gpus=num_gpus,
        vllm=vllm,
        in_memory=in_memory,
        model_storage_path=os.path.join(storage_path, "merged"),
        batch_size=batch_size,
        task_search_path=task_search_path,
    )
    # ----------------------------------------------------------------------- #
    # ---------------------- Creating the population ------------------------ #
    # ----------------------------------------------------------------------- #

    population_size = 100

    # population = [genome.initial_genotype(random=config.random_init).view(-1).numpy() for _ in range(population_size)]   # Creating population of genotypes If I want to use GA.
    def individ():
        print("individual")
        return genome.initial_genotype(random=config.random_init).view(-1)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # This creates a class named 'FitnessMin', with negative weight so 'minimizing' such that I maximize it at evaluating func.
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)  # This creates a class named 'Individual' inherited from numpy.ndarray.


    toolbox = base.Toolbox()
    toolbox.register("attr_float", individ)  # This creates only one individual with the shape of the 'genome.initial_genotype'.
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, 1)  # This will create the indiviual and transfer it to 'np.ndarray' shape.
    # I can access any individual using 'toolbox.individual()'.

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.population(n=population_size)

    ## Note ##
    # I can use another way for initializing population by reading it from a json file. LooK to this link at the end of the webpage.
    # https://deap.readthedocs.io/en/master/tutorials/basic/part1.html#creating-types

    # --------------------------------------------------------------------------------------------- #


    def parallel_evaluate(x: List[np.ndarray]) -> list[tuple[Any]]:
        print(f"Received {len(x)} genotypes")
        res = strat.evaluate_genotypes(x)  # We note that function 'evaluate_genotypes' takes a list of genotypes, 
                                          # then it often uses ways like multiprocessing or multitasking to keep the logic
                                          # of parallel evaluation.
                                          # It returns a list of results, where each result is a dictionary with 'score' and 'results' keys.
                                          # 'score' is the fitness of the genotype, and 'results' is a dictionary of task results.

        if use_wandb:
            res = list(res)
            score_mean = np.mean([r["score"] for r in res])
            score_std = np.std([r["score"] for r in res])
            run.log(
                {
                    "population/score_mean": score_mean,
                    "population/score_std": score_std,
                },
                commit=False,
            )
            for task in res[0]["results"]:
                for metric in res[0]["results"][task]:
                    values = [r["results"][task][metric] for r in res]
                    values = [v for v in values if v is not None]
                    if not values or all(isinstance(v, str) for v in values):
                        continue

                    mean = np.mean(values)
                    max_val = max(values)
                    min_val = min(values)

                    metric_pretty = metric.replace(",none", "")
                    if metric_pretty.endswith("_stderr"):
                        # don't log stats for stderr that's just silly
                        continue

                    run.log(
                        {
                            f"population/{task}_{metric_pretty}_mean": mean,
                            f"population/{task}_{metric_pretty}_max": max_val,
                            f"population/{task}_{metric_pretty}_min": min_val,
                        },
                        commit=False,
                    )

        return [(-x["score"], ) for x in res]  # maximize
    
    # This is for setting the fitness of the population.
    for i in range(len(parallel_evaluate(population))):
        population[i].fitness.values = parallel_evaluate(population)[i]

    toolbox.register("evaluate",   parallel_evaluate, x=population)
    toolbox.register("mate",       tools.cxUniform, indpb=0.5)  # Crossover.
    toolbox.register("select",     tools.selTournament, tournsize=10)
    toolbox.register("mutate",     tools.mutShuffleIndexes, indpb=0.5)


    ngen = 100  # Gerations
    npop = 100  # Population

    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    # Statistics
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)


    # Evolution "using algorithms".
    pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=npop, lambda_=npop,
                                              cxpb=0.7,   mutpb=0.3, ngen=ngen,
                                              stats=stats, halloffame=hof)
    
    # print(pop)
    # print(logbook)
    
    # Assuming 'pop' and 'logbook' are the results from the evolutionary algorithm
    for record in logbook:
        print(f"Generation: {record['gen']}")
        print(f"Average Fitness: {record['avg']}")
        print(f"Standard Deviation of Fitness: {record['std']}")
        print(f"Minimum Fitness: {record['min']}")
        print(f"Maximum Fitness: {record['max']}")
        print("----------")


    # Best Solution
    best_solution = tools.selBest(pop, 1)[0]  # pop: the list to select from, 1: number of selection.
    

    # ----------------------------------------------------------------------------------- #

    print("!!! OPTIMIZATION COMPLETE !!!")
    # print(f"Best cost: {best_solution:.4f}")
    # print()
    print("")
    print("[{}] best_score: {}".format(logbook[-1]['gen'], logbook[-1]['min'][0]))


    # save the best merge configuration using original model references
    genome_pretty = ModelGenome(config.genome, trust_remote_code=trust_remote_code)
    best_config = genome_pretty.genotype_merge_config(best_solution)
    print("Best merge configuration:")
    print(best_config.to_yaml())

    if save_final_model:
        print("Saving final model...")
        run_merge(best_config, os.path.join(storage_path, "final_model"), merge_options)


def _reshard_model(
    model: ModelReference, storage_path: str, merge_cache: str, trust_remote_code: bool
) -> ModelReference:
    merged = model.merged(
        cache_dir=merge_cache,
        trust_remote_code=trust_remote_code,
    )
    out_path = os.path.join(
        storage_path,
        "input_models",
        merged.model._unique_id(),
    )

    if os.path.exists(out_path):
        logging.info(f"Using existing resharded model at {out_path}")
        return ModelReference(model=out_path)

    model_hf = transformers.AutoModelForCausalLM.from_pretrained(
        merged.model.path,
        revision=merged.model.revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        cache_dir=os.path.join(storage_path, "transformers_cache"),
    )
    model_hf.save_pretrained(
        out_path, safe_serialization=True, out_shard_size=1_000_000_000_000
    )
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model.model.path,
            revision=model.model.revision,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        tokenizer.save_pretrained(out_path)
    except Exception as e:
        logging.warning(f"Could not save tokenizer for {model.model}", exc_info=e)

    return ModelReference(model=out_path)


if __name__ == "__main__":
    main()
