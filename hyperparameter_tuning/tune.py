from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
import ray
from ray import train
import os

from actor_with_composite_distribution.train import do_train


def do_tune(num_samples=1000,max_num_epochs=200, cpus_per_trial=1, gpus_per_trial=0.1):
    """
    Executes hyperparameter tuning for a machine learning model using Ray Tune.

    This function configures the parameter search space, initiates Ray, and
    sets up a scheduler for experiments. It then creates a Tuner instance to
    find the optimal hyperparameters based on the specified criteria and runs
    the tuning process.

    :param num_samples: Number of samples to try. Represents different sets of
        hyperparameters to evaluate.
    :param max_num_epochs: Maximum number of epochs for training each trial.
    :param cpus_per_trial: Number of CPU cores allocated per trial.
    :param gpus_per_trial: Number of GPU units allocated per trial.
    :return: Best configuration found and the corresponding final loss.
    """
    # Define search space
    config = {
        'frames_per_batch': tune.randint(500, 5000),
        'learn_rate': tune.loguniform(1e-5, 1e-1),
        'lr_scheduling': tune.choice([True, False]),
        'num_epochs': tune.randint(1000, 10000),
        'total_frames': tune.randint(5000, 100000),
        'actor_n_layers': tune.randint(1, 3),
        'actor_hidden_features': tune.randint(1, 50),
        'critic_n_layers': tune.randint(1, 3),
        'critic_hidden_features': tune.randint(1, 50),
        'gamma': tune.uniform(0.6, 0.99),
        'lmbda': tune.uniform(0.6, 0.99),
        "average_gae": tune.choice([True, False]),
        "split_trajs": tune.choice([True, False]),
        'clip_epsilon': tune.uniform(0.15, 0.25),
        'entropy_eps': tune.loguniform(1e-4, 1e-3),
        'normalize_advantage': tune.choice([True, False]),
        'clip_value': tune.choice([True, False]),
        'separate_losses': tune.choice([True, False]),
        'critic_coef': tune.uniform(0., 1.)
    }

    # ray.init(local_mode=True)
    ray.init(num_gpus=1,num_cpus=os.cpu_count() // 3 * 2)

    # Set up the scheduler
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    # Create a Tuner instance and fit it
    tuner = Tuner(
        trainable=tune.with_resources(do_train, {"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="loss",
            mode="min",
            scheduler=scheduler
        ),
        run_config=train.RunConfig(
            stop={"loss": 0.5},
        )
    )

    results = tuner.fit()

    # Get the best configuration and trial results
    best_result = results.get_best_result(metric="loss", mode="min")
    print("Best config:", best_result.config)
    print("Best trial final loss:", best_result.metrics["loss"])


if __name__ == "__main__":
    do_tune()