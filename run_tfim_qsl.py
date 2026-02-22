from module.experiment import ExperimentConfig
from module.runner import run_tfim_experiment


def main() -> None:
    config = ExperimentConfig(
        n_qubits=4,
        j=-1.0,
        h=0.5,
        boundary="open",
        layers=2,
        entanglement="linear",
        seed=17,
        random_couplings=False,
        disorder_scale=0.0,
        init_mode="random",
        optimizers=("COBYLA", "Nelder-Mead", "L-BFGS-B"),
        maxiter=200,
    )

    run_tfim_experiment(
        config,
        sweep_optimizers=config.optimizers,
        sweep_layers=(1, 2, 3, 4),
    )


if __name__ == "__main__":
    main()
