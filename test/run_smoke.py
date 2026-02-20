from module.experiment import ExperimentConfig
from module.runner import run_tfim_experiment


def main() -> None:
    config = ExperimentConfig(
        n_qubits=2,
        j=1.0,
        h=0.7,
        layers=1,
        entanglement="linear",
        seed=3,
        random_couplings=False,
        maxiter=40,
        optimizers=("COBYLA",),
    )
    run_tfim_experiment(config, sweep_layers=(1,))


if __name__ == "__main__":
    main()
