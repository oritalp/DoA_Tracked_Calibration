"""
This script is used to run the simulation with the given parameters. The parameters can be set in the script or
by using the command line arguments. The script will run the simulation with the given parameters and save the
results to the results folder. The results will include the learning curves, RMSE results, and the accuracy results
of the evaluation. The results will be saved in the results folder in the project directory.

The script can be run with the following command line arguments:
    --snr: SNR value
    --N: Number of antennas
    --M: Number of sources
    --field_type: Field type
    --signal_nature: Signal nature
    --model_type: Model type
    --train: Train model
    --train_criteria: Training criteria
    --eval: Evaluate model
    --eval_criteria: Evaluation criteria
    --samples_size: Samples size
    --train_test_ratio: Train test ratio

"""
# Imports
import os
import warnings
import time
import matplotlib.pyplot as plt
from run_simulation import run_simulation
import argparse
import torch

#TODO: add appropriate model parameters for diffmusic after being built.
#ORI: here we set the parameters manually, but we can also argparse them which allows command line
#execution.

# Initialization
os.system("cls||clear")
plt.close("all")

scenario_dict = {
    # "SNR": [-10, -5, 0, 5, 10],
    # "T": [10, 20, 30, 50, 70, 100],
    # "eta": [0.0, 0.01, 0.02, 0.03, 0.04],
    # "M": [2, 3, 4, 5, 6, 7],
}

simulation_commands = {
    "CREATE_DATA": True,
    "data_loading_path": "datasets/N:16_M:5_T:100_snr:10_location_pert_boundary:0.25_gain_perturbation_var:0.36_seed:42/03_06_2025_15_06/data.pkl"
    # This is the path to the data file, ONLY USED if CREATE_DATA is False!
    # By now, this gets set manually.
}



system_model_params = {
    "N": 16,  # number of antennas
    "M": 5,  # number of sources
    "T": 100,  # number of snapshots
    "snr": 10,  # if defined, values in scenario_dict will be ignored 
    "bias": 0, # steering vector bias error
    "sv_noise_var": 0.0, # steering vector addative gaussian error noise variance
    "doa_range": 60, # The range of the DOA values [-doa_range, doa_range]
    "doa_resolution": .5, # The resolution of the DOA values in degrees
    "wavelength": 1, # The carrier wavelength of the signal in meters, 1 can be fine for reserch,
    # 0.06 is for wifi 5 GHz for example.

    "location_perturbation": "wavelength/4",  # The boundaries of the location perturbation in meters, 
    # insert any vlaid float between 0 and wavelength/4 or "wavelength/n" to use with refrence to the wavelength
    
    "gain_perturbation_var": 0.36, # The variance of the gain perturbation
    "seed": 42,  # Seed for reproducibility
    ###############################Fixed for now##################################
    "field_type": "Far",  # Near, Far
    "signal_type": "Narrowband",  # Narrowband, broadband
    "signal_nature": "non-coherent"  # if defined, values in scenario_dict will be ignored

}

system_model_params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = {
    "model_type": "diffMUSIC",  # diffMUSIC
    "model_params": {"window_size": 21}
}


training_params = {
    "batch_size": 128,
    "epochs": 50,
    "optimizer": "Adam",  # Adam, SGD
    "scheduler": "ReduceLROnPlateau",  # StepLR, ReduceLROnPlateau
    "learning_rate": 0.001,
    "step_size": 50,
    "use_wandb": False
}




def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with optional parameters.")
    parser.add_argument('-s', "--snr" ,type=int, help='SNR value', default=None)
    parser.add_argument('-n', "--number_of_sensors",type=int, help='Number of antennas', default=None)
    parser.add_argument('-m', "--number_of_sources", help='Number of sources, could be int or tuple for random case', default=None)
    parser.add_argument('-snap', "--number_of_snapshots", type=int, help='Number of snapshots', default=None)
    parser.add_argument('-eta', "--sv_error_var", type=float, help='Steering vector uniform error variance', default=None)
    parser.add_argument('-ft', '--field_type', type=str, help='Field type, far or near field.', default=None)
    parser.add_argument('-sn', '--signal_nature', type=str, help='Signal nature; non-coherent or coherent', default=None)
    parser.add_argument('-wav', '--wavelength', type=float, help='Wavelength of the signal in meters', default=None)
    parser.add_argument("--location_perturbation", type=float, help="Location perturbation variance", default=None)
    parser.add_argument("--gain_perturbation_var", type=float, help="Gain perturbation variance", default=None)
    parser.add_argument("--seed", type=int, help="Seed for reproducibility", default=None)


    parser.add_argument('-mt', '--model_type', type=str, help='Model type; diffMUSIC, SubspaceNet, DCD-MUSIC, DeepCNN, TransMUSIC, DR_MUSIC', default=None)


    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=None)
    parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs', default=None)
    parser.add_argument('-op', '--optimizer', type=str, help='Optimizer; Adam, SGD', default=None)
    parser.add_argument('-sch', '--scheduler', type=str, help='Scheduler; StepLR, ReduceLROnPlateau', default=None)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=None)
    parser.add_argument('-wd', '--weight_decay', type=float, help='Weight decay', default=None)
    parser.add_argument('-step', '--step_size', type=int, help='Step size', default=None)

    parser.add_argument('-w', '--wandb', action="store_true", help='Use wandb')

    parser.add_argument('-c', '--create', action="store_true", help='create a new dataset')

    return parser.parse_args()


if __name__ == "__main__":
    # torch.set_printoptions(precision=12)

    args = parse_arguments()
    if args.snr is not None:
        system_model_params["snr"] = args.snr
    if args.number_of_sensors is not None:
        system_model_params["N"] = args.number_of_sensors
    if args.number_of_sources is not None:
        # catch a case of random number of sources between two possible values
        str_m = args.number_of_sources
        if str_m.isnumeric():
            system_model_params["M"] = int(str_m)
        else:
            system_model_params["M"] = tuple(map(int, str_m.split(',')))
    if args.number_of_snapshots is not None:
        system_model_params["T"] = args.number_of_snapshots
    if args.sv_error_var is not None:
        system_model_params["eta"] = args.sv_error_var
    if args.field_type is not None:
        system_model_params["field_type"] = args.field_type
    if args.signal_nature is not None:
        system_model_params["signal_nature"] = args.signal_nature
    if args.wavelength is not None:
        system_model_params["wavelength"] = args.wavelength

    if args.location_perturbation is not None:
        system_model_params["location_perturbation"] = args.location_perturbation
    elif (isinstance(system_model_params["location_perturbation"], str)) and ("wavelength" in system_model_params["location_perturbation"]):
        int_idx = system_model_params["location_perturbation"].find("/") + 1
        system_model_params["location_perturbation"] = (system_model_params["wavelength"] /
                                                         float(system_model_params["location_perturbation"][int_idx:int_idx + 1]))

    if args.gain_perturbation_var is not None:
        system_model_params["gain_perturbation_var"] = args.gain_perturbation_var
    if args.seed is not None:
        system_model_params["seed"] = args.seed

    if args.model_type is not None:
        warnings.warn("Please make sure to configure the model parameters in the script.")
        model_config["model_type"] = args.model_type
    if model_config["model_type"] == "SubspaceNet":
        model_config["model_params"]["regularization"] = None if args.regularization == "None" else args.regularization
        model_config["model_params"]["tau"] = args.tau
        model_config["model_params"]["variant"] = args.variant

    if args.batch_size is not None:
        training_params["batch_size"] = args.batch_size
    if args.epochs is not None:
        training_params["epochs"] = args.epochs
    if args.optimizer is not None:
        training_params["optimizer"] = args.optimizer
    if args.scheduler is not None:
        training_params["scheduler"] = args.scheduler
    if args.learning_rate is not None:
        training_params["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        training_params["weight_decay"] = args.weight_decay
    if args.step_size is not None:
        training_params["step_size"] = args.step_size

    if args.wandb:
        training_params["use_wandb"] = args.wandb
    if args.create:
        simulation_commands["CREATE_DATA"] = args.create
    simulation_commands["LOAD_DATA"] = not simulation_commands["CREATE_DATA"]

    if system_model_params["location_perturbation"] > system_model_params["wavelength"] / 4:
        raise ValueError("Location perturbation should be less than wavelength/4, "
                         "This may result in oreder switching between neigboring array sensors.")

    start = time.time()
    loss = run_simulation(simulation_commands=simulation_commands,
                          system_model_params=system_model_params,
                          model_config=model_config,
                          training_params=training_params,
                          scenario_dict=scenario_dict)
    print("Total time: ", time.time() - start)
