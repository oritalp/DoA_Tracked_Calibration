"""
This script is used to run an end to end simulation, including creating or loading data, training or loading
a NN model and do an evaluation for the algorithms.
The type of the run is based on the scenrio_dict. the avialble scrorios are:
- SNR: a list of SNR values to be tested
- T: a list of number of snapshots to be tested
- eta: a list of steering vector error values to be tested
- M: a list of number of sources to be tested


"""
# Imports
import sys
from pathlib import Path
from src.signal_creation import Samples, SystemModelParams
import src.utils as utils
from datetime import datetime
import torch
import numpy as np
from doa_runner import DoARunner


def __run_simulation(**kwargs):
    SIMULATION_COMMANDS = kwargs["simulation_commands"]
    SYSTEM_MODEL_PARAMS = kwargs["system_model_params"]
    MODEL_CONFIG = kwargs["model_config"]
    TRAINING_PARAMS = kwargs["training_params"]
    plot_results = SIMULATION_COMMANDS["plot_results"]
    create_data = SIMULATION_COMMANDS["create_data"]  # Creating new dataset
    save_data = SIMULATION_COMMANDS["save_data"]  # Save created data to file
    load_data = SIMULATION_COMMANDS["load_data"]  # Load specific model for training

    print("Running simulation...")

    now = datetime.now()
    plot_path = Path(__file__).parent / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    
    # Initialize seed
    utils.set_unified_seed(SYSTEM_MODEL_PARAMS["seed"])

    # Define system model parameters - unify all parameters
    system_model_params = SystemModelParams(**SYSTEM_MODEL_PARAMS, **MODEL_CONFIG, **TRAINING_PARAMS, **SIMULATION_COMMANDS)
    
    # Add dt_string for wandb project naming
    system_model_params.dt_string_for_save = dt_string_for_save
    
    # Initialize paths
    #TODO: change the paths to be per model_connfig["model_type"]
    data_saving_path, results_path = utils.initialize_paths(Path(__file__).parent, system_model_params)
    data_loading_path = SIMULATION_COMMANDS["data_loading_path"] #ONLY USED if CREATE_DATA is False!

    # Prepare data dictionary
    data_dict = {}

    if create_data:
        signals_creator = Samples(system_model_params)
        signals_creator.set_labels(None) # creates random angles
        measurements, signals, steering_mat, noise = signals_creator.samples_creation()
        true_angles = signals_creator.get_labels()
        physical_array = signals_creator.get_array()
        physical_antennas_gains = signals_creator.get_antenna_gains()
        
        # Pack data into dictionary
        data_dict = {
            'measurements': measurements,
            'signals': signals,
            'steering_matrix': steering_mat,
            'noise': noise,
            'true_angles': true_angles,
            'physical_array': physical_array,
            'physical_antennas_gains': physical_antennas_gains
        }
        
        # Save the created data under the data_path
        if save_data:
            utils.save_data_to_file(data_saving_path, measurements, signals, steering_mat,
                                    noise, true_angles, physical_array, physical_antennas_gains, system_model_params)
    else:
        # Load data from file
        loaded_data = utils.load_data_from_file(data_loading_path)
        measurements, signals, steering_mat, noise, true_angles, physical_array, physical_antennas_gains, system_model_params = loaded_data
        
        # Add dt_string for wandb project naming (in case of loaded data)
        system_model_params.dt_string_for_save = dt_string_for_save
        
        # Pack loaded data into dictionary
        data_dict = {
            'measurements': measurements,
            'signals': signals,
            'steering_matrix': steering_mat,
            'noise': noise,
            'true_angles': true_angles,
            'physical_array': physical_array,
            'physical_antennas_gains': physical_antennas_gains
        }

    # Create and run DoA algorithm
    print(f"Running {system_model_params.model_type} algorithm...")
    doa_runner = DoARunner(system_model_params, data_dict)
    results = doa_runner.run()
    if plot_results:
        doa_runner.plot_graphs(results_path)
    
    # Save results
    results_file = results_path / "results.pkl"
    utils.save_data_to_file(results_path, results, system_model_params)
    
    # # Save trained model if training was performed
    # if doa_runner._needs_training():
    #     model_file = results_path / "trained_model.pth"
    #     doa_runner.save_model(model_file)
    
    print(f"Results saved to: {results_path}")
    print(f"RMSPE: {results.get('rmspe', 'N/A'):.6f}")
    
    return results


def run_simulation(**kwargs):
    """
    This function is used to run an end to end simulation, including creating or loading data, training or loading
    a NN model and do an evaluation for the algorithms.
    The type of the run is based on the scenrio_dict. the avialble scrorios are:
    - SNR: a list of SNR values to be tested
    - T: a list of number of snapshots to be tested
    - eta: a list of steering vector error values to be tested
    - M: a list of number of sources to be tested
    """
    #TODO: check if anything is missing for the scenario_dict option once needed.
    if kwargs["scenario_dict"] == {}:
        results = __run_simulation(**kwargs)
        return results
    
    # from this on the option of multiple scenarios is used. this is activated when we specify the sceario_dict
    # in main.py 
    # TODO: Later adjust it to our way.

    # loss_dict = {}
    # default_snr = kwargs["system_model_params"]["snr"]
    # default_T = kwargs["system_model_params"]["T"]
    # default_eta = kwargs["system_model_params"]["eta"]
    # default_m = kwargs["system_model_params"]["M"]
    # for key, value in kwargs["scenario_dict"].items():
    #     if key == "SNR":
    #         loss_dict["SNR"] = {snr: None for snr in value}
    #         print(f"Testing SNR values: {value}")
    #         for snr in value:
    #             kwargs["system_model_params"]["snr"] = snr
    #             loss = __run_simulation(**kwargs)
    #             loss_dict["SNR"][snr] = loss
    #             kwargs["system_model_params"]["snr"] = default_snr
    #     if key == "T":
    #         loss_dict["T"] = {T: None for T in value}
    #         print(f"Testing T values: {value}")
    #         for T in value:
    #             kwargs["system_model_params"]["T"] = T
    #             loss = __run_simulation(**kwargs)
    #             loss_dict["T"][T] = loss
    #             kwargs["system_model_params"]["T"] = default_T
    #     if key == "eta":
    #         loss_dict["eta"] = {eta: None for eta in value}
    #         print(f"Testing eta values: {value}")
    #         for eta in value:
    #             kwargs["system_model_params"]["eta"] = eta
    #             loss = __run_simulation(**kwargs)
    #             loss_dict["eta"][eta] = loss
    #             kwargs["system_model_params"]["eta"] = default_eta
    #     if key == "M":
    #         loss_dict["M"] = {m: None for m in value}
    #         print(f"Testing M values: {value}")
    #         for m in value:
    #             kwargs["system_model_params"]["M"] = m
    #             loss = __run_simulation(**kwargs)
    #             loss_dict["M"][m] = loss
    #             kwargs["system_model_params"]["M"] = default_m
    # if None not in list(next(iter(loss_dict.values())).values()):
    #     print_loss_results_from_simulation(loss_dict)
    #     if kwargs["simulation_commands"]["PLOT_LOSS_RESULTS"]:
    #         plot_results(loss_dict, kwargs["system_model_params"]["field_type"],
    #                      plot_acc=kwargs["simulation_commands"]["PLOT_ACC_RESULTS"],
    #                      save_to_file=kwargs["simulation_commands"]["SAVE_PLOTS"])

    # return loss_dict


if __name__ == "__main__":
    now = datetime.now()