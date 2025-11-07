# TITLE: Test Model
# CONTACT: Gayathri Girish Nair (girishng@tcd.ie).

# IMPORTS
import time
import argparse
from cptv_cvae import ModelInterface

def init_mi(name_mi, path_config_mi):
    """ Initializes model interface.
    
    Arguments:
    name_mi {str} -- Name of model interface
    path_config_mi {str} -- Path to model interface configuration file.

    Returns:
    mi {str} -- Model interface object.
    """
    # Initialize the model interface.
    mi = ModelInterface(name=name_mi)
    mi.config.configure(data=path_config_mi)
    return mi

def test(mi, session_id, path_checkpoint, beta):
    """ Tests a given model using the model interface.
    
    Arguments:
    mi {ModelInterface} -- Model interface object.
    session_id {str} -- ID assigned to this training 
                        session.
    path_checkpoint {str} -- Path to the model weights file to 
                             load before training.
    beta {float} -- Beta value for the loss function.
    """
    time_start = time.time()
    test_id = mi.test(session_id=session_id, 
                      path_checkpoint=path_checkpoint, 
                      beta=beta)
    time_end = time.time()
    print(f"Time taken for test {test_id} =",
          f"{round(time_end - time_start, 2)} s.")
    
if __name__ == "__main__":
    """ Main function to run the testing script. """
    
    # Set up argument parser.
    parser = argparse.ArgumentParser(description="Test a model.")
    parser.add_argument("-n", "--name_mi", type=str, required=True,
                        help="Name of the model interface.")
    parser.add_argument("-p", "--path_config_mi", type=str, required=True,
                        help="Path to the model interface configuration "
                             "JSON file.")
    parser.add_argument("-cp", "--path_checkpoint", type=str, 
                        required=True,
                        help="Path to the model weights file.")
    parser.add_argument("-b", "--beta", type=float, required=True,
                        help="Beta value for the loss function.")
    parser.add_argument("-sid", "--session_id", type=str, required=True,
                        help="ID of the training session that the model"
                             " tested here is associated with.")
    
    # Parse command line arguments.
    args = parser.parse_args()

    # Initialize model interface.
    mi = init_mi(args.name_mi, args.path_config_mi)
    
    # Train the model interface.
    test(mi, args.session_id, args.path_checkpoint, args.beta)