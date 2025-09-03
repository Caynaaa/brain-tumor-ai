# Import necessary libraries
from pytorch_lightning.loggers import TensorBoardLogger

# Get a TensorBoard logger instance for logging training metrics
def get_logger(log_dir='logs', name='2D_binary_classifier'):
    """
    Returns a TensorBoard logger.

    Args:
        log_dir (str): Path where TensorBoard logs will be saved.
        name (str): Name of the model/run (used as folder name).

    Returns:
        TensorBoardLogger: Configured logger instance.
    """
    
    # Create a TensorBoard logger
    # This will save logs to logs/2D_binary_classifier/version_x
    return TensorBoardLogger(
        save_dir = log_dir, 
        name = name
    )