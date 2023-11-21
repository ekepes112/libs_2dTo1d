PROJECT_NAME = "transferLearning_2dTo1d"

BATCH_SIZE = 64
EPOCH_COUNT = 500
MSE_LOW_THRESHOLD = 15

PROJECT_NAME = "libs_2dTo1d"
DRIVE_PATH = '/content/onedrive'
RESULTS_PATH = '/content/gdrive/My Drive/projects/libs_2dTo1d/temp'
DATA_PATH = '.archive/datasets/'
CHECKPOINT_DIR = '/content/checkpoints'

DATASET_TYPE = 'means'
TRAIN_FOLDS = [1, 2, 4, 5]
TEST_FOLD = 3
COMPOUND_LIST = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
SHIFT_MAGNITUDES = [-3, -2, -1, 1, 2, 3]
NORMALIZE_TO_UNIT_MAXIMUM = False
