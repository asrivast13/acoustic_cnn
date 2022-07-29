SEED = 42

FB_HEIGHT = 40  # filter banks
WIDTH = 198
COLOR_DEPTH = 1
INPUT_SHAPE = (FB_HEIGHT, WIDTH, COLOR_DEPTH)

DATA_TYPE = 'float32'
DATA_KEY = 'data'
FEAT_RANGE = None

CLASSES = ['vm', 'lp']
GENDERS = ['u']

LANGUAGE_INDEX = 0
GENDER_INDEX = 1

THRESHOLD = 0.8
BATCH_SIZE = 8
NUM_FOLDS=10
NUM_EPOCHS = 10
DO_K_FOLD_X_VALIDATION = False

FRAGMENT_DURATION = 2  # seconds

DATASET_DIST = '/Users/asrivast/Data/AMD/allpy/audio'
FEATS_DIST   = '/Users/asrivast/Data/AMD/allpy/feats'
EXPTS_INT    = '/Users/asrivast/Data/AMD/allpy/expts'
MODEL_DIST   = '/Users/asrivast/Data/AMD/allpy/models'
