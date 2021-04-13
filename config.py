from easydict import EasyDict as edict

__C = edict()

cfg = __C
# Global
__C.NUM_EPOCHS = 1
__C.LEARNING_RATE = 2e-4
__C.BATCH_SIZE = 128
# IMG
__C.IMG_SIZE = 64
__C.CHANNELS_IMG = 3
__C.Z_DIMENSION = 100
# Models
FEATURES_DISC = 64
FEATURES_GEN = 64
# Paths and saves
__C.SAVE_EACH_EPOCH = 2
__C.OUT_DIR = ""
__C.SAVE_CHECKPOINT_PATH = ""
