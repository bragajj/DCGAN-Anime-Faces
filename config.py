from easydict import EasyDict as edict

__C = edict()

cfg = __C
# Global
__C.NUM_EPOCHS = 1000
__C.LEARNING_RATE = 2e-4
__C.BATCH_SIZE = 128
# IMG
__C.IMG_SIZE = 64
__C.CHANNELS_IMG = 3
__C.Z_DIMENSION = 100
# Models
__C.FEATURES_DISC = 64
__C.FEATURES_GEN = 64
# Paths and saves
__C.SAVE_EACH_EPOCH = 100
__C.OUT_DIR = ""
__C.SAVE_CHECKPOINT_PATH = ""
