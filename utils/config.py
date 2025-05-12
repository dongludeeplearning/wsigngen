import os

SMPL_DATA_PATH = "./body_models/smpl"

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

SMPLX_DATA_PATH = "./body_models/smplx"
SMPLX_MODEL_PATH = os.path.join(SMPLX_DATA_PATH, "SMPLX_NEUTRAL.npz")

use_skeleton55 = False
if use_skeleton55: 
    SMPLX_KINTREE_PATH =os.path.join(SMPLX_DATA_PATH, "kin_skl55_smplx.pkl")  # skeleton55
else:
    SMPLX_KINTREE_PATH =os.path.join(SMPLX_DATA_PATH, "kin_pose53_smplx.pkl")  # pose 53


ROT_CONVENTION_TO_ROT_NUMBER = {
    'legacy': 23,
    'no_hands': 21,
    'full_hands': 51,
    'mitten_hands': 33,
}

GENDERS = ['neutral', 'male', 'female']
NUM_BETAS = 10