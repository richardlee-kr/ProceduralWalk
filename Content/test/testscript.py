import os

# dataset
CLASS='frankengan'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=256 # scale images to this size
CROP_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image
DATASET_MODE="custom"
A_FOLDER="window_merge_labelmap"
B_FOLDER="window_realimg"
LATENT_VECTOR=8
# change aspect ratio for the test images

RESULTS_DIR='./results/frankengan'+'/{}2{}'.format(A_FOLDER,B_FOLDER)

# misc
GPU_ID=0   # gpu id
NUM_TEST=5 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images

temp="python ./integrated.py "
temp+='--dataroot ./datasets/{} '.format(CLASS)
temp+='--results_dir {} '.format(RESULTS_DIR)
temp+='--checkpoints_dir ./pretrained_models/ '
temp+='--name {} '.format(CLASS)
temp+='--direction {} '.format(DIRECTION)
temp+='--load_size {} '.format(LOAD_SIZE)
temp+='--crop_size {} '.format(CROP_SIZE)
temp+='--input_nc {} '.format(INPUT_NC)
temp+='--num_test {} '.format(NUM_TEST)
temp+='--n_samples {} '.format(NUM_SAMPLES)
temp+='--dataset_mode {} '.format(DATASET_MODE)
temp+='--A_folder {} '.format(A_FOLDER)
temp+='--B_folder {} '.format(B_FOLDER)
temp+='--nz {} '.format(LATENT_VECTOR)
temp+='--center_crop '
temp+='--no_flip '

os.system(temp)