#!/bin/bash -l

#PBS -N USProject_NN_GP
#PBS -l walltime=20:00:00
#PBS -l ncpus=1
#PBS -l mem=20GB
#PBS -l ngpus=1
#PBS -l gputype=T4
#PBS -j oe

module purge
#module load python/3.6.4-foss-2018a
# 
module load tensorflow/1.12.0-gpu-p100-foss-2018a-python-3.6.4
#or module load tensorflow/1.12.0-gpu-m40-foss-2016a-python-3.6.4
module load cuda/9.2.88

#pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html

#module purge
#module load python/3.6.4-foss-2018a-pytorch-0.5.0
#module load cuda/9.1.85

source ~/.local/bin/virtualenvwrapper.sh
export PATH=~/.local/bin:$PATH
workon MonoDepth

#test
#python test_simple.py --image_path assets/0000000000.png --model_name mono+stereo_640x192
#train
Home="/home/jonmoham/Python/monodepth2-master"
cd $Home


#DATA_Folder=/home/jonmoham/DataForTraining/3DPrintedKnee/Air/Raw_Rectif/
#OUTPUT='3DprintedKnee_Stereo_Mono1234_poseSharedAll'
#mkdir -p $OUTPUT
#Splits='3DPrintedKnee_Stereo1234'
#python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2 -3 1 2 3 --use_stereo   --pose_model_type shared --pose_model_input all --pose_model_type shared 



#OUTPUT='Cada_Stereo6_Stereo_MonoCropped1234_PreTrai_PredMask_posecnnAll'
#mkdir -p $OUTPUT
#PreTrained='/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono1234_PredMask_posecnnAll/mdp/models/weights_19'
#Splits='Cadavar_2019-10_stereo1234'
#DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/
#python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2 -3 1 2 3 --use_stereo   --pose_model_type posecnn --pose_model_input all --load_weights_folder $PreTrained --models_to_load encoder depth pose --disable_automasking --predictive_mask



#DATA_Folder=/home/jonmoham/DataForTraining/3DPrintedKnee/Air/Raw_Rectif/
#OUTPUT='3DprintedKnee_Stereo_Mono0-1-2-3123_Pairs_BatchSZ22_Smth.01'
#mkdir -p $OUTPUT
#Splits='3DPrintedKnee_Stereo1234'
#python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --batch_size 22 --frame_ids 0 -1 -2 -3 1 2 3 --use_stereo  --pose_model_input pairs --pose_model_type separate_resnet --num_epochs 25 --disparity_smoothness .01

#--disable_automasking --predictive_mask



#DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/
#OUTPUT='Cada_Stereo6_MonoCropped0-1-212_Pairs_Batch22_Smth.01'
#mkdir -p $OUTPUT 
#Splits='Cadavar_2019-10_stereo1234'
#PreTrained='/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono0-1-212_Pairs_Good/mdp/models/weights_22'
#python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --load_weights_folder $PreTrained --no_eval --dataset Custom --height 256 --width 256 --num_epochs 25 --use_stereo --frame_ids 0 -1 -2 1 2 --pose_model_input pairs  --models_to_load depth encoder pose pose_encoder --pose_model_type separate_resnet --batch_size 22 --disparity_smoothness .01 #--disable_automasking --predictive_mask # posecnn


#DATA_Folder=/home/jonmoham/DataForTraining/
#OUTPUT='Cad_3D_Mono0-11_All_Bach1_fx243_LR-5_PredMask_Smooth.005'
#mkdir -p $OUTPUT
#Splits='Cadavar_3Dprint_Mono'
#python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --batch_size 22 --#frame_ids 0 -1 1 --pose_model_input pairs --pose_model_type separate_resnet --num_epochs 30 --disparity_smoothness .005 --learning_rate 0.00001 

#python trainer_Separate.py > logs.txt


#DATA_Folder=/home/jonmoham/DataForTraining/
#OUTPUT='3D_Mono0-1-212_Pair_Bach12_LR-5_Smooth.005'
#mkdir -p $OUTPUT
#Splits='3Dprint_water_Mono'
#python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --batch_size 12 --frame_ids 0 -1 -2 1 2 --pose_model_input pairs --pose_model_type separate_resnet --num_epochs 30 --disparity_smoothness .005 --learning_rate 0.00001 --use_pose False
#--disable_automasking --predictive_mask



DATA_Folder=/home/joon/Documents/HPC/Data/
OUTPUT='3Dprint_water_SheepfiltNDI_024_Bch8_Lr5-5_2Loss_NoNoPose_Smoo.01_NymLay50_NoStereo'
#OUTPUT='TestFolde'
mkdir -p $OUTPUT
#Splits='2020-09-18-13-12-14'
#Splits='3Dprint_waterPose_Mono'
#Splits='3Dprint_waterPose_Mono_sub'
#Splits='3Dprint_water_SheepPose_sub'
#Splits='3DPrint_Sheep_filt'
#Splits='3DPrint_Sheep_quat_NDI_filt'
#Splits='3DPrint_Sheep_NDI_filt'
Splits='Cadava5_3DPrint_Sheep_NDI_filt'
#Splits='Cadava5_3DPrint_Sheep'
#Splits='Cadava5_3DPrint_Sheep_filt'
#Splits='3Dprint_water_Sheep'
#Splits='2020-10-08-13-29-17_sub'
#PreTrained='/home/jonmoham/Python/monodepth2-master/3DPrint_Sheep_NDI_filt_02_Bch12_Lr5-5_2Loss_pairs_UsePose_Smoo.02_NymLay50_Pre/mdp/models/weights_4/'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --batch_size 1 --frame_ids 0 1 --pose_model_input pairs  --num_epochs 60 --disparity_smoothness .01  --learning_rate 0.00005 --trans_weight .5 .5 1.5 --use_pose 1 --num_layers 18 
#--use_stereo # --min_depth 0.05 --max_depth 200   #--load_weights_folder  $PreTrained  --models_to_load pose pose_encoder  depth encoder # --no_ssim #   
#--use_quat 1                   --use_stereo 
# --use_pose 1
#--pose_model_type shared 
#--use_pose 1
#--num_layers 50
#--pose_model_type shared 

#--pose_model_type shared  pose_encoder


#--min_depth 0.05 --max_depth 150 

#--min_depth 0.2

# --num_layers 50

# --load_weights_folder $PreTrained  --models_to_load pose_encoder2 pose2 #--pose2_optim SGD
#--models_to_load pose_encoder2 pose2  --load_weights_folder $PreTrained --trans_weight 0.5 1 0.9
#--pose_model_type posecnn
#--pose2_optim SGD    
#--pose2_optim SGD  --pose2_loss L2
#--pose_model_type posecnn
#--models_to_load encoder pose_encoder depth pose pose_encoder2 pose2 --load_weights_folder $PreTrained

#--pose_model_type posecnn

#--num_layers 34



#--models_to_load encoder pose_encoder depth pose pose_encoder2 pose2 --load_weights_folder $PreTrained

#--learning_rate 0.00002 --use_pose 1 #> 3D_MonoPose0-1-2-3123_Pair_Bach12_Smooth.txt # --load_weights_folder $PreTrained --models_to_load encoder pose_encoder depth pose > 3D_MonoPose0-1-212_Pair_Bach12_Smooth.txt
#--disable_automasking --predictive_mask


#--disparity_smoothness .001
