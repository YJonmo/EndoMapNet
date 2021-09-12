#!/bin/bash -l

#PBS -N USProject_NN_GP
#PBS -l walltime=15:00:00
#PBS -l ncpus=1
#PBS -l mem=60GB
#PBS -l ngpus=1
#PBS -l gputype=M40
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
#mkvirtualenv PackNet
workon MonoDepth

#test
#python test_simple.py --image_path assets/0000000000.png --model_name mono+stereo_640x192
#train
Home="/home/jonmoham/Python/monodepth2-master"
cd $Home
#OUTPUT='checkpoint2/'
OUTPUT='TrainedBlenderRight2'
OUTPUT='KittiTrained'
OUTPUT='3DprintedKnee_Stereo_Mono'
OUTPUT='Stereo6_Simul_Stereo_Mono'

mkdir -p $OUTPUT
DATA_Folder=/home/jonmoham/DataForTraining/KITTI/kitti_raw/
DATA_Folder=/home/jonmoham/DataForTraining/BlenderData/
DATA_Folder=/home/jonmoham/DataForTraining/Stereo6/Simul_Kittystyle/
DATA_Folder=/home/jonmoham/DataForTraining/3DPrintedKnee/Air/Raw_Rectif/
DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style/
DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/

CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name mono_model --log_dir $OUTPUT/ --png --save_pred_disps 
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name mono_model --log_dir $OUTPUT/ --png --save_pred_disps --data_path $DATA_Folder/ --pose_model_type separate_resnet 
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name mono_model --log_dir $OUTPUT/ --png --save_pred_disps --pose_model_type posecnn 
Splits='Stereo6_Simul_Stereo'

python train.py --data_path $DATA_Folder --model_name mono_model --log_dir $OUTPUT/ --png --no_eval

python train.py --data_path $DATA_Folder --split BlenderRight --model_name mono_model --log_dir $OUTPUT/ --png --no_eval --dataset Endoscope --height 256 --width 256

python train.py --data_path $DATA_Folder --split BlenderRightStereo --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 --use_stereo

python train.py --data_path $DATA_Folder --split BlenderRightStereo --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 1 --use_stereo

#Stereo
python train.py --data_path $DATA_Folder --split 3DPrintedKnee_Stereo --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 --use_stereo

#Stereo + mono
OUTPUT='3DprintedKnee_Stereo_Mono1234_All'
mkdir -p $OUTPUT
Splits='3DPrintedKnee_Stereo1234'
PreTrained='/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono1234_PredicMask/mdp/models/weights_14'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2 -3 1 2 3 --use_stereo  --pose_model_input all --num_epochs 25

#Stereo + mono
OUTPUT='3DprintedKnee_Stereo_Mono123_PredicMask_posecnnAll'
mkdir -p $OUTPUT
Splits='3DPrintedKnee_Stereo'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2  1 2  --use_stereo --disable_automasking --predictive_mask  --pose_model_type posecnn --pose_model_input all

#Stereo + mono
OUTPUT='3DprintedKnee_Stereo_Mono123_posesharedAll'
mkdir -p $OUTPUT
Splits='3DPrintedKnee_Stereo'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2  1 2  --use_stereo   --pose_model_type shared --pose_model_input all





OUTPUT='3DprintedKnee_Stereo_MonoCropped123_posesharedAll_Smooth1-e2'
mkdir -p $OUTPUT

Splits='3DPrintedKnee_Stereo'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2  1 2  --use_stereo   --pose_model_type shared --pose_model_input all  --disparity_smoothness .01   --models_to_load encoder depth pose 


OUTPUT='3DprintedKnee_Stereo_MonoCropped1234_PredMask_posecnndAll'
mkdir -p $OUTPUT
Splits='3DPrintedKnee_Stereo1234'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2 -3 1 2 3 --use_stereo   --pose_model_type shared --pose_model_input all --pose_model_type posecnn --predictive_mask --disable_automasking




#Mono
OUTPUT='Stereo6_Simul_Mono'
mkdir -p $OUTPUT
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 1 

#Stereo + mono
Splits='Stereo6_Simul_Mono'
OUTPUT='Stereo6_Simul_Stereo_Mono'
mkdir -p $OUTPUT
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 1 --use_stereo

#Stereo + mono
Splits='Stereo6_Simul_Stereo'
OUTPUT='Stereo6_Simul_Stereo'
mkdir -p $OUTPUT
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 --use_stereo

##evaluate pose kitti
Splits='odom_9'
DATA_Folder=/home/jonmoham/DataForTraining/KITTI/kitti_odom/
python evaluate_pose.py --eval_split odom_9  --load_weights_folder $Home/PreTrained/Mono_Stereo_640x192 --data_path $DATA_Folder  

##evaluate pose custom
Splits='2020-05-26-16-07-19.txt'
Weight=$Home/3DprintedKnee_Stereo_Mono123_PredicMask/mdp/models/weights_19
DATA_Folder=/home/jonmoham/DataForTraining/3DPrintedKnee/Air/Raw_Rectif/
python evaluate_pose_custom.py --eval_split $Splits  --load_weights_folder $Weight --data_path $DATA_Folder  

##evaluate pose custom
Splits='2020-05-26-16-07-19.txt'
Weight=$Home/3DprintedKnee_Stereo_Mono123_Good/mdp/models/weights_19
DATA_Folder=/home/jonmoham/DataForTraining/3DPrintedKnee/Air/Raw_Rectif/
python evaluate_pose_custom.py --eval_split $Splits  --load_weights_folder $Weight --data_path $DATA_Folder  

Splits='2020-05-26-16-07-19.txt'
Weight=$Home/3DprintedKnee_Stereo_Mono1234_PredicMask/mdp/models/weights_14
DATA_Folder=/home/jonmoham/DataForTraining/3DPrintedKnee/Air/Raw_Rectif/
python evaluate_pose_custom.py --eval_split $Splits  --load_weights_folder $Weight --png --dataset Custom --data_path $DATA_Folder  --height 256 --width 256 --models_to_load pose_encoder encoder depth pose  --frame_ids 0 1 --use_stereo  --pose_model_type separate_resnet --pose_model_input all --models_to_load depth pose_encoder encoder pose --batch_size 1



3Dprint_water_SheeSub0-3-2-1123_Bch18_Lr1-4_2Loss_pairs_NoPose









##evaluate pose custom2
Splits='2020-10-08-13-36-36'
Splits='2020-09-18-14-03-36_sub_filt'
Weight=$Home/Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre/mdp/models/weights_1
DATA_Folder=/home/jonmoham/DataForTraining/
python evaluate.py --split $Splits  --load_weights_folder $Weight --png --dataset Custom --data_path $DATA_Folder  --height 256 --width 256  --frame_ids  0  -5 -4 -3 -2 -1 1 2 3 4 5 --use_stereo   --pose_model_input pairs --models_to_load pose_encoder encoder depth pose --batch_size 1              --num_layers 50
#--pose_model_type shared
##evaluate pose custom2
Splits='2020-05-26-16-07-19'
Weight=$Home/3DprintedKnee_Stereo_Mono123_PredicMask/mdp/models/weights_19
DATA_Folder=/home/jonmoham/DataForTraining/3DPrintedKnee/Air/Raw_Rectif/
python evaluate.py --split $Splits  --load_weights_folder $Weight --png --dataset Custom --data_path $DATA_Folder  --height 256 --width 256 --models_to_load encoder depth pose  --frame_ids 0 -1 -2  1 2  --use_stereo   --models_to_load encoder pose_encoder pose --batch_size 1 

--pose_model_input all



##evaluate pose custom2
Splits='2020-05-26-16-07-19.txt'
Weight=$Home/3DprintedKnee_Stereo_Mono123_Good/mdp/models/weights_19
DATA_Folder=/home/jonmoham/DataForTraining/3DPrintedKnee/Air/Raw_Rectif/
python evaluate_pose_custom2.py --eval_split $Splits  --load_weights_folder $Weight --data_path $DATA_Folder  --height 256 --width 256 --models_to_load encoder depth pose  --frame_ids 0 -1 -2  1 2  --use_stereo  --pose_model_type shared --pose_model_input all --models_to_load pose_encoder encoder pose




Splits='2020-10-08-13-36-36'
#Splits='2020-10-08-13-29-17'
Weight=$Home/Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyAnglRawMost_NymLay50/mdp/models/weights_0
DATA_Folder=/home/jonmoham/DataForTraining/
python evaluate.py --split $Splits  --load_weights_folder $Weight --png --dataset Custom --data_path $DATA_Folder  --height 256 --width 256 --frame_ids 0   5 --use_stereo   --pose_model_input all --models_to_load  pose2 pose_encoder2 --batch_size 1 --use_pose 1 --pose_model_type separate_resnet            --num_layers 50

--pose_model_type posecnn








Splits=(
'2020-10-08-13-36-36'
'2020-09-18-14-03-36_sub_filt'
)

Networks=(
'Cadava5_3DPrint_Sheep_filt05_Bch18_Lr5-5_pairs_PoseSup_MaskAllSum_NymLay50'
'Cadava5_3DPrint_Sheep_filt05_Bch18_Lr5-5_pairs_PoseSup_MaskAngSum_NymLay50'
'Cadava5_3DPrint_Sheep_filt06_Bch18_Lr5-5_pairs_PoseSup_Mask_NymLay50'
'Cadava5_3DPrint_Sheep_filt01_Bch18_Lr5-5_pairs_PoseSup_Mask_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_all_PoseSup_Mask_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_pairs_PoseSup_Mask_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt05_Bch18_Lr5-5_pairs_PoseSup_Mask_NymLay50'
'Cadava5_3DPrint_Sheep_filt024_Bch18_Lr5-5_pairs_PoseSup_Mask_NymLay50'
'Cadava5_3DPrint_Sheep_filt06_Bch14_Lr5-5_pairs_PoseSup_NymLay50'
'Cadava5_3DPrint_Sheep_filt05_Bch14_Lr5-5_pairs_PoseSup_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-55_Bch14_Lr5-5_pairs_PoseSup_NymLay50'
)
DATA_Folder=/home/jonmoham/DataForTraining/


Counter=0
for network in "${Networks[@]}"; do 
echo $network
Counter=0
for weight in '/home/jonmoham/Python/monodepth2-master'/$network/mdp/models/weights_2* ; do
#weight='/home/jonmoham/Python/monodepth2-master'/$network
#((Counter=Counter+1))
#Weights=$(ls -t '/home/jonmoham/Python/monodepth2-master'/$network/mdp/models | grep "weight*")
#for weight in "${Weights[@]}"; do
#if (( $(($Counter % 2 )) == 0 )); then 
echo $weight
for Split in "${Splits[@]}"; do 
python evaluate_auto_together.py --split $Split  --load_weights_folder $weight --data_path $DATA_Folder  --height 256 --width 256    --batch_size 1  --models_to_load pose_encoder encoder depth pose   #--models_to_load pose_encoder2  pose2 #   
#fi;
done
done
done

--models_to_load pose_encoder encoder depth pose


New

'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_pairs_PoseSup_MaskAllSum_NymLay50_MinDep.05Max200'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_all_PoseSup_MaskAllSum_NymLay50_MinDep.05Max200'



'Cadava5_3DPrint_Sheep_filt05_Bch18_Lr5-5_pairs_PoseSup_MaskAllSum_NymLay50'
'Cadava5_3DPrint_Sheep_filt05_Bch18_Lr5-5_pairs_PoseSup_MaskAngSum_NymLay50'
'Cadava5_3DPrint_Sheep_filt06_Bch18_Lr5-5_pairs_PoseSup_Mask_NymLay50'
'Cadava5_3DPrint_Sheep_filt01_Bch18_Lr5-5_pairs_PoseSup_Mask_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_all_PoseSup_Mask_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_pairs_PoseSup_Mask_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt05_Bch18_Lr5-5_pairs_PoseSup_Mask_NymLay50'
'Cadava5_3DPrint_Sheep_filt024_Bch18_Lr5-5_pairs_PoseSup_Mask_NymLay50'
'Cadava5_3DPrint_Sheep_filt06_Bch14_Lr5-5_pairs_PoseSup_NymLay50'
'Cadava5_3DPrint_Sheep_filt05_Bch14_Lr5-5_pairs_PoseSup_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-55_Bch14_Lr5-5_pairs_PoseSup_NymLay50'







'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_all_PoseOnlyAngle_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyAngle_NymLay50'

bad
'3Dprint_water_SheepPose_sub_filt0--5-4-3-2-112345_Bch16_Lr1-4_2Loss_all_PosOnly'




'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyAnglRawMost_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyAng_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyBoth_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch24_Lr5-5_2Loss_all_PoseOnlyBoth_NymLay50'
'3Dprint_water_SheepPose_sub_filt0-5-4-3-2-112345_Bch26_Lr5-5_2Loss_all_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheepfilt0--5-4-3-2-112345_Bch16_Lr1-4_2Loss_all_PosOnly'

'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50/mdp/models/weights_20'
'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyAnglRawMost_NymLay50/mdp/models/weights_16'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyAng_NymLay50_Pre/mdp/models/weights_7'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50/mdp/models/weights_5'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyBoth_NymLay50_Pre/mdp/models/weights_4'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch24_Lr5-5_2Loss_all_PoseOnlyBoth_NymLay50/mdp/models/weights_16'
'3Dprint_water_SheepPose_sub_filt0-5-4-3-2-112345_Bch26_Lr5-5_2Loss_all_PoseOnlyBoth_NymLay50/mdp/models/weights_6'
'Cadava5_3DPrint_Sheepfilt0--5-4-3-2-112345_Bch16_Lr1-4_2Loss_all_PosOnly/mdp/models/weights_16'












'Cadava5_3DPrint_Sheep_filt0-7-5-3-11357_Bch22_Lr5-5_2Loss_all_AngTranLoss_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-6-3-1136_Bch18_Lr1-4_2Loss_all_AngTranLoss'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_NoPoseLoss'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-3-2-1135_Bch20_Lr1-4_2Loss_all_AngTranLoss'
'Cadava5_3DPrint_Sheep0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_PoseLossDiv15_2_Pre'
'3Dprint_water_SheepPose_sub_filt0-7-5-3-11357_Bch22_Lr5-5_2Loss_all_AngTranLoss_NymLay50'
'3Dprint_water_SheepPose_sub_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre'
'3Dprint_water_SheepPose_sub_filt0-5-3-2-1135_Bch18_Lr1-4_2Loss_all_AngTranLoss'

'Cadava5_3DPrint_Sheep_filt0-7-5-3-11357_Bch22_Lr5-5_2Loss_all_AngTranLoss_NymLay50/mdp/models/weights_16'
'Cadava5_3DPrint_Sheep_filt0-6-3-1136_Bch18_Lr1-4_2Loss_all_AngTranLoss/mdp/models/weights_17'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_NoPoseLoss/mdp/models/weights_5'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre/mdp/models/weights_12'
'Cadava5_3DPrint_Sheep_filt0-5-3-2-1135_Bch20_Lr1-4_2Loss_all_AngTranLoss/mdp/models/weights_16'
'Cadava5_3DPrint_Sheep0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_PoseLossDiv15_2_Pre/mdp/models/weights_4'
'3Dprint_water_SheepPose_sub_filt0-7-5-3-11357_Bch22_Lr5-5_2Loss_all_AngTranLoss_NymLay50/mdp/models/weights_9'
'3Dprint_water_SheepPose_sub_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre/mdp/models/weights_19'
'3Dprint_water_SheepPose_sub_filt0-5-3-2-1135_Bch18_Lr1-4_2Loss_all_AngTranLoss/mdp/models/weights_21'


'Cadava5_3DPrint_Sheep0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_PoseLossDiv15_2_Pre'
'Cadava5_3DPrint_Sheep0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_pairs_PoseLossDiv15_2'
'3Dprint_water_SheeSub0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_PoseLossDiv15_2_Pre'



best depth:
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre/mdp/models/weights_12/'
good one forr depth:
#'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_pairs_PoseLossDiv15_Pre'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200'
'3Dprint_water_SheeSub0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_NoPoseLoss2'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200'

good ones pose:
'3Dprint_water_SheepfiltNDI_02_Bch14_Lr5-5_2Loss_pairs_PoseOnly_NymLay50'
'3Dprint_water_SheepfiltNDI_01_Bch14_Lr5-5_2Loss_pairs_PoseOnly_NymLay50_Pre'
'3Dprint_water_SheepPose_sub_filt0-5-4-3-2-112345_Bch26_Lr5-5_2Loss_all_PoseOnlyBoth_NymLay50/mdp/models/weights_7'
'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch24_Lr5-5_2Loss_all_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyBoth_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_NoPoseLoss/mdp/models/weights_12/'  FrameIndex = 6


ok
'3Dprint_water_SheepPose_sub_filt0-5-4-3-2-112345_Bch26_Lr5-5_2Loss_all_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheepfilt0--5-4-3-2-112345_Bch16_Lr1-4_2Loss_all_PosOnly/mdp/models/weights_16/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_2Loss_all_AngTranLoss_Pre_Pre'
'Cadava5_3DPrint_Sheep0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_PoseLossDiv15_2_Pre'
'3Dprint_water_SheepPose_sub_filt0-7-5-3-11357_Bch22_Lr5-5_2Loss_all_AngTranLoss_NymLay50'
'3Dprint_water_SheepPose_sub_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre'
'Cadava5_3DPrint_Sheep0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_pairs_PoseLossDiv15_2'
bad 
'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyAnglRawMost_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyAng_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-7-5-3-11357_Bch22_Lr5-5_2Loss_all_AngTranLoss_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-6-3-1136_Bch18_Lr1-4_2Loss_all_AngTranLoss'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-3-2-1135_Bch20_Lr1-4_2Loss_all_AngTranLoss'

Networks=(
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_all_PoseOnlyAngle_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyAngle_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyAnglRawMost_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyAng_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_all_PoseOnlyBoth_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch24_Lr5-5_2Loss_all_PoseOnlyBoth_NymLay50'
'3Dprint_water_SheepPose_sub_filt0-5-4-3-2-112345_Bch26_Lr5-5_2Loss_all_PoseOnlyBoth_NymLay50'
'Cadava5_3DPrint_Sheepfilt0--5-4-3-2-112345_Bch16_Lr1-4_2Loss_all_PosOnly'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_2Loss_all_AngTranLoss_Pre_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre'
'Cadava5_3DPrint_Sheep_filt0-7-5-3-11357_Bch22_Lr5-5_2Loss_all_AngTranLoss_NymLay50'
'Cadava5_3DPrint_Sheep_filt0-6-3-1136_Bch18_Lr1-4_2Loss_all_AngTranLoss'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_NoPoseLoss'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre'
'Cadava5_3DPrint_Sheep_filt0-5-3-2-1135_Bch20_Lr1-4_2Loss_all_AngTranLoss'
'Cadava5_3DPrint_Sheep0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_PoseLossDiv15_2_Pre'
'3Dprint_water_SheepPose_sub_filt0-7-5-3-11357_Bch22_Lr5-5_2Loss_all_AngTranLoss_NymLay50'
'3Dprint_water_SheepPose_sub_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre'
'3Dprint_water_SheepPose_sub_filt0-5-3-2-1135_Bch18_Lr1-4_2Loss_all_AngTranLoss'
'Cadava5_3DPrint_Sheep0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_pairs_PoseLossDiv15_2'
'3Dprint_water_SheeSub0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_PoseLossDiv15_2_Pre'
)
HomeHome=/home/jonmoham/Python/monodepth2-master


Counter=0
for network in "${Networks[@]}"; do 
echo $network
for weight in '/home/jonmoham/Python/monodepth2-master'/$network/mdp/models/weights_* ; do
cd $weight
#weight='/home/jonmoham/Python/monodepth2-master'/$network
#((Counter=Counter+1))
#Weights=$(ls -t '/home/jonmoham/Python/monodepth2-master'/$network/mdp/models | grep "weight*")
#for weight in "${Weights[@]}"; do
#if (( $(($Counter % 2 )) == 0 )); then 
rename  _SupVis.n .n  *_SupVis.npy
cd $HomeHome
#fi;
done
done
done

















##evaluate pose custom2
Splits='2020-08-12-10-39-17'
Weight=$Home/Cad_3D_Stereo_Mono0-1-2-3123_Pairs_BaCh22/mdp/models/weights_8
DATA_Folder=/home/jonmoham/DataForTraining/
python evaluate.py --split $Splits  --load_weights_folder $Weight --png --dataset Custom --data_path $DATA_Folder  --height 256 --width 256  --frame_ids  0 -1 -2 -3 1 2 3 --use_stereo --batch_size 1 --pose_model_type separate_resnet --pose_model_input pairs --models_to_load depth encoder pose pose_encoder 


Splits='2019-10-24-13-05-43'
Weight=$Home/Cada_Stereo6_MonoCropped0-1-212_Pairs/mdp/models/weights_22
DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/
python evaluate.py --split $Splits  --load_weights_folder $Weight --png --dataset Custom --data_path $DATA_Folder  --height 256 --width 256 --batch_size 1 --use_stereo --frame_ids  0 -1 -2  1 2   --pose_model_type separate_resnet --pose_model_input pairs --models_to_load depth encoder pose pose_encoder 





##evaluate depth custom
Weight=$Home/Cad_3D_Stereo_Mono0-1-2-3123_Pairs_shared_Bach22/mdp/models/weights_8/
DATA_Folder=/home/jonmoham/Python/monodepth2-master/DepthTestFolder/2020-10-08-13-36-36/data/
python test_simple.py --image_path $DATA_Folder --model_name $Weight 
cd ${DATA_Folder}
#zip -r Depth.zip *.jpeg
#mv Depth.zip $Weight
rm -rf *disp.npy
#rm -rf *disp.jpeg
cd $Home

Weight=$Home/Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_12/
DATA_Folder=/home/jonmoham/Python/tsdf-fusion-python-master/CustomData/2020-10-08-13-36-36/image/
python test_simple.py --image_path $DATA_Folder --model_name $Weight 
cd ${DATA_Folder}
mkdir -p ../depth
zip -r Depth.zip *.jpeg
mv Depth.zip $Weight
rm -rf *disp.npy
#rm -rf *_depth.jpeg
mv *_depth.jpeg ../depth
mv *_disp.jpeg ../depth
cd $Home





folders=(
'2019-10-24-13-01-04_LR'
'2019-10-24-13-01-41_LR'
'2019-10-24-13-02-46_LR'
'2019-10-24-13-05-43_LR'
'2019-10-24-13-06-57_LR'
'2019-10-24-13-09-02_LR'
'2019-10-24-13-11-26_LR'
'2019-10-24-13-18-50_LR'
'2019-10-24-13-20-47_LR'
'2019-10-24-13-24-37_LR'
)

'3Dprint_waterPose_Mono0-3-2-1123_pairs_Bch20_Lr3-5_Full_OKDepth/mdp/models/weights_30/'
'3Dprint_waterPose_Mono0-4-224_pairs_Bch20_Lr2-4_Full_OK_GoodDepth/mdp/models/weights_35/'
'3Dprint_waterPose_Mono03_pairs_Bch5_Lr5-5_Good/mdp/models/weights_50/'
'3Dprint_water_SheeSub0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_NoPoseLoss2/mdp/models/weights_37/'

'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_NoPoseLoss/mdp/models/weights_13/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre/mdp/models/weights_12/'


Networks=(
'3DPrint_Sheep_NDI_filt_02_Bch12_Lr5-5_2Loss_pairs_UsePose_Smoo.02_NymLay50_Pre/mdp/models/weights_02/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre/mdp/models/weights_12/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_NoPoseLoss/mdp/models/weights_13/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre/mdp/models/weights_19/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_11/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_14/'

'Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_39/'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_32/'
'Cadava5_3DPrint_Sheep_filt_024_Bch14_Lr5-5_2Loss_pairs_NoPose_Smoo.01_NymLay50/mdp/models/weights_59/'
'Cadava5_3DPrint_Sheep_filt_024_Bch14_Lr5-5_2Loss_pairs_NoPose_Smoo.005_NymLay50/mdp/models/weights_59/'
'3DPrint_Sheep_NDI_filt_02_Bch14_Lr5-5_2Loss_pairs_UsePose_Smoo.005_NymLay50/mdp/models/weights_59/'
'3DPrint_Sheep_NDI_filt_02_Bch12_Lr5-5_2Loss_pairs_NoPose_Smoo.005_NymLay50/mdp/models/weights_59/'
'Cadava5_3DPrint_Sheep_filt_02_Bch10_Lr5-5_2Loss_pairs_NoPose_Smoo.03_NymLay50/mdp/models/weights_59/'

'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre/mdp/models/weights_19/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_11/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_14/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_pairs_PoseSup_Mask_NymLay50_Pre/mdp/models/weights_03/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre/mdp/models/weights_11/'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_32/'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_37/'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200_Pre/mdp/models/weights_35/'
)
Networks=(
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre/mdp/models/weights_19/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_11/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_14/'

'Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_39/'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_32/'
'Cadava5_3DPrint_Sheep_filt_024_Bch14_Lr5-5_2Loss_pairs_NoPose_Smoo.01_NymLay50/mdp/models/weights_59/'
'Cadava5_3DPrint_Sheep_filt_024_Bch14_Lr5-5_2Loss_pairs_NoPose_Smoo.005_NymLay50/mdp/models/weights_59/'
'3DPrint_Sheep_NDI_filt_02_Bch14_Lr5-5_2Loss_pairs_UsePose_Smoo.005_NymLay50/mdp/models/weights_59/'
'3DPrint_Sheep_NDI_filt_02_Bch12_Lr5-5_2Loss_pairs_NoPose_Smoo.005_NymLay50/mdp/models/weights_59/'
'Cadava5_3DPrint_Sheep_filt_02_Bch10_Lr5-5_2Loss_pairs_NoPose_Smoo.03_NymLay50/mdp/models/weights_59/'

'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre/mdp/models/weights_19/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_11/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_14/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_pairs_PoseSup_Mask_NymLay50_Pre/mdp/models/weights_03/'
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre/mdp/models/weights_11/'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_32/'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_37/'
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200_Pre/mdp/models/weights_35/'
)
folders=('Artur_SubTotal')
for Network in "${Networks[@]}"; do
#Network='Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_39/'
Weight=$Home/$Network
Data_Root=/home/jonmoham/DataForTraining/Artur_SubTotal_26-02/Cropped_Down/
#Data_Root=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/ #2019-10-24-13-05-43/data/
for folder in "${folders[@]}"; do
Data_Folder=$Data_Root'/image_02/data'
mkdir -p $Data_Root$folder'/'${Network::-23}
python test_simple.py --image_path $Data_Folder --model_name $Weight 
cd ${Data_Folder}
rm -rf *disp.npy
mv *_depth* $Data_Root$folder'/'${Network::-23}
cd $Home
done
done

folders=(
'2019-10-24-13-01-04_LR'
'2019-10-24-13-01-41_LR'
'2019-10-24-13-02-46_LR'
'2019-10-24-13-05-43_LR'
'2019-10-24-13-06-57_LR'
'2019-10-24-13-09-02_LR'
'2019-10-24-13-11-26_LR'
'2019-10-24-13-18-50_LR'
'2019-10-24-13-20-47_LR'
'2019-10-24-13-24-37_LR'
)

folders=(
'Artur_SubTotal'
)
Weight=$Home/3Dprint_water_SheepfiltNDI_02_Bch14_Lr5-5_2Loss_pairs_PoseOnly_NymLay50/mdp/models/weights_29/
Data_Base=/home/jonmoham/DataForTraining/Artur_SubTotal_26-02/
for folder in "${folders[@]}"; do
python evaluate_auto_poseOnly.py --split $folder  --load_weights_folder $Weight --data_path $Data_Base  --height 256 --width 256    --batch_size 1  --models_to_load pose_encoder2   pose2   #--models_to_load
done


sshpass -p "Qoqenkopf8" scp jonmoham@lyra.qut.edu.au:/home/jonmoham/Python/monodepth2-master/Cad_3D_Stereo_Mono0-1-2-3123_All_Bach22_fx243_LR-5_Smooth005/mdp/models/weights_9/Depth.zip .
rename.ul Depth.zip Cad_3D_Stereo_Mono0-1-2-3123_All_Bach22_fx243_LR-5_Smooth005_W9.zip Depth.zip




sshpass -p "Qoqenkopf7" scp jonmoham@lyra.qut.edu.au:/home/jonmoham/Python/monodepth2-master/Cad_3D_Stereo_Mono0-1-2-3123_Pairs_Bach22_fx243/mdp/models/weights_9/Depth.zip .
rename.ul Depth.zip Cad_3D_Stereo_Mono0-1-2-3123_Pairs_Bach22_fx243_W9.zip Depth.zip


















#Stereo + mono Cada stereo6
OUTPUT='Cada_Stereo6_Stereo_Mono123PredMask'
Splits='Cadavar_2019-10_stereo'
mkdir -p $OUTPUT
DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/
Splits='3DPrintedKnee_Stereo'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2 1 3 --use_stereo --disable_automasking --disable_automasking 



OUTPUT='Cada_Stereo6_Stereo_Mono123PredMask'
Splits='Cadavar_2019-10_stereo'
mkdir -p $OUTPUT
DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2 1 2 --use_stereo --predictive_mask --disable_automasking 


OUTPUT='Cada_Stereo6_Stereo_Mono13PredMask'
Splits='Cadavar_2019-10_stereo'
mkdir -p $OUTPUT
DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -2  2 --use_stereo --predictive_mask --disable_automasking 


OUTPUT='Cada_Stereo6_Stereo_Mono123PreTrai'
mkdir -p $OUTPUT
PreTrained='/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono123/mdp/models/weights_19'
Splits='Cadavar_2019-10_stereo'
DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2 1 2  --use_stereo  --models_to_load encoder depth pose_encoder pose --load_weights_folder $PreTrained



OUTPUT='Cada_Stereo6_Stereo_MonoCroppedPreTraiPredMask'
mkdir -p $OUTPUT
PreTrained='/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono2_PredicMask/mdp/models/weights_19'
Splits='Cadavar_2019-10_stereo'
DATA_Folder=/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped/
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1  1  --use_stereo  --models_to_load encoder depth pose_encoder pose --load_weights_folder $PreTrained --predictive_mask --disable_automasking 


OUTPUT='Cada_Stereo6_Stereo_MonoCropped1234_PreTraiPredicMask'
mkdir -p $OUTPUT
PreTrained='/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono1234_PredicMask/mdp/models/weights_14'
Splits='Cadavar_2019-10_stereo1234'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2 -3 1 2 3 --use_stereo --predictive_mask --disable_automasking --load_weights_folder $PreTrained 


OUTPUT='Cada_Stereo6_Stereo_MonoCropped123_PreTrai_posesharedAll'
mkdir -p $OUTPUT
PreTrained='/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono123_posesharedAll/mdp/models/weights_19'
Splits='Cadavar_2019-10_stereo'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2  1 2  --use_stereo   --pose_model_type shared --pose_model_input all --load_weights_folder $PreTrained --models_to_load encoder depth pose



OUTPUT='Cada_Stereo6_Stereo_MonoCropped123_PreTrai_posesharedAll_Smooth1-e2'
mkdir -p $OUTPUT
PreTrained='/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_MonoCropped123_PreTrai_posesharedAll_Smooth1-e2/mdp/models/weights_13'
Splits='Cadavar_2019-10_stereo'
python train.py --data_path $DATA_Folder --split $Splits --log_dir $OUTPUT/ --png --no_eval --dataset Custom --height 256 --width 256 --frame_ids 0 -1 -2  1 2  --use_stereo   --pose_model_type shared --pose_model_input all  --disparity_smoothness .01   --models_to_load encoder depth pose 





exit

checkpoint_path=(
'3Dprint_water_SheepfiltNDI_024_Bch8_Lr5-5_2Loss_NoNoPose_Smoo.01_NymLay50'
'3Dprint_water_SheepfiltNDI_024_Bch8_Lr5-5_2Loss_NoNoPose_Smoo.01_NymLay50_NoStereo'
'3Dprint_water_SheepfiltNDI_024_Bch8_Lr5-5_2Loss_UsePose_Smoo.01_NymLay50_NoStereo'
'3Dprint_water_SheepfiltNDI_024_Bch8_Lr5-5_2Loss_UsePose_Smoo.01_NymLay50'
)


for models in "${checkpoint_path[@]}"; do
	echo "${models}"
	mkdir -p "${models}"
	cd ${models}
	sshpass -p "Qoqenkopf8" scp "jonmoham@lyra.qut.edu.au:/home/jonmoham/Python/monodepth2-master/${models}/mdp/train/*" .
	cd ../
done

# qsub -I -S /bin/bash -l walltime=5:00:00,ncpus=1,ngpus=1,gputype=M40,mem=20GB


activate Jupyter
jupyter lab --allow-root --ip=0.0.0.0 --no-browser
