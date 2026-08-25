# Editable
cfg_file=./configs/resnet50_cub.yaml
output_file=./Outputs/resnet50_cub/Apr16-15-50-44_archsteen_step
dataset=cubval
iter_time=model_step42499

##############
# Not editable
ckpt=${output_file}/ckpt/${iter_time}.pth
result_pkl=${output_file}/test/${iter_time}/detections.pkl

# generate detections.pkl on test set
CUDA_VISIBLE_DEVICES=0,1 python -u tools/test_net_cub.py \
--cfg ${cfg_file} \
--load_ckpt ${ckpt} \
--dataset ${dataset} \

# report mAP
python tools/evaluation_cub.py \
--cfg ${cfg_file} \
--result_path ${result_pkl} \
--dataset ${dataset}
###########
