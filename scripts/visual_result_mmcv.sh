# Editable
dataset=voc_val
result_file=result/CIM-VOC-val.json
save_dir=./vis_VOCO12_val

##############
# Not editable
# train CIM
python ./visualize/vis_json_mmcv.py --dataset ${dataset} \
--result_file ${result_file} \
--save_dir ${save_dir}
############