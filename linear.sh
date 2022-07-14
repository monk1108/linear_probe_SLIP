python -m torch.distributed.launch --nproc_per_node=2 --master_port 19502 main_linear.py  \
--arch vit_base_patch16_224  \
--pretrained /ckpts/checkpoint_best.pt --output-dir linear_output/5e-4_2gpu_bs96 \
--batch-size 96 --lr 5e-4 --wd 1e-6 \
--pretrained ckpts/try2/checkpoint_best.pt \
--log_dir linear_log/5e-4_2gpu_bs96 \
--dataset chexpert
# --gpu 2, 3, 4, 5, 6, 7 \
