python main_uda.py --data_root /mnt/WXRG0348/shitaili/uda/data/cycle_gta \
--ckpt checkpoints/latest_uda_GTA_to_cityscape.pth \
--ckpt_discriminator checkpoints/latest_uda_Discriminator.pth \
--discriminator_loss_threshold 0.4 --dis_lr 0.00001 --lr 0.0001 \
--gpu_id 0,1,2,3,4,5,6,7 --batch_size 64 --val_batch_size 8 --val_interval 500 \
--total_itrs 50000 --save_val_results