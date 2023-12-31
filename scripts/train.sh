#!/bin/bash
if [ "$1" = "--dryrun" ]; then
	accelerate launch scripts/train.py configs/default.yml --dryrun
	gcloud storage cp gs://datasets-cdminix/libritts_feats.tar.gz /dev/shm/libritts
	tar -xzf /dev/shm/libritts/libritts_feats.tar.gz -C /dev/shm/libritts
	gcloud storage cp gs://datasets-cdminix/default_config.yaml /dev/shm/
	rm /dev/shm/hf/accelerate/default_config.yaml
	mkdir -p /dev/shm/hf/accelerate
	mv /dev/shm/default_config.yaml /dev/shm/hf/accelerate/
	exit
fi
# Machine 1
if [ "$1" = "--machine" ] && [ "$2" = "v3-1" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 4 --mask_length 1 --run_name "bin4_mask1" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 4 --mask_length 2 --run_name "bin4_mask2" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 4 --mask_length 4 --run_name "bin4_mask4" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 4 --mask_length 8 --run_name "bin4_mask8" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 4 --mask_length 16 --run_name "bin4_mask16" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 4 --mask_length 32 --run_name "bin4_mask32" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 4 --mask_length 64 --run_name "bin4_mask64" --wandb_mode=online
fi
# Machine 2
if [ "$1" = "--machine" ] && [ "$2" = "v3-2" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 4 --mask_length 128 --run_name "bin4_mask128" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 8 --mask_length 1 --run_name "bin8_mask1" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 8 --mask_length 2 --run_name "bin8_mask2" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 8 --mask_length 4 --run_name "bin8_mask4" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 8 --mask_length 8 --run_name "bin8_mask8" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 8 --mask_length 16 --run_name "bin8_mask16" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 8 --mask_length 32 --run_name "bin8_mask32" --wandb_mode=online
fi
# Machine 3
if [ "$1" = "--machine" ] && [ "$2" = "v3-3" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 8 --mask_length 64 --run_name "bin8_mask64" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 8 --mask_length 128 --run_name "bin8_mask128" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 16 --mask_length 1 --run_name "bin16_mask1" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 16 --mask_length 2 --run_name "bin16_mask2" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 16 --mask_length 4 --run_name "bin16_mask4" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 16 --mask_length 8 --run_name "bin16_mask8" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 16 --mask_length 16 --run_name "bin16_mask16" --wandb_mode=online
fi
# Machine 4
if [ "$1" = "--machine" ] && [ "$2" = "v3-4" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 16 --mask_length 32 --run_name "bin16_mask32" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 16 --mask_length 64 --run_name "bin16_mask64" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 16 --mask_length 128 --run_name "bin16_mask128" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 32 --mask_length 1 --run_name "bin32_mask1" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 32 --mask_length 2 --run_name "bin32_mask2" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 32 --mask_length 4 --run_name "bin32_mask4" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 32 --mask_length 8 --run_name "bin32_mask8" --wandb_mode=online
fi
# Machine 5
if [ "$1" = "--machine" ] && [ "$2" = "v3-5" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 32 --mask_length 16 --run_name "bin32_mask16" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 32 --mask_length 32 --run_name "bin32_mask32" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 32 --mask_length 64 --run_name "bin32_mask64" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 32 --mask_length 128 --run_name "bin32_mask128" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 64 --mask_length 1 --run_name "bin64_mask1" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 64 --mask_length 2 --run_name "bin64_mask2" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 64 --mask_length 4 --run_name "bin64_mask4" --wandb_mode=online
fi
# Machine 6
if [ "$1" = "--machine" ] && [ "$2" = "v2-1" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 64 --mask_length 8 --run_name "bin64_mask8" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 64 --mask_length 16 --run_name "bin64_mask16" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 64 --mask_length 32 --run_name "bin64_mask32" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 64 --mask_length 64 --run_name "bin64_mask64" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 64 --mask_length 128 --run_name "bin64_mask128" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 128 --mask_length 1 --run_name "bin128_mask1" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 128 --mask_length 2 --run_name "bin128_mask2" --wandb_mode=online
fi
# Machine 7
if [ "$1" = "--machine" ] && [ "$2" = "v2-2" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 128 --mask_length 4 --run_name "bin128_mask4" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 128 --mask_length 8 --run_name "bin128_mask8" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 128 --mask_length 16 --run_name "bin128_mask16" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 128 --mask_length 32 --run_name "bin128_mask32" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 128 --mask_length 64 --run_name "bin128_mask64" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 128 --mask_length 128 --run_name "bin128_mask128" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 512 --mask_length 1 --run_name "bin512_mask1" --wandb_mode=online
fi
# Machine 8
if [ "$1" = "--machine" ] && [ "$2" = "v2-3" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 512 --mask_length 2 --run_name "bin512_mask2" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 512 --mask_length 4 --run_name "bin512_mask4" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 512 --mask_length 8 --run_name "bin512_mask8" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 512 --mask_length 16 --run_name "bin512_mask16" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 512 --mask_length 32 --run_name "bin512_mask32" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 512 --mask_length 64 --run_name "bin512_mask64" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 512 --mask_length 128 --run_name "bin512_mask128" --wandb_mode=online
fi
# Machine 9
if [ "$1" = "--machine" ] && [ "$2" = "v2-4" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 1024 --mask_length 1 --run_name "bin1024_mask1" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 1024 --mask_length 2 --run_name "bin1024_mask2" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 1024 --mask_length 4 --run_name "bin1024_mask4" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 1024 --mask_length 8 --run_name "bin1024_mask8" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 1024 --mask_length 16 --run_name "bin1024_mask16" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 1024 --mask_length 32 --run_name "bin1024_mask32" --wandb_mode=online
	accelerate launch scripts/train.py configs/default.yml --bin_size 1024 --mask_length 64 --run_name "bin1024_mask64" --wandb_mode=online
fi
# Machine 10
if [ "$1" = "--machine" ] && [ "$2" = "v2-5" ]; then
	accelerate launch scripts/train.py configs/default.yml --bin_size 1024 --mask_length 128 --run_name "bin1024_mask128" --wandb_mode=online
