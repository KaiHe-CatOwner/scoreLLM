


# nohup  accelerate launch --config_file=/hpc/home/kai_he/workshop/My_project/Score_LLM/accelerate_configs/deepspeed_zero3.yaml  run.py --gpu 7 >.log 2>&1 &


nohup  accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/Score_LLM/accelerate_configs/deepspeed_zero2.yaml  run.py \
    --gpu 6,7 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --eval_steps 300 \
     >.log 2>&1 &




nohup  accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml  run.py \
    --gpu 5,6,7 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --eval_steps 300 \
     >.log 2>&1 &


nohup  accelerate launch --config_file=/raid/hpc/hekai/WorkShop/My_project/PathLLM_new/accelerate_configs/deepspeed_zero2.yaml  run.py --gpu 4,6 --resume_from_checkpoint True --llm_requires_grad True >.log 2>&1 &



