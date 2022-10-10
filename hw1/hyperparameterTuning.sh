## Hyperparameter tuning the Walker Environment

#-----Walker Environment

# Varying Training steps
python cs285/scripts/run_hw1.py \
       	--expert_policy_file cs285/policies/experts/Walker2d.pkl \
       	--env_name Walker2d-v4 \
       	--exp_name bc_walk_100 \
       	--n_iter 1 \
       	--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
       	--video_log_freq -1 \
	--ep_len 1000 \
	--eval_batch_size 5000 \
	--num_agent_train_steps_per_iter 100

python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/Walker2d.pkl \
        --env_name Walker2d-v4 \
        --exp_name bc_walk_1000 \
        --n_iter 1 \
        --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
        --video_log_freq -1 \
        --ep_len 1000 \
        --eval_batch_size 5000 \
        --num_agent_train_steps_per_iter 1000

python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/Walker2d.pkl \
        --env_name Walker2d-v4 \
        --exp_name bc_walk_10000 \
        --n_iter 1 \
        --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
        --video_log_freq -1 \
        --ep_len 1000 \
        --eval_batch_size 5000 \
        --num_agent_train_steps_per_iter 10000

python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/Walker2d.pkl \
        --env_name Walker2d-v4 \
        --exp_name bc_walk_100000 \
        --n_iter 1 \
        --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
        --video_log_freq -1 \
        --ep_len 1000 \
        --eval_batch_size 5000 \
        --num_agent_train_steps_per_iter 100000

