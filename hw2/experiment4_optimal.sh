for i in 50000 
do
	for j in 0.02
	do
		python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
		--discount 0.95 -n 100 -l 2 -s 32 -b $i -lr $j \
		--exp_name q4_b${i}_r${j}
		python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
		--discount 0.95 -n 100 -l 2 -s 32 -b $i -lr $j -rtg \
		--exp_name q4_b${i}_r${j}_rtg
		python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
		--discount 0.95 -n 100 -l 2 -s 32 -b $i -lr $j --nn_baseline \
		--exp_name q4_b${i}_r${j}_nnbaseline
		python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
		--discount 0.95 -n 100 -l 2 -s 32 -b $i -lr $j -rtg --nn_baseline \
		--exp_name q4_b${i}_r${j}_rtg_nnbaseline
	done
done

