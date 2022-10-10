for i in 10000 30000 50000
do
        for j in 0.005 0.01 0.02
        do
                python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
		--discount 0.95 -n 100 -l 2 -s 32 -b $i -lr $j -rtg --nn_baseline \
		--exp_name q4_search_b${i}_lr${j}_rtg_nnbaseline
        done
done

