for i in 100 500 1000
do
	for j in 0.02 0.03 0.05
       	do  
		python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
		--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $i -lr $j -rtg \
		--exp_name q2_b${i}_r${j} 
	done
done

