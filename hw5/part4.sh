for lambda in 0.1 1 2 10 20 50
do

    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam$lambda --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=$lambda

    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=$lambda --exp_name q4_awac_easy_supervised_lam$lambda

done

for lambda in 0.1 1 2 10 20 50
do

    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam$lambda --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda=$lambda 
    
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=$lambda --exp_name q4_awac_medium_supervised_lam$lambda 

done