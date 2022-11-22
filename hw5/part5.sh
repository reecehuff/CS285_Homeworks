for tau in 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99
do

    python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_iql_easy_supervised_lam10_tau${tau} --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile=${tau}

    python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_iql_easy_unsupervised_lam10_tau${tau} --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=10 --iql_expectile=${tau}

    python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam20_tau${tau} --use_rnd --num_exploration_steps=20000 --awac_lambda=20 --iql_expectile=${tau}

    python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam20_tau${tau} --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=20 --iql_expectile=${tau}

done