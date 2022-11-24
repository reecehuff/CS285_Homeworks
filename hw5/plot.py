import glob
import tensorflow as tf
import numpy as np
import os 

#---Plotting imports 
import matplotlib.pyplot as plt
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2021/bin/universal-darwin'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

#%% Make a dir for saving our results 
if not os.path.exists('figures'):
    os.makedirs('figures')

#%% Clear and close
os.system("clear")
plt.close("all")

#%% Functions
def get_results(file):
    """
        requires tensorflow==1.12.0
        
        Possibilities:

        Train_EnvstepsSoFar
        Train_AverageReturn
        Train_BestReturn
        TimeSinceStart
        Exploration_Critic_Loss
        Exploitation_Critic_Loss
        Exploration_Model_Loss
        Actor_Loss
        Eval_AverageReturn
        Eval_StdReturn
        Eval_MaxReturn
        Eval_MinReturn
        Eval_AverageEpLen
        Buffer_size
    """
    train_Y = []
    eval_Y  = []
    train_std = []
    eval_std = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_AverageReturn':
                train_Y.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                eval_Y.append(v.simple_value)
            elif v.tag == 'Train_StdReturn':
                train_std.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                eval_std.append(v.simple_value)
                
    return np.array(eval_Y), np.array(eval_std) 

def get_q_values(file):
    """
        requires tensorflow==1.12.0
        
        Possibilities:

        Train_EnvstepsSoFar
        Train_AverageReturn
        Train_BestReturn
        TimeSinceStart
        Exploitation_Critic_Loss
        Exploration_Critic_Loss
        Exploration_Model_Loss
        Exploitation_Data_q-values
        Exploitation_OOD_q-values
        Exploitation_CQL_Loss
        Eval_AverageReturn
        Eval_StdReturn
        Eval_MaxReturn
        Eval_MinReturn
        Eval_AverageEpLen
        Buffer_size
    """
    train_Y = []
    eval_Y  = []
    train_std = []
    eval_std = []
    data_q = []
    ood_q = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_AverageReturn':
                train_Y.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                eval_Y.append(v.simple_value)
            elif v.tag == 'Train_StdReturn':
                train_std.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                eval_std.append(v.simple_value)
            elif v.tag == 'Exploitation_Data_q-values':
                data_q.append(v.simple_value)
            elif v.tag == 'Exploitation_OOD_q-values':
                ood_q.append(v.simple_value)

    return np.array(ood_q)         
    # return np.array(data_q)

def AddBayesPlot(xaxis,mean,std,input_label,color1):
    lower = mean - std
    upper = mean + std
    # Accuracy plots
    plt.plot(xaxis, mean, color=color1,label=input_label) #, marker='.', markersize=10)
    plt.plot()
    plt.fill_between(xaxis, lower, upper, alpha=0.2, color=color1)
    # plt.errorbar(xaxis, line,  upper - line, color=color1)
    # plt.plot(xaxis, upper,'--', color=color1)
    # plt.plot(xaxis, lower,'--', color=color1)

def AddBayesPlotWithScatter(xaxis,mean,std,input_label,color1):
    lower = mean - std
    upper = mean + std
    # Accuracy plots
    plt.plot(xaxis, mean, color=color1,label=input_label)
    plt.fill_between(xaxis, lower, upper, alpha=0.2, color=color1)
    plt.scatter(xaxis, mean, s=100, c=color1)
    # plt.errorbar(xaxis, line,  upper - line, color=color1)
    # plt.plot(xaxis, upper,'--', color=color1)
    # plt.plot(xaxis, lower,'--', color=color1)

def AddLinePlot(xaxis,line,input_label,color1):
    # Accuracy plots
    plt.plot(xaxis, line, color=color1,label=input_label)
    

#%% Plotting hyperparameters
# Colors
color1 = (105/255,53/255 ,157/255) 
color2 = (0/255  ,130/255,126/255) 
color3 = (200/255, 75/255,109/255)
color4 = (252/255,194/255,  1/255)
color5 = (118/255,122/255,121/255)
color6 = (0/255  ,255/255,255/255)
color7 = (72/255 ,118/255,255/255)
color8 = (255/255,131/255,250/255)
color9 = (127/255,255/255,0/255  )


colors = [color1, color2, color3, color4, color5, color6, color7, color8, color9]

# Size
bayes_Fig_size = [10,8]
x_bayes_font_size = 24 
y_bayes_font_size = 24 
legend_bayes_font_size = 20

# Base dir (where all of the folders that contain the event files are)
base_dir = '/home/rdhuff/Desktop/submit/data/'

#%% Part 1

#==============================================================================#
#                         Part 1: Part 1, Environment 1                        #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q1_env1*/events*'
eventfiles = sorted(glob.glob(logdir))

# Find the labels
label_strs = []
for fn in eventfiles:
    start = fn.find('q1_') + 3
    end = fn.find('_Point')
    label_strs.append(fn[start:end])

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassEasy} Environment",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part1_part1_env1.png", dpi=600)

#==============================================================================#
#                         Part 1: Part 1, Environment 2                        #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q1_env2*/events*'
eventfiles = sorted(glob.glob(logdir))

# Find the labels
label_strs = []
for fn in eventfiles:
    start = fn.find('q1_') + 3
    end = fn.find('_Point')
    label_strs.append(fn[start:end])

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part1_part1_env2.png", dpi=600)

#==============================================================================#
#                         Part 1: Part 2, Algorithms                           #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q1_alg*/events*'
eventfiles = sorted(glob.glob(logdir), reverse=True)

# Find the labels
label_strs = []
for fn in eventfiles:
    start = fn.find('q1_') + 3
    end = fn.find('_Point')
    label_strs.append(fn[start:end])

label_strs = ['Medium', 'Hard']

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} and \textbf{PointmassHard} Environments",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part1_part2_algs.png", dpi=600)

#==============================================================================#
#                         Part 1: Part 2, Medium Comparison                    #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q1_*_PointmassMedium*/events*'
eventfiles = glob.glob(logdir)

# Find the labels
label_strs = ['Random', 'RND', 'Algorithm']

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part1_part2_compare.png", dpi=600)

# %% Part 2

#==============================================================================#
#                                Part 2: Part 1                                #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q2_dqn_Point*/events*'
eventfile1 = glob.glob(logdir)

logdir = base_dir + 'hw5_expl_q2_cql_Point*/events*'
eventfile2 = glob.glob(logdir)

logdir = base_dir + 'hw5_expl_q2_cql_shift*_Point*/events*'
eventfile3 = glob.glob(logdir)

eventfiles = [eventfile1[0], eventfile2[0], eventfile3[0]]

# Find the labels
label_strs = []
for fn in eventfiles:
    start = fn.find('q2_') + 3
    end = fn.find('_Point')
    label_strs.append(fn[start:end])

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part2_part1.png", dpi=600)

# Read in the data
logdir = base_dir + 'hw5_expl_q2_dqn_Point*/events*'
eventfile1 = glob.glob(logdir)

logdir = base_dir + 'hw5_expl_q2_cql_Point*/events*'
eventfile2 = glob.glob(logdir)

logdir = base_dir + 'hw5_expl_q2_cql_shift*_Point*/events*'
eventfile3 = glob.glob(logdir)

eventfiles = [eventfile1[0], eventfile2[0], eventfile3[0]]

# Find the labels
label_strs = []
for fn in eventfiles:
    start = fn.find('q2_') + 3
    end = fn.find('_Point')
    label_strs.append(fn[start:end])

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y = get_q_values(file)
    eval_std = np.zeros(len(eval_Y))

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="upper left")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Data Q Values}",fontsize=y_bayes_font_size)
plt.title(r"Q-Values in \textbf{PointmassMedium} Environment",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part2_part1_qvalues.png", dpi=600)

#==============================================================================#
#                                Part 2: Part 2                                #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q2_dqn_numsteps_*/events*'
eventfiles = sorted(glob.glob(logdir))

logdir = base_dir + 'hw5_expl_q2_cql_numsteps_*/events*'
eventfiles2 = sorted(glob.glob(logdir))

for file in eventfiles2:
    eventfiles.append(file)

print(eventfiles)

# Find the labels
label_strs = []
for fn in eventfiles:
    start = fn.find('q2_') + 3
    end = fn.find('_Point')
    label_strs.append(fn[start:end])

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="upper left")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part2_part2.png", dpi=600)

#==============================================================================#
#                                Part 2: Part 2    HARD                        #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q2_*_hard_numsteps_*/events*'
eventfiles = glob.glob(logdir)

# Find the labels
label_strs = []
for fn in eventfiles:
    start = fn.find('q2_') + 3
    end = fn.find('_Point')
    label_strs.append(fn[start:end])

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="upper left")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassHard} Environment",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part2_part2_hard.png", dpi=600)

#==============================================================================#
#                                Part 2: Part 3                                #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q2_alpha*/events*'
eventfiles = glob.glob(logdir)

# Find the labels
label_strs = []
for fn in eventfiles:
    start = fn.find('alpha') + 5
    end = fn.find('_Point')
    label_strs.append(fn[start:end])

# Read in the data
logdir = base_dir + 'hw5_expl_q2_dqn_Point*/events*'
eventfile1 = glob.glob(logdir)[0]

logdir = base_dir + 'hw5_expl_q2_cql_Point*/events*'
eventfile2 = glob.glob(logdir)[0]

eventfiles.append(eventfile1)
eventfiles.append(eventfile2)

label_strs.append('0.0')
label_strs.append('0.1')

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, 'alpha = ' + label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="upper left")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part2_part3.png", dpi=600)


#%% Part 3

#==============================================================================#
#                                Part 3: Medium                                #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q3_medium_*/events*'
eventfiles = glob.glob(logdir)

# data from part 2 part 2
logdir = base_dir + 'hw5_expl_q2_cql_num*/events*'
eventfile2 = sorted(glob.glob(logdir))[0]

logdir = base_dir + 'hw5_expl_q2_dqn_num*/events*'
eventfile3 = sorted(glob.glob(logdir))[0]

# Add part 1 data
logdir = base_dir + 'hw5_expl_q1_env2_rnd*/events*'
eventfile4 = sorted(glob.glob(logdir))[0]

eventfiles.append(eventfile2)
eventfiles.append(eventfile3)
eventfiles.append(eventfile4)

# Find the labels
label_strs = ['CQL Supervised', 'DQN Supervised', 'CQL Unsupervised nsteps=15000', 'DQN Unsupervised nsteps=15000', 'RND from Part 1']

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="upper left")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment \textbf{Supervised}",fontsize=24)

# Change the x limits
plt.xlim([-50000, np.max(X) + 1000])

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part3_medium.png", dpi=600)

#==============================================================================#
#                                Part 3: Hard                                  #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q3_hard_*/events*'
eventfiles = sorted(glob.glob(logdir))

# data from part 2 part 2
logdir = base_dir + 'hw5_expl_q2_*hard_numsteps_15000*/events*'
eventfiles2 = sorted(glob.glob(logdir))

for file in eventfiles2:
    eventfiles.append(file)

# Find the labels
label_strs = ['CQL Supervised', 'DQN Supervised', 'CQL Unsupervised nsteps=15000', 'DQN Unsupervised nsteps=15000']


 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="upper left")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassHard} Environment \textbf{Supervised}",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part3_hard.png", dpi=600)

#%% Part 4

#==============================================================================#
#                           Part 4: Easy, Supervised                           #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q4_awac_easy_super*/events*'
eventfiles = glob.glob(logdir)
lambdas = []
lambda_strs = []
for fn in eventfiles:
    start = fn.find('lam') + 3
    end = fn.find('_Point')
    lambdas.append(float(fn[start:end]))
    lambda_strs.append(fn[start:end])
reordered = np.argsort(lambdas)
eventfiles = np.array(eventfiles)[reordered]
lambdas = np.array(lambdas)[reordered]
lambda_strs = np.array(lambda_strs)[reordered]

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, r"$\lambda$ = " + lambda_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassEasy} Environment \textbf{Supervised}",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part4_easy_supervised.png", dpi=600)



#==============================================================================#
#                           Part 4: Medium, Supervised                         #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q4_awac_medium_super*/events*'
eventfiles = glob.glob(logdir)
lambdas = []
lambda_strs = []
for fn in eventfiles:
    start = fn.find('lam') + 3
    end = fn.find('_Point')
    lambdas.append(float(fn[start:end]))
    lambda_strs.append(fn[start:end])
reordered = np.argsort(lambdas)
eventfiles = np.array(eventfiles)[reordered]
lambdas = np.array(lambdas)[reordered]
lambda_strs = np.array(lambda_strs)[reordered]

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, r"$\lambda$ = " + lambda_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment \textbf{Supervised}",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part4_medium_supervised.png", dpi=600)


#==============================================================================#
#                           Part 4: Easy, Unsupervised                         #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q4_awac_easy_unsuper*/events*'
eventfiles = glob.glob(logdir)
lambdas = []
lambda_strs = []
for fn in eventfiles:
    start = fn.find('lam') + 3
    end = fn.find('_Point')
    lambdas.append(float(fn[start:end]))
    lambda_strs.append(fn[start:end])
reordered = np.argsort(lambdas)
eventfiles = np.array(eventfiles)[reordered]
lambdas = np.array(lambdas)[reordered]
lambda_strs = np.array(lambda_strs)[reordered]

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, r"$\lambda$ = " + lambda_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassEasy} Environment \textbf{Unsupervised}",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part4_easy_unsupervised.png", dpi=600)



#==============================================================================#
#                           Part 4: Medium, Unsupervised                       #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q4_awac_medium_unsuper*/events*'
eventfiles = glob.glob(logdir)
lambdas = []
lambda_strs = []
for fn in eventfiles:
    start = fn.find('lam') + 3
    end = fn.find('_Point')
    lambdas.append(float(fn[start:end]))
    lambda_strs.append(fn[start:end])
reordered = np.argsort(lambdas)
eventfiles = np.array(eventfiles)[reordered]
lambdas = np.array(lambdas)[reordered]
lambda_strs = np.array(lambda_strs)[reordered]

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, r"$\lambda$ = " + lambda_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment \textbf{Unsupervised}",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part4_medium_unsupervised.png", dpi=600)


#%% Part 5

#==============================================================================#
#                           Part 5: Easy, Unsupervised                         #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q5_iql_easy_unsuper*/events*'
eventfiles = glob.glob(logdir)
lambdas = []
lambda_strs = []
for fn in eventfiles:
    start = fn.find('tau') + 3
    end = fn.find('_Point')
    lambdas.append(float(fn[start:end]))
    lambda_strs.append(fn[start:end])
reordered = np.argsort(lambdas)
eventfiles = np.array(eventfiles)[reordered]
lambdas = np.array(lambdas)[reordered]
lambda_strs = np.array(lambda_strs)[reordered]

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, r"$\tau$ = " + lambda_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassEasy} Environment \textbf{Unsupervised}",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part5_easy_unsupervised.png", dpi=600)

#==============================================================================#
#                           Part 5: Easy, Supervised                           #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q5_iql_easy_super*/events*'
eventfiles = glob.glob(logdir)
lambdas = []
lambda_strs = []
for fn in eventfiles:
    start = fn.find('tau') + 3
    end = fn.find('_Point')
    lambdas.append(float(fn[start:end]))
    lambda_strs.append(fn[start:end])
reordered = np.argsort(lambdas)
eventfiles = np.array(eventfiles)[reordered]
lambdas = np.array(lambdas)[reordered]
lambda_strs = np.array(lambda_strs)[reordered]

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, r"$\tau$ = " + lambda_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassEasy} Environment \textbf{Supervised}",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part5_easy_supervised.png", dpi=600)

#==============================================================================#
#                           Part 5: Medium, Unsupervised                       #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q5_iql_medium_unsuper*/events*'
eventfiles = glob.glob(logdir)
lambdas = []
lambda_strs = []
for fn in eventfiles:
    start = fn.find('tau') + 3
    end = fn.find('_Point')
    lambdas.append(float(fn[start:end]))
    lambda_strs.append(fn[start:end])
reordered = np.argsort(lambdas)
eventfiles = np.array(eventfiles)[reordered]
lambdas = np.array(lambdas)[reordered]
lambda_strs = np.array(lambda_strs)[reordered]

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, r"$\tau$ = " + lambda_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment \textbf{Unsupervised}",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part5_medium_unsupervised.png", dpi=600)


#==============================================================================#
#                           Part 5: Medium, Supervised                         #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q5_iql_medium_super*/events*'
eventfiles = glob.glob(logdir)
lambdas = []
lambda_strs = []
for fn in eventfiles:
    start = fn.find('tau') + 3
    end = fn.find('_Point')
    lambdas.append(float(fn[start:end]))
    lambda_strs.append(fn[start:end])
reordered = np.argsort(lambdas)
eventfiles = np.array(eventfiles)[reordered]
lambdas = np.array(lambdas)[reordered]
lambda_strs = np.array(lambda_strs)[reordered]

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, r"$\tau$ = " + lambda_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment \textbf{Supervised}",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part5_medium_supervised.png", dpi=600)


#==============================================================================#
#                      Part 5: MEDIUM, ULTIMATE COMPARISON                     #
#==============================================================================#

# Read in the data
logdir = base_dir + 'hw5_expl_q2_*_shift_*/events*'
eventfile1 = glob.glob(logdir)[0]

logdir = base_dir + 'hw5_expl_q4_awac_medium_super*_lam20_*/events*'
eventfile2 = glob.glob(logdir)[0]

logdir = base_dir + 'hw5_expl_q5_iql_medium_super*_tau0.9_*/events*'
eventfile3 = glob.glob(logdir)[0]

eventfiles = [eventfile1, eventfile2, eventfile3]

label_strs = ['CQL', 'AWAC', 'IQL']

 # Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Loop through the files and plot the results
for i, file in enumerate(eventfiles):
    print(file)
    eval_Y, eval_std = get_results(file)

    # Add line plot
    X = np.arange(len(eval_Y)) * 1000
    AddBayesPlot(X, eval_Y, eval_std, label_strs[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{PointmassMedium} Environment",fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/part5_medium_ultimate.png", dpi=600)