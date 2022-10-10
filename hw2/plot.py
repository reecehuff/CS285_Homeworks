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
def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    V1 = []
    V2 = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Z.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                V1.append(v.simple_value)
            elif v.tag == 'Train_StdReturn':
                V2.append(v.simple_value)
                
    mat = np.stack((X,Y,Z,V1,V2))
    # print(mat)
    # print(mat.shape)
    return mat

def AddBayesPlot(xaxis,line,upper,lower,input_label,color1):
    # Accuracy plots
    plt.plot(xaxis, line, color=color1,label=input_label)
    plt.fill_between(xaxis, lower, upper, alpha=0.2, color=color1)
    # plt.errorbar(xaxis, line,  upper - line, color=color1)
    # plt.plot(xaxis, upper,'--', color=color1)
    # plt.plot(xaxis, lower,'--', color=color1)

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
base_dir = '/home/rdhuff/Desktop/submit/run_logs/'
# base_dir = '/home/rdhuff/Desktop/CS285/hw2/run_logs4play/'

#%% EXPERIMENT 1 PLOTS

# Read in the data
logdir = base_dir + '*q1_sb_*/events*'
eventfiles = sorted(glob.glob(logdir),reverse=True)
mats = []
eval_means = []
eval_stds = []
for file in eventfiles:
    print(file)
    mat = get_section_results(file)
    mats.append(mat)
    eval_means.append(mat[1])
    eval_stds.append(mat[3])
eval_means = np.array(eval_means)
eval_stds = np.array(eval_stds)

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Define your x-axis
x_axis = np.arange(100)

for i in range(eval_means.shape[0]):
    HP_mean = eval_means[i]
    HP_upper = eval_means[i] +  eval_stds[i]
    HP_lower = eval_means[i] -  eval_stds[i]
    HP_label = eventfiles[i]
    start_index = HP_label.find('sb')
    stop_index  = HP_label.find('Car')
    HP_label = HP_label[start_index:stop_index-1]
    AddBayesPlot(x_axis,HP_mean,HP_upper,HP_lower,HP_label,colors[i])

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{CartPole} Environment: Small Batch Size",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/exp1_smallBatch.png", dpi=600)
plt.show()


#-----------------------------------------------------------------------------#
#----------------------------Large batch experiments--------------------------#
#-----------------------------------------------------------------------------#

# Read in the data
logdir = base_dir + '*q1_lb_*/events*'
eventfiles = sorted(glob.glob(logdir),reverse=True)
mats = []
eval_means = []
eval_stds = []
for file in eventfiles:
    print(file)
    mat = get_section_results(file)
    mats.append(mat)
    eval_means.append(mat[1])
    eval_stds.append(mat[3])
eval_means = np.array(eval_means)
eval_stds = np.array(eval_stds)

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Define your x-axis
x_axis = np.arange(100)

for i in range(eval_means.shape[0]):
    HP_mean = eval_means[i]
    HP_upper = eval_means[i] +  eval_stds[i]
    HP_lower = eval_means[i] -  eval_stds[i]
    HP_label = eventfiles[i]
    start_index = HP_label.find('lb')
    stop_index  = HP_label.find('Car')
    HP_label = HP_label[start_index:stop_index-1]
    AddBayesPlot(x_axis,HP_mean,HP_upper,HP_lower,HP_label,colors[i])

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{CartPole} Environment: Large Batch Size",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/exp1_largeBatch.png", dpi=600)
plt.show()

#%% EXPERIMENT 2 PLOTS

#-----------------------------------------------------------------------------#
#--------------------------------all hyperparameters--------------------------#
#-----------------------------------------------------------------------------#

# Read in the data
logdir = base_dir + '*q2_b*/events*'
eventfiles = sorted(glob.glob(logdir),reverse=True)
mats = []
eval_means = []
eval_stds = []
for file in eventfiles:
    print(file)
    mat = get_section_results(file)
    mats.append(mat)
    eval_means.append(mat[1])
    eval_stds.append(mat[3])
eval_means = np.array(eval_means)
eval_stds = np.array(eval_stds)

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Define your x-axis
x_axis = np.arange(eval_means.shape[1])

for i in range(eval_means.shape[0]):
    HP_mean = eval_means[i]
    HP_upper = eval_means[i] +  eval_stds[i]
    HP_lower = eval_means[i] -  eval_stds[i]
    HP_label = eventfiles[i]
    start_index = HP_label.find('q2_b')
    stop_index  = HP_label.find('InvertedPendulum-v4')
    HP_label = HP_label[start_index+4:stop_index-1]
    AddBayesPlot(x_axis,HP_mean,HP_upper,HP_lower,HP_label,colors[i])

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{InvertedPendulum} Environment",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/exp2_pendulumAll.png", dpi=600)
plt.show()

#-----------------------------------------------------------------------------#
#------------------------------optimal hyperparameters------------------------#
#-----------------------------------------------------------------------------#

# Read in the data
logdir = base_dir + '*q2_b100_r0.05*/events*'
eventfiles = sorted(glob.glob(logdir),reverse=True)
mats = []
eval_means = []
eval_stds = []
for file in eventfiles:
    print(file)
    mat = get_section_results(file)
    mats.append(mat)
    eval_means.append(mat[1])
    eval_stds.append(mat[3])
eval_means = np.array(eval_means)
eval_stds = np.array(eval_stds)

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Define your x-axis
x_axis = np.arange(eval_means.shape[1])

for i in range(eval_means.shape[0]):
    HP_mean = eval_means[i]
    HP_upper = eval_means[i] +  eval_stds[i]
    HP_lower = eval_means[i] -  eval_stds[i]
    HP_label = eventfiles[i]
    start_index = HP_label.find('q2_b')
    stop_index  = HP_label.find('InvertedPendulum-v4')
    HP_label = HP_label[start_index+3:stop_index-1]
    AddBayesPlot(x_axis,HP_mean,HP_upper,HP_lower,HP_label,colors[i])

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="upper right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{InvertedPendulum} Environment",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/exp2_optimal.png", dpi=600)
plt.show()

#%% EXPERIMENT 3 PLOTS

# Read in the data
logdir = base_dir + '*q3*/events*'
eventfiles = sorted(glob.glob(logdir),reverse=True)
mats = []
eval_means = []
eval_stds = []
for file in eventfiles:
    print(file)
    mat = get_section_results(file)
    mats.append(mat)
    eval_means.append(mat[1])
    eval_stds.append(mat[3])
eval_means = np.array(eval_means)
eval_stds = np.array(eval_stds)

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Define your x-axis
x_axis = np.arange(eval_means.shape[1])

for i in range(eval_means.shape[0]):
    HP_mean = eval_means[i]
    HP_upper = eval_means[i] +  eval_stds[i]
    HP_lower = eval_means[i] -  eval_stds[i]
    HP_label = eventfiles[i]
    start_index = HP_label.find('r0.005_')
    stop_index  = HP_label.find('Continuous')
    HP_label = HP_label[start_index+7:stop_index]
    AddBayesPlot(x_axis,HP_mean,HP_upper,HP_lower,HP_label,colors[i])

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{LunarLander} Environment: Implementing Baselines",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/exp3_baselines.png", dpi=600)
plt.show()

#%% EXPERIMENT 4 PLOTS

#-----------------------------------------------------------------------------#
#--------------------------------all hyperparameters--------------------------#
#-----------------------------------------------------------------------------#

# Read in the data
logdir = base_dir + '*q4_search*/events*'
eventfiles = sorted(glob.glob(logdir),reverse=True)
mats = []
eval_means = []
eval_stds = []
for file in eventfiles:
    print(file)
    mat = get_section_results(file)
    mats.append(mat)
    eval_means.append(mat[1])
    eval_stds.append(mat[3])
eval_means = np.array(eval_means)
eval_stds = np.array(eval_stds)

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Define your x-axis
x_axis = np.arange(eval_means.shape[1])

for i in range(eval_means.shape[0]):
    HP_mean = eval_means[i]
    HP_upper = eval_means[i] +  eval_stds[i]
    HP_lower = eval_means[i] -  eval_stds[i]
    HP_label = eventfiles[i]
    start_index = HP_label.find('search_')
    stop_index  = HP_label.find('_rtg')
    HP_label = HP_label[start_index+7:stop_index]
    AddBayesPlot(x_axis,HP_mean,HP_upper,HP_lower,HP_label,colors[i])

# Add a danky legend
leg = plt.legend(fontsize=18,frameon=True,shadow=True,loc="upper left")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{HalfCheetah} Environment: Search",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Change limits
plt.ylim([-200, 400])

# Save the figure
plt.savefig("figures/exp4_search.png", dpi=600)
plt.show()

#-----------------------------------------------------------------------------#
#------------------------------optimal hyperparameters------------------------#
#-----------------------------------------------------------------------------#

# Read in the data
logdir = base_dir + '*q4_b*/events*'
eventfiles = sorted(glob.glob(logdir),reverse=True)
mats = []
eval_means = []
eval_stds = []
for file in eventfiles:
    print(file)
    mat = get_section_results(file)
    mats.append(mat)
    eval_means.append(mat[1])
    eval_stds.append(mat[3])
eval_means = np.array(eval_means)
eval_stds = np.array(eval_stds)

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Define your x-axis
x_axis = np.arange(eval_means.shape[1])

for i in range(eval_means.shape[0]):
    HP_mean = eval_means[i]
    HP_upper = eval_means[i] +  eval_stds[i]
    HP_lower = eval_means[i] -  eval_stds[i]
    HP_label = eventfiles[i]
    start_index = HP_label.find('q4_b')
    stop_index  = HP_label.find('_HalfCheetah')
    HP_label = HP_label[start_index+3:stop_index]
    AddBayesPlot(x_axis,HP_mean,HP_upper,HP_lower,HP_label,colors[i])

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{HalfCheetah} Environment: RTG and Baseline Variety",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/exp4_optimal.png", dpi=600)
plt.show()


#%% EXPERIMENT 5 PLOTS

# Read in the data
logdir = base_dir + '*q5*/events*'
eventfiles = sorted(glob.glob(logdir),reverse=True)
mats = []
eval_means = []
eval_stds = []
for file in eventfiles:
    print(file)
    mat = get_section_results(file)
    mats.append(mat)
    eval_means.append(mat[1])
    eval_stds.append(mat[3])
eval_means = np.array(eval_means)
eval_stds = np.array(eval_stds)

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Define your x-axis
x_axis = np.arange(eval_means.shape[1])

for i in range(eval_means.shape[0]):
    HP_mean = eval_means[i]
    HP_upper = eval_means[i] +  eval_stds[i]
    HP_lower = eval_means[i] -  eval_stds[i]
    HP_label = eventfiles[i]
    start_index = HP_label.find('r0.001_')
    stop_index  = HP_label.find('_Hopper')
    HP_label = HP_label[start_index+7:stop_index]
    AddBayesPlot(x_axis,HP_mean,HP_upper,HP_lower,HP_label,colors[i])

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="upper left")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{Hopper} Environment: Generalized Advantage Estimation",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/exp5_GAE.png", dpi=600)
plt.show()


