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
    # plt.fill_between(xaxis, lower, upper, alpha=0.2, color=color1)
    plt.errorbar(xaxis, line,  upper - line, color=color1)
    # plt.plot(xaxis, upper,'--', color=color1)
    # plt.plot(xaxis, lower,'--', color=color1)

#%% Plotting hyperparameters
# Colors
color1 = (105/255,53/255 ,157/255) 
color2 = (0/255  ,130/255,126/255) 
color3 = (200/255, 75/255,109/255)
color4 = (252/255,194/255,  1/255)
color5 = (118/255,122/255,121/255)

# Size
bayes_Fig_size = [10,8]
x_bayes_font_size = 24 
y_bayes_font_size = 24 
legend_bayes_font_size = 20

#%% Read in the data 
#-----------------------------------------------------------------------------#
#---------------------------HYPERPARAMETER TUNING PLOTS-----------------------#
#-----------------------------------------------------------------------------#

# BC Walking
logdir = '/home/rdhuff/Desktop/CS285/hw1/data/*bc_walk_*/events*'
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
eval_means = np.array(eval_means).flatten()
eval_stds = np.array(eval_stds).flatten()

training_steps = np.array([100,1000,10000,100000]).flatten()

#-----------------------------------------------------------------------------#
#-----------------------------DEFAULT SETTING PLOTS---------------------------#
#-----------------------------------------------------------------------------#

# BC Ant
logdir = '/home/rdhuff/Desktop/CS285/hw1/data/q1*ant*/events*'
eventfile = glob.glob(logdir)[0]

mat_BA = get_section_results(eventfile)
for i, (x, y) in enumerate(zip(mat_BA[0], mat_BA[1])):
    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

# BC Walker
logdir = '/home/rdhuff/Desktop/CS285/hw1/data/q1*walker*/events*'
eventfile = glob.glob(logdir)[0]

mat_BW = get_section_results(eventfile)
for i, (x, y) in enumerate(zip(mat_BW[0], mat_BW[1])):
    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

# Dagger Ant
logdir = '/home/rdhuff/Desktop/CS285/hw1/data/q2*ant*/events*'
eventfile = glob.glob(logdir)[0]

mat_DA = get_section_results(eventfile)
for i, (x, y) in enumerate(zip(mat_DA[0], mat_DA[1])):
    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

# Dagger Walker
logdir = '/home/rdhuff/Desktop/CS285/hw1/data/q2*walker*/events*'
eventfile = glob.glob(logdir)[0]

mat_DW = get_section_results(eventfile)
for i, (x, y) in enumerate(zip(mat_DW[0], mat_DW[1])):
    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

#%% Process the data
number_of_DAgger_iters = np.arange(10)

#%% Plot the dankies 
#-----------------------------------------------------------------------------#
#-------------------------HYPERPARAMETER TUNING PLOT--------------------------#
#-----------------------------------------------------------------------------#
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

HP_mean = eval_means
HP_upper = eval_means +  eval_stds
HP_lower = eval_means -  eval_stds
HP_label = r"Walker \texttt{EvalReturn}"
AddBayesPlot(training_steps,HP_mean,HP_upper,HP_lower,HP_label,color1)

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="upper left")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Amount of training steps}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{Walker} Environment",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Change x axis to log
plt.xscale("log")

# Save the figure
plt.savefig("figures/hyperparameterTuning_Walker.png", dpi=600)
plt.show()


#-----------------------------------------------------------------------------#
#------------------------------Question 2: Ant--------------------------------#
#-----------------------------------------------------------------------------#
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

DA_mean = mat_DA[1]
DA_upper = mat_DA[1] +  mat_DA[3]
DA_lower = mat_DA[1] -  mat_DA[3]
DA_label = "DAgger Policy"
AddBayesPlot(number_of_DAgger_iters,DA_mean,DA_upper,DA_lower,DA_label,color1)

DA_mean = mat_DA[2]
DA_upper = mat_DA[2] +  mat_DA[4]
DA_lower = mat_DA[2] -  mat_DA[4]
DA_label = "Expert Policy"
AddBayesPlot(number_of_DAgger_iters,DA_mean,DA_upper,DA_lower,DA_label,color2)

x_axis = np.array([[np.min(number_of_DAgger_iters), np.max(number_of_DAgger_iters)]]).flatten()
DA_mean = mat_BW[1]
DA_upper = mat_BW[1] +  mat_BW[3]
DA_lower = mat_BW[1] -  mat_BW[3]
DA_mean = np.stack((DA_mean, DA_mean)).flatten()
DA_upper = DA_mean # np.stack((DA_upper, DA_upper)).flatten()
DA_lower = np.stack((DA_lower, DA_lower)).flatten()
DA_label = "Behavioral Cloning Policy"
AddBayesPlot(x_axis,DA_mean,DA_upper,DA_lower,DA_label,color3)

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="best")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{Ant} Environment",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/question2_Ant.png", dpi=600)
plt.show()

#-----------------------------------------------------------------------------#
#------------------------------Question 2: Walker-----------------------------#
#-----------------------------------------------------------------------------#

plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

DW_mean = mat_DW[1]
DW_upper = mat_DW[1] +  mat_DW[3]
DW_lower = mat_DW[1] -  mat_DW[3]
DW_label = "DAgger Policy"
AddBayesPlot(number_of_DAgger_iters,DW_mean,DW_upper,DW_lower,DW_label,color1)

DW_mean = mat_DW[2]
DW_upper = mat_DW[2] +  mat_DW[4]
DW_lower = mat_DW[2] -  mat_DW[4]
DW_label = "Expert Policy"
AddBayesPlot(number_of_DAgger_iters,DW_mean,DW_upper,DW_lower,DW_label,color2)

x_axis = np.array([[np.min(number_of_DAgger_iters), np.max(number_of_DAgger_iters)]]).flatten()
DW_mean = mat_BW[1]
DW_upper = mat_BW[1] +  mat_BW[3]
DW_lower = mat_BW[1] -  mat_BW[3]
DW_mean = np.stack((DW_mean, DW_mean)).flatten()
DW_upper = DW_mean # np.stack((DW_upper, DW_upper)).flatten()
DW_lower = np.stack((DW_lower, DW_lower)).flatten()
DW_label = "Behavioral Cloning Policy"
AddBayesPlot(x_axis,DW_mean,DW_upper,DW_lower,DW_label,color3)

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="best")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{Walker} Environment",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/question2_Walker.png", dpi=600)
plt.show()

