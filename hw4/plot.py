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
        
        Possibilities
        Eval_AverageReturn
        Eval_StdReturn
        Eval_MaxReturn
        Eval_MinReturn
        Eval_AverageEpLen
        Train_AverageReturn
        Train_StdReturn
        Train_MaxReturn
        Train_MinReturn
        Train_AverageEpLen
        Train_EnvstepsSoFar
        TimeSinceStart
        Training_Loss
        Initial_DataCollection_AverageReturn
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
                
    return np.array(train_Y), np.array(train_std), np.array(eval_Y), np.array(eval_std) 

def AddBayesPlot(xaxis,mean,std,input_label,color1):
    lower = mean - std
    upper = mean + std
    # Accuracy plots
    plt.plot(xaxis, mean, color=color1,label=input_label)
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
base_dir = '/home/rdhuff/Desktop/CS285_Homeworks/hw3/submit/data/'
base_dir = '/home/rdhuff/Desktop/submit/data/'

#%% Problem 2 

# Read in the data
logdir = base_dir + 'hw4_q2_*/events*'
eventfile = glob.glob(logdir)[0]
print(eventfile)
train_Y, train_std, eval_Y, eval_std = get_results(eventfile)

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Add line plot
X = np.arange(len(train_Y)) + 1
AddBayesPlotWithScatter(X, train_Y, train_std, r"Train Avg Return", color1)
AddBayesPlotWithScatter(X, eval_Y, eval_std, r"Eval Avg Return", color2)


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{Obstacles} Environment",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/problem2.png", dpi=600)
plt.show()
    
#%% Problem 3

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Read in the data
logdir = base_dir + 'hw4_q3_*/events*'
eventfiles = glob.glob(logdir)

labels = ["Cheetah", "Reacher", "Obstacles"]

for i, eventfile in enumerate(eventfiles):
    print(eventfile)
    # Read in data and plot
    train_Y, train_std, eval_Y, eval_std = get_results(eventfile)
    X = np.arange(len(train_Y)) + 1
    AddBayesPlot(X, eval_Y, eval_std, labels[i], colors[i])


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Iteration}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Eval Return}",fontsize=y_bayes_font_size)
plt.title(r"\textbf{Returns with MBRL Algorithm}",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/problem3.png", dpi=600)
plt.show()
    
