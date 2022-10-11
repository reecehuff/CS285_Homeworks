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
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return np.array(X), np.array(Y)

def AddBayesPlot(xaxis,line,upper,lower,input_label,color1):
    # Accuracy plots
    plt.plot(xaxis, line, color=color1,label=input_label)
    plt.fill_between(xaxis, lower, upper, alpha=0.2, color=color1)
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
# base_dir = '/home/rdhuff/Desktop/CS285/hw2/run_logs4play/'

#%% Question 1 

# Read in the data
logdir = base_dir + 'q1_*/events*'
eventfile = glob.glob(logdir)[0]
print(eventfile)
X, Y = get_section_results(eventfile)

# Input label 
input_label = r"Ms. Pacman"

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Add line plot
X = X[:-1]
X = X / 1000000 # Convert to millions
AddLinePlot(X,Y,input_label,color1)

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Evironment steps [millons]}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{Ms. Pacman} Environment",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/question1.png", dpi=600)
plt.show()
    
#%% Question 2

# Read in the data
#-----------DQN
logdir = base_dir + 'q2_dqn*/events*'
eventfiles = sorted(glob.glob(logdir))
for i, file in enumerate(eventfiles):
    print(i)
    print(file)
    X, Y = get_section_results(file)
    if i == 0:
        Y_dqn = Y;
    else:
        Y_dqn = np.vstack((Y_dqn,Y))
    
Y_dqn = np.average(Y_dqn,axis=0)

#-----------DDQN
logdir = base_dir + 'q2_doubledqn*/events*'
eventfiles = sorted(glob.glob(logdir))
for i, file in enumerate(eventfiles):
    print(i)
    print(file)
    X, Y = get_section_results(file)
    if i == 0:
        Y_ddqn = Y;
    else:
        Y_ddqn = np.vstack((Y_ddqn,Y))
    
Y_ddqn = np.average(Y_ddqn,axis=0)

# Input label 
input_label1 = r"DQN"
input_label2 = r"DDQN"

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Add line plot
X = X[:-1]
X = X / 1000 # Convert to thousands
AddLinePlot(X,Y_dqn,input_label1,color1)
AddLinePlot(X,Y_ddqn,input_label2,color2)


# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Evironment steps [thousands]}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{Lunar Lander} Environment",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/question2.png", dpi=600)
plt.show()
    

#%% Question 3

# Read in the data
#-----------Across the variation of learning rate
logdir = base_dir + 'q3_*/events*'
eventfiles = sorted(glob.glob(logdir))
for i, file in enumerate(eventfiles):
    print(i)
    print(file)
    X, Y = get_section_results(file)
    if i == 0:
        Y_s = Y;
    else:
        Y_s = np.vstack((Y_s,Y))

# Input label 
input_labels = [r"lr=0.0001",r"lr=0.001",r"lr=0.01",r"lr=0.03"] 

# Plot the results 
plt.figure(figsize=(bayes_Fig_size[0],bayes_Fig_size[1]))

# Add line plot
X = X[:-1]
X = X / 1000 # Convert to thousands
for i in range(Y_s.shape[0]):
    AddLinePlot(X,Y_s[i],input_labels[i],colors[i])

# Add a danky legend
leg = plt.legend(fontsize=legend_bayes_font_size,frameon=True,shadow=True,loc="lower right")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
    
# xlabel, ylabel, and title 
plt.xlabel(r'\textbf{Evironment steps [thousands]}',fontsize=x_bayes_font_size)
plt.ylabel(r"\textbf{Average Return}",fontsize=y_bayes_font_size)
plt.title(r"Returns in \textbf{Lunar Lander} Environment",
          fontsize=24)

# Increase tick font size
plt.xticks(fontsize=x_bayes_font_size)
plt.yticks(fontsize=y_bayes_font_size)

# Save the figure
plt.savefig("figures/question3.png", dpi=600)
plt.show()
    

