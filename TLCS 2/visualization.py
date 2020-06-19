import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def training_save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
        
        min_val = min(data)
        max_val = max(data)

        #apply roling window the same length as the amount of trained on scenarios
        roling_window = 6
        data = self.rollavg_pandas(data, roling_window)
       
        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(data)
        # plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        
                    
                    
    def testing_save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt.
        For training we plot the results of several episodes. We plot both the mean and the std deviation.
        """
        #data comes in as a list. First it is transformed to a numpy array
        data_array = np.asarray(data)
        
        #calculate the mean and std deviation of the gathered data
        mean = np.mean(data_array, axis=0)
        # print("mean", mean)
        std_dev = np.std(data_array, axis=0)
        steps = np.arange(len(mean))
        
        #apply a rolling window to make data more readable and less noisy
        roling_window = 10
        mean = self.rollavg_pandas(mean, roling_window)
        std_dev = self.rollavg_pandas(std_dev, roling_window)
        
        #plot figure. Plot both the error bars (std_dev) and mean.
        plt.figure(figsize=(20, 11.25)) 
        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        ax = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  

        #limit only to where the data is
        min_val = min(min(mean),min(mean-std_dev))
        max_val = max(max(mean),max(mean+std_dev))
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        

        #plot the error bars in blue
        plt.fill_between(steps, mean - std_dev, mean + std_dev, color="#00a2e8") 
        #3F5D7D
        

        #plot the means in white
        plt.plot(steps, mean, color="black", lw=2)  

        fig = plt.gcf()
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.csv'), "w") as file:
            np.savetxt(file, data_array, delimiter=",",  fmt = '%.6f')

    
    
    
    def rollavg_pandas(self, a,n):
        'Pandas rolling average over data set a with window size n. Returns a centered np array of same size'
        return np.ravel(pd.DataFrame(a).rolling(n, center=True, min_periods=1).mean().to_numpy())