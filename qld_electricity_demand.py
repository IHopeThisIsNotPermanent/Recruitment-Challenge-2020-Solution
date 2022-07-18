import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py
import numpy as np
import pandas as pd

FILES = ["QLD_Demand_2015.csv", "QLD_demand_2016.csv", "QLD_demand_2017.csv", "QLD_demand_2018.csv", "QLD_demand_2019.csv"]

def ezplot(title, xlabel, ylabel, xvals, yvals, custom = False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not custom:
        plt.scatter(xvals, yvals)
    else:
        plt.plot(custom)
    plt.show()
    plt.figure()

def mean(x):
    return(sum(x)/len(x))

def list_sum(inp):
    base = list(inp[0])
    for x in list(inp[1:]):
        base += list(x)
    return base

def csvlist2list(file_names):
    """
    

    Parameters
    ----------
    file_names : list<string>
        The list of file names, e.g [QLD_Demand_2015,QLD_demand_2016]

    Returns
    -------
    The list of enegery demand values formatted like: [[Year],[[Month],[Day,...,Day]]]

    """
    data = []
    for file in [open(x) for x in file_names]:
        filecontent = csv.reader(file, delimiter = ",")
        data.append([])
        data[len(data)-1].append([])
        current_month = 7
        for line in list(filecontent)[1:]:
            if int(line[1]) != current_month:
                data[len(data)-1].append([])
                current_month = int(line[1])
            data[len(data)-1][len(data[len(data)-1])-1].append([float(line[x]) for x in range(3, 50)])
    return data
      
def zero(x):
    if x<10:
        return "0" + str(x)
    return str(x)

def csvbad2csvgood():
    ret = [["Year", "Month", "Day"] + list(range(1,49))]
    for file in [open(x) for x in ["./New2020Data/PRICE_AND_DEMAND_2020"+zero((x+6-1)%12+1)+"_QLD1.csv" for x in range(1,13)]]:
        filecontent = csv.reader(file, delimiter = ",")
        head = 0
        filecontent = list(filecontent)[1:]
        while head != len(filecontent):
            ymd = filecontent[head][1].split(" ")[0].split("/")
            vals = []
            while head != len(filecontent) and ymd == filecontent[head][1].split(" ")[0].split("/"):
                vals.append(filecontent[head][2])
                head += 1
            if len(vals) > 1:
                ret.append(ymd+vals)
    pd.DataFrame(ret).to_csv("QLD_Demand_2020.csv", index = False, header = False)
      
if __name__ == "__main__":
    a = csvlist2list(FILES)
           
    lines = csvbad2csvgood()
    
    Demo = False
    if Demo:
        ezplot("Half-Hourly Power Demand For 1/7/2015", "time", "power demand", [4+0.5*x for x in range(len(a[0][0][0]))], a[0][0][0])
        ezplot("Daily Average Demand For 7/2015", "day", "average power demand", range(1,1+len(a[0][0])), [mean(x) for x in a[0][0]])
        all_days_2015 = [mean(x) for x in list_sum(a[0])]
        ezplot("Daily Average Demand For 2015", "day", "average power demand", range(1,1+len(all_days_2015)), all_days_2015)
        all_days_2016 = [mean(x) for x in list_sum(a[1])]
        ezplot("Daily Average Demand For 2016", "day", "average power demand", range(1,1+len(all_days_2016)), all_days_2016)
        all_days = [mean(x) for x in list_sum(list_sum(a))]
        ezplot("Daily Average Demand For every year", "day", "average power demand", range(1,1+len(all_days)), all_days)
        ezplot("Yearly Averages For All Years", "Year", "average power demand", [2015+x for x in range(5)], [mean([mean([mean(z) for z in y ]) for y in x]) for x in a])
       
        plt.title("Monthly Average For All Year Overlayed")
        plt.xlabel("Month")
        plt.ylabel("Average Power Demand")
        year_track = 2015
        for year in a:
            plt.plot([7 + y for y in range(12)], [mean([mean(day) for day in month]) for month in year],  label = str(year_track) )
            year_track += 1
        plt.legend()
        plt.show()
        plt.figure()
        
        # qq-plots
        for months in range(0,12):
            points = np.array(list_sum([[mean(day) for day in year[months]] for year in a]))
            sm.qqplot(points, line = "45", fit = True)
            py.title("qq-plot for daily averages for the month " + str((7+months-1)%12+1))
            py.show()
            
            
            
            