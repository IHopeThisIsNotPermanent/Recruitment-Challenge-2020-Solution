import csv, math, random
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pylab as py
import numpy as np
import pandas as pd

from scipy.integrate import quad

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
    csvlist2list converts the nice csv files to a python list

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
            data[len(data)-1][len(data[len(data)-1])-1].append([float(line[x]) for x in range(3, 51)])
    return data
      


def zero(x):
    if x<10:
        return "0" + str(x)
    return str(x)

def PriceNDemand(year, months):
    return ["./New2020Data/PRICE_AND_DEMAND_"+str(year)+zero(x)+"_QLD1.csv" for x in months]

def updateTimeNDate(string):
    ymd = string.split(" ")[0].split("/")
    time = string.split(" ")[1].split(":")
    time = int(time[0])+int(time[1])/30*0.5
    return ymd, time

def csvbad2csvgood():
    vals = []
    act_ymd = [2020,7,1]
    ret = [["Year", "Month", "Day"] + list(range(1,49))]
    for file in [open(x) for x in PriceNDemand(2020,range(7,12)) + PriceNDemand(2021, range(1,8))]:
        filecontent = csv.reader(file, delimiter = ",")
        head = 0
        filecontent = list(filecontent)[1:]
        ymd, time = updateTimeNDate(filecontent[head][1])
        while head < len(filecontent) and time != 4:
            head += 1
            ymd, time = updateTimeNDate(filecontent[head][1])
        while head < len(filecontent):
            ymd, time = updateTimeNDate(filecontent[head][1])
            if len(vals) == 48:
                vals = []
                act_ymd = ymd
            while head < len(filecontent):
                ymd, time = updateTimeNDate(filecontent[head][1])
                vals.append(filecontent[head][2])
                head += 1
                if len(vals) == 48:
                    break
            if int(act_ymd[0]) == 2021 and int(act_ymd[1]) >=7 :
                break
            if len(vals) == 48:
                ret.append(act_ymd+vals)
    pd.DataFrame(ret).to_csv("QLD_Demand_2020.csv", index = False, header = False)
    
class poly_predictor:
    """
    poly_predictor is a class that calculates a polynomial that fits to the dataset data_sets


    """
    def __init__(self, data_sets, labels = None, report = False):
        """
        """
        self.data_sets = data_sets
        n_data = len(self.data_sets)   #number of sets
        l_data = len(self.data_sets[0])#size of the sets
        
        #Outlier removal (do later maybe)
        

        #Calculating the polynomial to fit
        
        sscurve = []
        
        for degree in range(1,10): #checks from polynomail degree 1-10
        
            error = []
        
            for robin in range(n_data): #calculate error using LOOCV
                validation_set = self.data_sets[robin]
                training_set = self.data_sets[:robin]+self.data_sets[robin+1:]
                traning_points = [mean(x) for x in [[training_set[sets][points] for sets in range(n_data-1)] for points in range(l_data)]]
                func = np.polyfit(range(l_data), traning_points, degree) 
                error.append(math.sqrt(sum([(validation_set[point]-poly_predictor.get_val(point, func))**2 for point in range(l_data)])))
                
            sscurve.append(mean(error))
            
        best_degree = 0
        degree = 1
            
        for x in range(len(sscurve)-1):
            if sscurve[x+1]-sscurve[x] <= best_degree:
                best_degree = sscurve[x+1]-sscurve[x]
                degree = x+1
                
        self.func = func
                
        if report:
            for sets in self.data_sets:
                plt.plot(range(len(sets)), sets, label = "set" + str(self.data_sets.index(sets)))
            plt.plot(range(l_data), [poly_predictor.get_val(x, func) for x in range(l_data)], label = "Polynomal of degree " + str(degree))
            plt.legend()
            plt.show()
            plt.figure()

    def get_val(x, func):
        ret = 0
        for t in range(len(func)):
            ret += func[t]*(x**(len(func)-t-1))
        return ret
    
    def predict(self, x):
        return poly_predictor.get_val(x, self.func)


class kernel_density:
    def __init__(self, data):
        self.data = [[x,] for x in data]
        self.model = KernelDensity()
        self.model.fit(self.data)
        
    def sample(self, n):
        return self.model.sample(n)

class year_model:
    def __init__(self, years):
        
        DAY_WIDTH = 2
        self.years = years
        self.daily_models = []
        all_days = [[] for x in range(366)]
        for year in years:
            count = 0
            for month in year:
                for day in month:
                    all_days[count].append(mean(day))
                    count += 1
        
        all_days = all_days[:-1]
        for x in range(len(all_days)):
            self.daily_models.append(kernel_density(list_sum([all_days[(x+t)%264] for t in range(-DAY_WIDTH, 1+DAY_WIDTH)])))
            
        
        self.daily_funcs = []
        for month in range(12):
            self.daily_funcs.append([])
            for day in range(len(years[1][month])):
                self.daily_funcs[len(self.daily_funcs)-1].append(poly_predictor([[x - mean(year[month][day]) for x in year[month][day]] for year in years]))
    
    def predict_day(self, day, month):
        """
        

        Parameters
        ----------
        day : actual day value
            DESCRIPTION.
        month : actual month value
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        day_of_year = day + sum((31,28,31,30,31,30,31,31,30,31,30,31)[:(month-7)%12])
        
        daily_mean = self.daily_models[day_of_year].sample(1)[0]
        
        if len(self.daily_funcs[month]) <= day: #deal with the leap year day
            day = len(self.daily_funcs[month]) - 1
        if day < 0:
            day = 0
            
        print(day, month)
        return [list(self.daily_funcs[month][day].predict(x) + daily_mean) for x in range(48)]
    
    def plot(self, leap = False):
        if leap:
            month_list = (31,29,31,30,31,30,31,31,30,31,30,31)
        else:
            month_list = (31,28,31,30,31,30,31,31,30,31,30,31)
        day_values = []
        for month in [(x+8)%12-1 for x in range(12)]:
            for day in range(month_list[month]):
                day_values.append(mean(self.predict_day(day, month, noise = True)))
        plt.scatter(range(sum(month_list)), day_values)
        plt.show()
        plt.figure()
        

if __name__ == "__main__":
    a = csvlist2list(FILES)
           
    #csvbad2csvgood() #Convert the files in new files into the complied qld2020 csv.
    
    Demo = True
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
        if False:
            for months in range(0,12):
                points = np.array(list_sum([[mean(day) for day in year[months]] for year in a]))
                sm.qqplot(points, line = "45", fit = True)
                py.title("qq-plot for daily averages for the month " + str((7+months-1)%12+1))
                py.show()
                plt.figure()
            
    #Time to do some calculations:
    
    #We'll start with deterining experimental the values for the normal dist for each month:
    
    monthly_params = []
    for year in a:
        for month in year:
            month_mean = mean([mean(day) for day in month])
            month_sd = math.sqrt(sum([(mean(day)-month_mean)**2 for day in month]))
            monthly_params.append((month_mean, month_sd))
            
    Demo_lots = False
            
    if Demo_lots:
        plt.title("monthly standard deviation in daily means")
        plt.xlabel("Month")
        plt.ylabel("SD")
        plt.plot(list(range(len(monthly_params))), [x[0] for x in monthly_params])
        plt.show()
        plt.figure()
        
        #Displaying the daily values overlayed, for each day
        
        for year in range(len(a)):
            for month in range(len(a[year])):
                for day in range(len(a[year][month])):
                    plt.title("Power Demand per day for all years, day" + str(day))
                    plt.xlabel("hour")
                    plt.ylabel("Power Demand")
                    for yr in range(len(a)):
                        dat = a[yr][month][day]
                        plt.plot([4+0.5*x for x in range(len(dat))], [x-mean(dat) for x in dat], label = str(2015+yr))
                    plt.legend()
                    plt.show()
                    plt.figure()
            

    Test = year_model(a) 
    
            
            