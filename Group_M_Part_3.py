import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def value_calc(df):
    # normalize ratings
    df["points"] = df["points"] - df["points"].mean()
    # take log of price
    X = pd.DataFrame(np.log(df['price']))
    y = pd.DataFrame(df['points'])
    # create regression
    reg = LinearRegression().fit(X, y)
    # set value equal to residuals
    df['value'] = y - reg.predict(X)
    return df

def factor_filter(df,target_factor):
    # drop rows with nan for target factor role
    df.dropna(subset=[target_factor])
    # create conditional if it is not equal to description 
    if target_factor != 'description':
        # group by factor and count number of entries
        most_common = df.groupby(target_factor).price.count().sort_values(ascending=False)
        # if fewer than 20 rows, use all
        if len(most_common) < 20:
            # get inndex
            most_common = most_common.index
        # otherwise use all rows greater than 10
        else:
            most_common = most_common[most_common > 10].index
        # get 10 top factors
        if len(most_common) > 10:
            most_common = most_common[0:10]
        # filter by factors included in the most common
        df = df[df[target_factor].isin(most_common)] 
    return df

def desc_predictors(df):
    # split words
    text_splt = " ".join(df["description"])
    # tokenize words
    allWords = nltk.tokenize.word_tokenize(text_splt)
    # create extra stop words
    extra_stop = ['%',"'s",';','a','also','cabernet','drink','flavors','good','great','it','like',
     'little','made','merlot','mouth',"n't",'notes','offers','one','palate','pinot',
     'sauvignon','shows','slightly','the','there','this','well','wine','years',',','.','the','it',
     'this','red','delicious',')','(']
    # get stop words
    stpword = nltk.corpus.stopwords.words('english')
    # merge stop words list
    stpword = stpword + extra_stop
    # get words that are not stop words
    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w.lower() not in stpword)
    # get top 100 words
    mostCommon = allWordExceptStopDist.most_common(100)
    # change format
    mostCommon = list(list(zip(*mostCommon))[0])
    # xreate dataframe
    flavor_profile = pd.DataFrame()
    # replace ( & ) symbols, since they are not words and create issues
    df["description"] = df["description"].str.replace('\(', '')
    df["description"] = df["description"].str.replace('\)', '')
    # create matrix for each wine to see if the description uses that word
    for flavor in mostCommon:
        flavor_profile[flavor] = df["description"].str.contains(flavor)
    # set matrix as x
    X = pd.DataFrame(flavor_profile)
    # transform to binary
    X = X.replace({True:1,False:0})
    # turn value into y variable
    y = pd.DataFrame(df.value)
    # train model
    model = sm.OLS(y, X).fit()
    # return the model parameters for each factor
    return pd.Series(model.params)


def wine_values(df, target_factor):
    # filter values
    df = factor_filter(df,target_factor)
    # calculate value
    df = value_calc(df)
    # conditional if factor is description
    if target_factor == 'description':
        # use description predictor
        factor_rankings = desc_predictors(df)
    else:
        # use target factor rankings
        factor_rankings = df.groupby(target_factor).value.mean().sort_values()
        # plot rankings
        plt.rcParams['axes.edgecolor']='#333F4B'
        plt.rcParams['axes.linewidth']=0.8
        plt.rcParams['xtick.color']='#333F4B'
        plt.rcParams['ytick.color']='#333F4B'
        plt.figure(figsize=(10,7.5))
        plt.bar(factor_rankings.index,factor_rankings.values)
        plt.xlabel(target_factor)
        plt.ylabel("Value")
        plt.title('Points above expected for the price')
        plt.show()
    # get top factors
    top_factors = pd.DataFrame(factor_rankings[factor_rankings > 0].sort_values(ascending = False), columns = ['value'])
    # get bottom factors
    bottom_factors = pd.DataFrame(factor_rankings[factor_rankings <= 0].sort_values(ascending = True), columns = ['value'])
    # print top and bottom
    print("The biggest factors in good value and their average points above the price are:\n")
    print(top_factors.head(5))
    print("\nThe biggest factors in below average value and their average points below the price are:\n")
    print(bottom_factors.head(5))
    # create option to continue
    to_continue = input("Would you like to filter further? (Y for yes, all else is no)\n")
    to_continue = to_continue.lower()
    # conditionalif the user wants to continue
    if to_continue == "y":
        while True:
            # get filter to continue
            drill_filter = input('Which ' + target_factor + ' would you like to filter?\n')
            if drill_filter in factor_rankings.index:
                df = df[df[target_factor] == drill_filter]
                break
            else:
                # if not a valid factor, retry it
                print("\nI'm sorry. " + drill_filter + " is not a valid "+ target_factor + ".\n")
                print("Please print the " + target_factor + " exactly as displayed in the list below\n")
                for i in range(len(factor_rankings.index.values)):
                    print(factor_rankings.index[i])
        return df
    # end program
    else:
        print("Enjoy your wine!")
        return 0

def main(factors,df):
    # create copies of the variables
    working_factors = factors.copy()
    working_df = df.copy()
    # try for error handling for small values
    try:
        # repeat while it is running
        while True:
            # pritn message
            print("Here are the potential factors:\n" + ', '.join(working_factors))
            # get target factor
            target_factor  = input("\nPlease select one:\n")
            if target_factor not in working_factors:
                print('This factor was not found')
                return "This factor was not found"
            # remove target factor
            working_factors.remove(target_factor)
            # creat temp variable
            hold = wine_values(working_df, target_factor)
            # test if it is a dataframe or 0
            if isinstance(hold, pd.DataFrame):
                # set working df equal to the filtered df
                working_df = hold
                print('\n')
            else:
                break
    except ValueError:
        # error handle statement
        print("Not enough remaining wines at this level of detail")
    return 0

# set wd
os.chdir('C:\\Users\\kelennon\\Desktop\MSDS\\Summer 2019\\Wine Project')
# turn of warning
pd.options.mode.chained_assignment = None  # default='warn'
# import data
df = pd.read_csv("winemag-data-130k-v2.csv")
# filter data
df = df.drop(['Unnamed: 0', 'designation',"taster_name","taster_twitter_handle"],1)
# select factors
df = df.dropna(subset=['country', 'description', 'points', 'price', 'province', 'variety'])
# select factors
factors = ['country', 'description', 'province', 'region_1', 'variety', 'winery']



'''
# print welcome statement
print("Welcome!\nThis program helps you find a good wine for the price!")
# run program
main(factors,df)
'''
