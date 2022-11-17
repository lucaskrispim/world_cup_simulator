# -*- coding: utf-8 -*-
"""world_cup2.ipynb
"""

import numpy as np
import pandas as pd
import datetime
import random

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest


df = pd.read_csv('international_matches.csv')
df.tail()

df['ewm_home_score'] = df.groupby('home_team')['home_team_score'].transform(lambda x: x.rolling(15,closed='left').mean())
df['ewm_away_score'] = df.groupby('away_team')['away_team_score'].transform(lambda x: x.rolling(15,closed='left').mean())

df = df[(df['ewm_home_score'].notna()) & (df['ewm_away_score'].notna())]

df['date'] = pd.to_datetime(df['date'])

# feature generation
df['rank_difference'] = df['home_team_fifa_rank'] - df['away_team_fifa_rank']
df['average_rank'] = (df['home_team_fifa_rank'] + df['away_team_fifa_rank'])/2
df['point_difference'] = df['home_team_total_fifa_points'] - df['away_team_total_fifa_points']
df['score_difference'] = df['home_team_score'] - df['away_team_score']
    
df['is_won'] =  df['home_team_result']

df['is_stake'] = df['tournament'] != 'Friendly'

X, y = df.loc[:,['average_rank', 'rank_difference', 'point_difference','ewm_home_score','ewm_away_score','is_stake']], df['is_won']

columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


features_list = list(columns)
k_best_features = SelectKBest(k='all')
k_best_features.fit_transform(X, y)
k_best_features_scores = k_best_features.scores_
raw_pairs = zip(features_list, k_best_features_scores)
ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))
k_best_features_final = dict(ordered_pairs)
best_features = k_best_features_final.keys()
print ('')
print ("Melhores features:")
print (k_best_features_final)
print ('')

# Random florest

logreg = RandomForestClassifier(bootstrap = False, max_depth = 200, 
                                    max_features = 'sqrt', min_samples_leaf = 3,
                                    min_samples_split = 3, n_estimators = 200)
features = PolynomialFeatures(degree=3)


standscaler = StandardScaler()

model = Pipeline([('scale', standscaler), ('random_florest', logreg)])


model = model.fit(X_train, y_train)
print ('Random florest')
print(classification_report(y_test,model.predict(X_test)))
print ('')


# MLPClassifier

logreg = MLPClassifier(alpha=1, max_iter=1000)
features = PolynomialFeatures(degree=3)

standscaler = StandardScaler()

model1 = Pipeline([('scale', standscaler), ('random_florest', logreg)])

model1 = model1.fit(X_train, y_train)
print ('MLPClassifier')
print(classification_report(y_test,model1.predict(X_test)))

# KNeighborsClassifier

logreg = KNeighborsClassifier(n_neighbors=3)

model2 = Pipeline([('scale', standscaler), ('random_florest', logreg)])

model2 = model2.fit(X_train, y_train)
print ('KNeighborsClassifier')
print(classification_report(y_test,model2.predict(X_test)))

# Logistic Regression

logreg = linear_model.LogisticRegression(C=1e-5, max_iter=10000)
features = PolynomialFeatures(degree=2)
model3 = Pipeline([
    ('polynomial_features', features),
    ('logistic_regression', logreg)
])
model3 = model3.fit(X_train, y_train)
print ('Logistic Regression')
print(classification_report(y_test,model3.predict(X_test)))

#List of all Teams in 2022 World Cup
teams_worldcup = ['Qatar', 'Ecuador', 'Senegal', 'Netherlands', 'England', 'IR Iran', 'USA',
                  'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland', 'France', 
                  'Australia', 'Denmark', 'Tunisia', 'Spain', 'Costa Rica', 'Germany', 
                  'Japan', 'Belgium', 'Canada', 'Morocco', 'Croatia', 'Brazil', 'Serbia', 
                  'Switzerland', 'Cameroon', 'Portugal', 'Ghana', 'Uruguay', 'Korea Republic']

world_cup_rankings_home = df[['home_team','home_team_fifa_rank','home_team_total_fifa_points','ewm_home_score']].loc[df['home_team'].isin(teams_worldcup) & (df['date']>'2021-01-01')]
world_cup_rankings_away = df[['away_team','away_team_fifa_rank','away_team_total_fifa_points','ewm_away_score']].loc[df['away_team'].isin(teams_worldcup) & (df['date']>'2021-01-01')]
world_cup_rankings_home = world_cup_rankings_home.set_index(['home_team'])

#The idea is to separete the performance of each Team as Home or Away.
world_cup_rankings_home = world_cup_rankings_home.groupby('home_team').mean()
world_cup_rankings_away = world_cup_rankings_away.groupby('away_team').mean()

# Creating a number list
num_lst = ['draw','loss','win']

matches = [                
# group A
{'home':'Qatar','away':'Ecuador','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Qatar','away':'Senegal','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Netherlands','away':'Qatar','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Senegal','away':'Netherlands','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Netherlands','away':'Ecuador','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Ecuador','away':'Senegal','home_w':0,'away_w':0 ,'draw':0 },
# group B
{'home':'England','away':'IR Iran','home_w':0,'away_w':0 ,'draw':0 },
{'home':'England','away':'USA','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Wales','away':'England','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Wales','away':'IR Iran','home_w':0,'away_w':0 ,'draw':0 },
{'home':'IR Iran','away':'USA','home_w':0,'away_w':0 ,'draw':0 },
{'home':'USA','away':'Wales','home_w':0,'away_w':0 ,'draw':0 },
# group C
{'home':'Argentina','away':'Saudi Arabia','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Argentina','away':'Mexico','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Poland','away':'Argentina','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Mexico','away':'Poland','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Poland','away':'Saudi Arabia','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Saudi Arabia','away':'Mexico','home_w':0,'away_w':0 ,'draw':0 },
# group D
{'home':'France','away':'Australia','home_w':0,'away_w':0 ,'draw':0 },
{'home':'France','away':'Denmark','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Tunisia','away':'France','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Tunisia','away':'Australia','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Australia','away':'Denmark','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Denmark','away':'Tunisia','home_w':0,'away_w':0 ,'draw':0 },
# group E
{'home':'Spain','away':'Costa Rica','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Spain','away':'Germany','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Japan','away':'Spain','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Japan','away':'Costa Rica','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Costa Rica','away':'Germany','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Germany','away':'Japan','home_w':0,'away_w':0 ,'draw':0 },
# group F
{'home':'Belgium','away':'Canada','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Belgium','away':'Morocco','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Croatia','away':'Belgium','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Croatia','away':'Canada','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Canada','away':'Morocco','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Morocco','away':'Croatia','home_w':0,'away_w':0 ,'draw':0 },
# group G
{'home':'Brazil','away':'Serbia','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Brazil','away':'Switzerland','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Cameroon','away':'Brazil','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Cameroon','away':'Serbia','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Serbia','away':'Switzerland','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Switzerland','away':'Cameroon','home_w':0,'away_w':0 ,'draw':0 },
# group H
{'home':'Uruguay','away':'Korea Republic','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Portugal','away':'Ghana','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Korea Republic','away':'Ghana','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Portugal','away':'Uruguay','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Ghana','away':'Uruguay','home_w':0,'away_w':0 ,'draw':0 },
{'home':'Korea Republic','away':'Portugal','home_w':0,'away_w':0 ,'draw':0 },
]
    
iterations = 1000  
    
for j in range(0,len(matches)):    
  home = matches[j]['home']
  away = matches[j]['away']
  
  row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, True]]), columns=columns)

  home_rank = world_cup_rankings_home.loc[home, 'home_team_fifa_rank']
  home_points = world_cup_rankings_home.loc[home, 'home_team_total_fifa_points']
  
  opp_rank = world_cup_rankings_away.loc[away, 'away_team_fifa_rank']
  opp_points = world_cup_rankings_away.loc[away, 'away_team_total_fifa_points']
  
  row['average_rank'] = (home_rank + opp_rank) / 2
  row['rank_difference'] = home_rank - opp_rank
  row['point_difference'] = home_points - opp_points
  row['ewm_home_score'] = world_cup_rankings_home.loc[home, 'ewm_home_score']
  row['ewm_away_score'] = world_cup_rankings_away.loc[away, 'ewm_away_score']
  row['is_stake'] = True

  prob = random.choices(num_lst, weights=model.predict_proba(row)[:][0], k=iterations)

  for p in prob:
    if p == 'draw':
      matches[j]['draw'] = matches[j]['draw'] +1
    elif p == 'loss':
      matches[j]['away_w'] = matches[j]['away_w'] +1
    elif p == 'win':
      matches[j]['home_w'] = matches[j]['home_w'] +1

for j in range(0,len(matches)-1):
  print(f"{matches[j]['home']} vs {matches[j]['away']} home win: { round( 100.0*(matches[j]['home_w']/iterations),2) }, Away win:{round( 100.0*(matches[j]['away_w']/iterations),2)}, Draw:{round( 100.0*(matches[j]['draw']/iterations),2)}  ")
    