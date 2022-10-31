import pandas as pd
import numpy as np
import os

from utils.aggregators import (media_0,var_0,iqr_range,count_anom_low,count_anom_top)


data_path = '../../datasets'
preprocess_path = '../../preprocess/version_1'
fill_na_num = -99999

# Taxpayer:

tax_payer = pd.read_csv(f'{data_path}/taxpayer.csv')
numeric_columns = tax_payer.select_dtypes(include=['number']).columns
for col in numeric_columns:
    tax_payer[col] = np.where(tax_payer[col]<0,0,tax_payer[col])


# Legal Representatives:

legal_representatives = pd.read_csv(f'{data_path}/legal_representatives.csv')
legal_representatives['bureau_risk'] = np.where(legal_representatives['bureau_risk'].isna(),'no_grupo',
                                                legal_representatives['bureau_risk'])
numeric_columns = legal_representatives.select_dtypes(include=['number']).columns

for col in numeric_columns:
    legal_representatives[col] = np.where(legal_representatives[col]<0,0,legal_representatives[col])

legal_representatives[numeric_columns] = legal_representatives[numeric_columns].fillna(fill_na_num)


# Transactionality:

transactionality = pd.read_csv(f'{data_path}/transactionality.csv')


# Acquisition

acquisition = pd.read_csv(f'{data_path}/acquisition.csv')


# Movements

movements = pd.read_csv(f'{data_path}/movements.csv')

################################# TRAIN ###########################################

## Universe:

universe_train = pd.read_csv(f'{data_path}/universe_train.csv') 

# Train join:

## tax payer
universe_train = universe_train.merge(tax_payer,how='left',on=['ID','period'])

## legal representatives
universe_train = universe_train.merge(legal_representatives,how='left',on=['ID','period'])

## transactionality
universe_train = universe_train.merge(transactionality,how='left',on=['ID','period'])
columns = [col for col in list(transactionality.columns) if col not in ['ID','period']]
universe_train[columns] = universe_train[columns].fillna(fill_na_num)

## acquisition
universe_train = universe_train.merge(acquisition,how='left',on=['ID','period'])
columns = [col for col in list(acquisition.columns) if col not in ['ID','period']]
universe_train[columns] = universe_train[columns].fillna(fill_na_num)

## movements
universe_train = universe_train.merge(movements,how='left',on=['ID','period'])
columns = [col for col in list(movements.columns) if col not in ['ID','period']]
universe_train[columns] = universe_train[columns].fillna(fill_na_num)


# Balance Train

balances_train = pd.read_csv(f'{data_path}/balances_train.csv')

balances_train_month = balances_train.groupby(by=['ID','period']).agg(
                                            count_month=pd.NamedAgg(column='month', 
                                                                    aggfunc= lambda x: x.nunique() - 12),
                                            count_product=pd.NamedAgg(column='product', 
                                                                    aggfunc= lambda x: x.nunique()),
                                            count_entity=pd.NamedAgg(column='entity', 
                                                                    aggfunc= lambda x: x.nunique())).reset_index()

balances_train = balances_train.groupby(by=['ID','period','type']).agg(
                                        balance_sum=pd.NamedAgg(column='balance_amount', aggfunc='sum'),
                                        balance_mean=pd.NamedAgg(column='balance_amount', aggfunc='mean'),
                                        balance_var=pd.NamedAgg(column='balance_amount', aggfunc='var'),
                                        balance_iqr=pd.NamedAgg(column='balance_amount', aggfunc=iqr_range),
                                        balance_anom_low=pd.NamedAgg(column='balance_amount', aggfunc=count_anom_low),
                                        balance_anom_top=pd.NamedAgg(column='balance_amount', aggfunc=count_anom_top),
                                        days_default=pd.NamedAgg(column='days_default', aggfunc='sum'),
                                        days_default_mean=pd.NamedAgg(column='days_default', aggfunc=media_0),
                                        days_default_var=pd.NamedAgg(column='days_default', aggfunc=var_0),
                                        ).reset_index()\
                                        .pivot(index=['ID','period'], columns=['type']).reset_index()

balances_train.columns = ['_'.join(col).strip() for col in balances_train.columns.values]
balances_train.rename(columns={'ID_':'ID','period_':'period'},inplace=True)
universe_train = universe_train.merge(balances_train,how='left',on=['ID','period'])
universe_train = universe_train.merge(balances_train_month,how='left',on=['ID','period'])
universe_train.fillna(-99999,inplace=True)

#SAVE FILE:
universe_train.to_parquet(f'{preprocess_path}/train.parquet')

print('TRAIN GENERADO')

################################# TEST ###########################################

## Universe:

universe_test = pd.read_csv(f'{data_path}/universe_test.csv')

# Test join:

## tax payer
universe_test = universe_test.merge(tax_payer,how='left',on=['ID','period'])

## legal representatives
universe_test = universe_test.merge(legal_representatives,how='left',on=['ID','period'])

## transactionality
universe_test = universe_test.merge(transactionality,how='left',on=['ID','period'])
columns = [col for col in list(transactionality.columns) if col not in ['ID','period']]
universe_test[columns] = universe_test[columns].fillna(fill_na_num)

## acquisition
universe_test = universe_test.merge(acquisition,how='left',on=['ID','period'])
columns = [col for col in list(acquisition.columns) if col not in ['ID','period']]
universe_test[columns] = universe_test[columns].fillna(fill_na_num)

## movements
universe_test = universe_test.merge(movements,how='left',on=['ID','period'])
columns = [col for col in list(movements.columns) if col not in ['ID','period']]
universe_test[columns] = universe_test[columns].fillna(fill_na_num)


# Balance Test

balances_test = pd.read_csv(f'{data_path}/balances_test.csv')

balances_test_month = balances_test.groupby(by=['ID','period']).agg(
                                            count_month=pd.NamedAgg(column='month', 
                                                                    aggfunc= lambda x: x.nunique() - 12),
                                            count_product=pd.NamedAgg(column='product', 
                                                                    aggfunc= lambda x: x.nunique()),
                                            count_entity=pd.NamedAgg(column='entity', 
                                                                    aggfunc= lambda x: x.nunique())).reset_index()
balances_test = balances_test.groupby(by=['ID','period','type']).agg(
                                        balance_sum=pd.NamedAgg(column='balance_amount', aggfunc='sum'),
                                        balance_mean=pd.NamedAgg(column='balance_amount', aggfunc='mean'),
                                        balance_var=pd.NamedAgg(column='balance_amount', aggfunc='var'),
                                        balance_iqr=pd.NamedAgg(column='balance_amount', aggfunc=iqr_range),
                                        balance_anom_low=pd.NamedAgg(column='balance_amount', aggfunc=count_anom_low),
                                        balance_anom_top=pd.NamedAgg(column='balance_amount', aggfunc=count_anom_top),
                                        days_default=pd.NamedAgg(column='days_default', aggfunc='sum'),
                                        days_default_mean=pd.NamedAgg(column='days_default', aggfunc=media_0),
                                        days_default_var=pd.NamedAgg(column='days_default', aggfunc=var_0),
                                        ).reset_index()\
                                        .pivot(index=['ID','period'], columns=['type']).reset_index()

balances_test.columns = ['_'.join(col).strip() for col in balances_test.columns.values]
balances_test.rename(columns={'ID_':'ID','period_':'period'},inplace=True)
universe_test = universe_test.merge(balances_test,how='left',on=['ID','period'])
universe_test = universe_test.merge(balances_test_month,how='left',on=['ID','period'])
universe_test.fillna(-99999,inplace=True)

#SAVE FILE:
universe_test.to_parquet(f'{preprocess_path}/test.parquet')

print('TEST GENERADO')