import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv("Hafta_06/Odev/Telco_customer_churn/Telco-Customer-Churn.csv")
df = df_.copy()

df.head()


# CustomerId:--> Müşteri İd’si
# Gender:--> Cinsiyet
# SeniorCitizen:--> Müşterinin yaşlı olup olmadığı (1, 0)
# Partner:--> Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents:--> Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure:--> Müşterinin şirkette kaldığı ay sayısı
# PhoneService:--> Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines:--> Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService:--> Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity:--> Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup:--> Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection:--> Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport:--> Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV:--> Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies:--> Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract:--> Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling:--> Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod:--> Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges:--> Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges:--> Müşteriden tahsil edilen toplam tutar
# Churn:--> Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

#################################
# Görev 1 : Keşifçi Veri Analizi
#################################

# Adım 1: Genel resmi inceleyiniz.


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
# İlk göze çarpan Total_charge ın float olması gerektiği
# Na değer yok gözüküyor fakat veri içersinde Na yerine farklı değerler girilmiş o zaman
# CustomerId verisine şimdilik ihtiyacımız yok gibi gözüküyor onu da drop edelim

df.drop("customerID", axis=1, inplace=True)

df.describe().T

def uniq_vals(dataframe):
    for col in dataframe.columns:
        print(col,": ",dataframe[col].unique())
        print(col,": ", dataframe[col].nunique())
        print("#################################")

uniq_vals(df)


def type_changer(dataframe, col, type):
    print("Before: ",dataframe[col].dtype)
    dataframe[col] = dataframe[col].astype(type)
    print("Now: ", dataframe[col].dtypes)

type_changer(df, "TotalCharges", "float") # veride boşluk olduğu için çeviremedik.
                                          # ValueError: could not convert string to float: ''
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")

# bir daha type değiştirmeyi deneyelim

type_changer(df, "TotalCharges", "float")
"""
Before:  object # şimdi çalıştı
Now:  float64"""
df.dtypes

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)