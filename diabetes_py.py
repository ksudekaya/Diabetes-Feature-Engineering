
#DIABETES FEATURE ENGINEERING

#Veri Seti Hikayesi

#Veriseti ABD'deki Ulusal Diyabet-Sindirim-BöbrekHastalıklarıEnstitüleri'ndetutulanbüyükverisetininparçasıdır.ABD'deki
# ArizonaEyaleti'ninenbüyük 5.şehriolanPhoenixşehrindeyaşayan21yaşveüzerindeolanPimaIndiankadınlarıüzerinde
# yapılandiyabetaraştırmasıiçinkullanılanverilerdir.# Hedef değişken "outcome" olarak belirtilmiş olup; 1diyabet test sonucunun pozitifoluşunu, 0ise negatifoluşunu belirtmektedir

# 9 Değişken- 768 Gözlem
#Değişkenler:
#Hamilelik Sayısı (Pregnancies): Hamilelik sayısı
#Glikoz (Glucose): Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
#Kan Basıncı (Blood Pressure): Küçük tansiyon (mm Hg)
#Cilt Kalınlığı (Skin Thickness): Cilt kalınlığı (mm)
#İnsülin (Insulin): 2 saatlik serum insülini (mu U/ml)
#Diyabet Pedigri Fonksiyonu (Diabetes Pedigree Function): Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
#Vücut Kitle İndeksi (BMI): Vücut kitle indeksi (kg/m²)
#Yaş (Age): Yaş (yıl)
#Sonuç (Outcome): Hastalığa sahip (1) ya da değil (0)

#KEŞİFÇİ VERİ ANALİZİ

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
!pip install missingno
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

def load_application_train():
    data = pd.read_csv("diabetes.csv")
    return data

df = load_application_train()
df.head()

#Genel Resim

df.head()
print("#######################")
df.tail()
print("#######################")
df.info()
print("#######################")
df.describe()
print("#######################")
df.shape
print("#######################")
df.columns
print("#######################")
df.isnull().sum()
print("#######################")
df.dtypes
print("#######################")
df['Outcome'].value_counts()

#Numerik ve Kategorik Değişken Bulma ve Analizi
dff = load_application_train()
dff.head()
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi#ticket

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

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
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

#Target Analizi
def target_summary_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
 for col in num_cols:
        target_summary_num(df, "Outcome", col)


#Aykırı Gözlem Analizi

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Insulin")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Insulin")

for col in num_cols:
    print(col, check_outlier(df, col))

def box_plot_analyses(df, col):
    sns.boxplot(df[col])
    plt.show()

for col in num_cols:
    print(col)
    box_plot_analyses(df,col)

#Eksik Gözlem

df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])#ratio virgülden sonra ayarlama-sütüna göre ayarlama yapılmışş
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
missing_values_table(df, True)


#Korelasyon Matrixi

msno.matrix(df)
plt.show()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

#FEATURE ENGINEERING

#Eksik Değer Analizi

selected=["Glucose","SkinThickness","Insulin","BMI","BloodPressure"]
for col in selected:
    df[col] = df[col].apply(lambda x: np.nan if x == 0 else x)
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])#ratio virgülden sonra ayarlama-sütüna göre ayarlama yapılmışş
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)

df.groupby("Outcome")["BMI"].mean()

for col in selected:
    df[col]=df[col].fillna(df.groupby("Outcome")[col].transform("mean"))

df.isnull().sum()


def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

df.columns= [col.upper() for col in df.columns]
df.columns

#Feature Extraction
df["AGE_NEW"]= pd.cut(df["AGE"],bins= [20,45,max(df["AGE"])], labels=["mature","senior"])
df["GLUCOSE_NEW"]= pd.cut(df["GLUCOSE"], bins=[0, 100, 140 , max(df["GLUCOSE"])], labels=["low","normal","high"])
df["BMI_NEW"]=pd.cut(df["BMI"], bins=[18,25,32,max(df["BMI"])], labels=["Normal Weight","Overweight","Obese"])
df.loc[df["INSULIN"]<=125,"INSULIN_NEW"]="normal"
df.loc[df["INSULIN"]>125, "INSULIN_NEW"]="anormal"

df["GLUCOSE_INSULIN"]=df["GLUCOSE"]*df["INSULIN"]
df["INSULIN_BMI"]=df["INSULIN"]*df["BMI"]
df["GLUCOSE_BLOODPRESSURE"]= df["GLUCOSE"]* df["BLOODPRESSURE"]
df["INSULIN_BLOODPRESSURE"]= df["INSULIN"]*df["BLOODPRESSURE"]

df.loc[(df["AGE_NEW"]=="mature") & (df["GLUCOSE_NEW"]=="low"),"AGE_GLUCOSE"]="maturelow"
df.loc[(df["AGE_NEW"]=="mature") & (df["GLUCOSE_NEW"]=="normal"),"AGE_GLUCOSE"]="maturenormal"
df.loc[(df["AGE_NEW"]=="mature") & (df["GLUCOSE_NEW"]=="high"),"AGE_GLUCOSE"]="maturehigh"

df.loc[(df["AGE_NEW"]=="senior") & (df["GLUCOSE_NEW"]=="low"),"AGE_GLUCOSE"]="seniorlow"
df.loc[(df["AGE_NEW"]=="senior") & (df["GLUCOSE_NEW"]=="normal"),"AGE_GLUCOSE"]="seniornormal"
df.loc[(df["AGE_NEW"]=="senior") & (df["GLUCOSE_NEW"]=="high"),"AGE_GLUCOSE"]="seniorhigh"

df.loc[(df["AGE_NEW"]=="mature") & (df["BMI_NEW"]=="Normal Weight"),"AGE_BMI"]="matureNormalWeight"
df.loc[(df["AGE_NEW"]=="mature") & (df["BMI_NEW"]=="Overweight"),"AGE_BMI"]="matureOverweight"
df.loc[(df["AGE_NEW"]=="mature") & (df["BMI_NEW"]=="Obese"),"AGE_BMI"]="matureObese"

df.loc[(df["AGE_NEW"]=="senior") & (df["BMI_NEW"]=="Normal Weight"),"AGE_BMI"]="seniorNormalWeight"
df.loc[(df["AGE_NEW"]=="senior") & (df["BMI_NEW"]=="Overweight"),"AGE_BMI"]="seniorOverweight"
df.loc[(df["AGE_NEW"]=="senior") & (df["BMI_NEW"]=="Obese"),"AGE_BMI"]="seniorObese"

df.head()
df.shape

#Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
cat_cols=[ col for col in cat_cols if col != "OUTCOME"]
#Label Encoding
def label_encoder(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

#One-Hot Encoding
#cat-cols güncelleme
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols = [col for col in cat_cols if col not in binary_cols]

df = one_hot_encoder(df, cat_cols, drop_first=True)

#Standarlaştırma
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df[num_cols].head()

#Modelleme
y = df["OUTCOME"]#bağımlı
X = df.drop(["OUTCOME"], axis=1)#dışındaki

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)