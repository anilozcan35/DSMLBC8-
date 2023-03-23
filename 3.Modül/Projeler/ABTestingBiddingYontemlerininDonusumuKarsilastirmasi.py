import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind


cdf = pd.read_excel("3.Modül/Projeler/ab_testing.xlsx", sheet_name= "Control Group")
tdf = pd.read_excel("3.Modül/Projeler/ab_testing.xlsx", sheet_name= "Test Group")

cdf = cdf.iloc[:, 0:4]
cdf.head()
cdf.isnull().sum()
cdf.describe().T

tdf = tdf.iloc[:, 0:4]
tdf.head()
tdf.isnull().sum()
tdf.describe().T

udf = pd.concat([cdf, tdf], keys= ["control", "test"])

udf.loc["control","Purchase"].mean()
udf.loc["test","Purchase"].mean()

########################
# H0: M1 = M2 (kontrol ve test gruplarında purchaseler arasında anlamlı bir farklılık yoktur)
# H1: M1 != M2 (.. vardır)

# Normallik Varsayımı
# H0: Normal dağılım varsaayımı sağlanmaktadır.
# H1: .. sağlanmamaktadır.

test_stats, p_value = shapiro(udf.loc["control","Purchase"])
print(f"test stats {test_stats}, pvalue: {p_value}") # --> p > 0.05 H0 kabul

test_stats, p_value = shapiro(udf.loc["test","Purchase"])
print(f"test stats {test_stats}, pvalue: {p_value}") # --> p > 0.05 H0 kabul

# iki sample da normal dağılıyor.

# Varyans Homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(udf.loc["control","Purchase"],udf.loc["control","Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # --> p-value > 0.05 H0 kabul edilir.

# Normallik testi ve varyans homojenliği de sağlandığına göre parametrik test yapılır.
# bağımsız iki örneklem t testi
test_stat, pvalue = ttest_ind(udf.loc["control","Purchase"],udf.loc["control","Purchase"],equal_var= True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # pvalue > 0.05 H0 hipotezi red edilemez.

# bağımsız iki örneklem t testini kullanmamızın gereği normallik ve varyans homojenliğini sağlamasıdır. Bu sebepten dolayı
# parametrik test ile ilerledik.

# Yapılan değişiklik purchase üzerinde yüzde 95 güven aralığında anlamlı bir farklılığa sebep olmuştur.








