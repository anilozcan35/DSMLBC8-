x = 8

y = 3.2

z = 8j + 18

a = "Hello World"

b = True

c = 23 < 22

l = [1, 2, 3, 4]

d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}

t = {"Machine Learning", "Data Science"}

s = {"Python", "Machine Learning", "Data Science"}

type(s)

################################################################

text = "The goal is to turn data into information, and information into insight"

text.replace(",", " ")
text = text.upper()
text.split(" ")

####################################

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

len(lst)
lst[0]
lst[10]
lst[0:4]
lst = lst.pop(8)

lst.append("N")

lst.insert(8, "N")
dir(lst)

####################

dict = {"Christian": ["America", 18],
        "Daisy": ["England",12],
        "Antonio":["Spain",22],
        "Dante":["Italy",25]}

dict.keys()
dict.values()
dict["Daisy"][1] = 13

dict["Ahmet"] = ["Turkey", 24 ]

dict.pop("Antonio")

##############################

l = [2, 13, 18, 93, 22]

def func(liste):
    tekler = []
    ciftler = []
    for i in liste:
        if i % 2 == 0:
            ciftler.append(i)
        else:
            tekler.append(i)
    return tekler, ciftler

tekler, ciftler = func(l)


#############################################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns = df.columns


liste = []

for column in df.columns:
    if column.startswith("ins_"):
        liste.append(column)

liste

[col.lower() if "ins_" in col else col.upper() for col in df.columns]



["NUM_" + kolon for kolon in df.columns]

[kolon.upper() if "no" in kolon else kolon.upper() + "_FLAG" for kolon in df.columns ]

##################################

og_list = ["abbrev", "no_previous"]

new_cols = [kolon for kolon in df.columns if kolon not in og_list]

df_new = df[new_cols]

df_new.head()

#################

df = sns.load_dataset("diamonds")
df.head()

num_columns = [col for col in df._get_numeric_data()]

df["price"].dtype

agg_list = ["mean","min", "max", "sum"]

new_dict = {col : agg_list for col in num_columns}

df[num_columns].agg(new_dict)