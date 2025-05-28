import pandas as pd
df = pd.read_excel("podaci3.xlsx")
q1 = df["OutletSales"].quantile(0.25)
q3 = df["OutletSales"].quantile(0.75)

def kategorizuj(sales):
    if sales < q1:
        return "Low"
    elif sales > q3:
        return "High"
    else:
        return "Medium"


df['SalesCategory'] = df['OutletSales'].apply(kategorizuj)
df["Weight"] = df.groupby(["ProductType", "OutletType"])["Weight"].transform(lambda x: x.fillna(x.median()))
df["Weight"] = df["Weight"].fillna(df["Weight"].median())
x = df.drop(columns=["SalesCategory","OutletSales"])
y = df["SalesCategory"]

numericke_kolone = x.select_dtypes(include=["int64","float64"]).columns.tolist()
kategorske_kolone = x.select_dtypes(include=["object","category"]).columns.tolist()

x_kategorije = pd.get_dummies(x[kategorske_kolone])

x_final = pd.concat([x[numericke_kolone], x_kategorije], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_final)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final, y, test_size=0.2, stratify=y, random_state=42)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(random_state=42)
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=5)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(x_train, y_train)

y_pred =  model.predict(x_test)
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, output_dict=True)
print(pd.DataFrame(report).transpose().round(2))

