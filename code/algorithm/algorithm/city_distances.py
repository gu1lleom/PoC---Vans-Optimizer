import pandas as pd

from code.data_extractor import FileTreatment


def load_tripulantes():
    f = FileTreatment()
    df = FileTreatment.readFile(f, filename='base2.xlsx', sheetname='base_oficial')
    df = df[["BP", "comuna", "lat", "lng"]]
    df["comuna"] = df["comuna"].str.lower()
    print(df.head())
    return df

def load_comunas():
    print("Cargando archivo maestro de comunas de santiago")
    f = FileTreatment()
    df_comunas = FileTreatment.read_csv_file(f, filename='latlon-chile.csv', sep=",")
    print(df_comunas.columns)
    df_comunas_santiago = df_comunas[df_comunas["Provincia"] == "Santiago"]
    df_comunas_santiago = df_comunas_santiago[["Comuna", "Latitud_(Decimal)", "Longitud_(decimal)", "Superficie_(km2)"]] 

    print(df_comunas_santiago.columns)
    df_comunas_santiago = df_comunas_santiago.rename(columns={'Comuna': 'comuna', 'Latitud_(Decimal)': 'lat_c', "Longitud_(decimal)": "lon_c", "Superficie_(km2)": "sup"})

    # Convertir todos los valores del DataFrame a minúsculas
    for columna in df_comunas_santiago.columns:
        if df_comunas_santiago[columna].dtype == 'object':  # Verificar si la columna es de tipo 'object' (normalmente, cadenas de texto)
            df_comunas_santiago[columna] = df_comunas_santiago[columna].str.lower()

    print("Cantidad de comunas cargadas por el maestro: ")
    print(df_comunas_santiago.shape[0])
    return df_comunas_santiago

df_tripulantes = load_tripulantes()
df_comunas = load_comunas()

df_fusionado = pd.merge(df_tripulantes, df_comunas[['comuna', 'lat_c', 'lon_c', 'sup']], on='comuna', how='left')
print(df_comunas)
print(df_fusionado.head())