from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI()

merged_steam_user_data = pd.read_parquet(
    "FastAPI/merged_steam_user_data.parquet")
merged_steam_rev_data = pd.read_parquet(
    "FastAPI/merged_steam_rev_data.parquet")
merged_recommend_model = pd.read_parquet(
    "FastAPI/merged_recommend_model.parquet")

relevant_data = merged_recommend_model[(
    merged_recommend_model['recommend'] == True) & merged_recommend_model[
        'sentiment_analysis'].isin([0, 1, 2])]
user_item_matrix = relevant_data.pivot_table(
    index='user_id', columns='item_id', values='sentiment_analysis',
    fill_value=0)
user_similarity = cosine_similarity(user_item_matrix)


item_item_matrix = relevant_data.pivot_table(
    index='item_id', columns='app_name', values='sentiment_analysis',
    fill_value=0)
item_similarity = cosine_similarity(item_item_matrix.T)


@app.get("/UserForGenre")
def UserForGenre(genero: str = 'Action') -> dict:
    """ 
    Esta funcion recibe el genero de un juego y devuelve el usuario con 
    mas horas jugadas y la lista de horas.

    Args:
        genero (str, optional): Genero de juegos. Defaults to 'Action'.

    Returns:
        dict: Usuario: user_id + Lista {Año: YYYY, Horas: HHHH} 
    """
    df_filtered_by_genre = merged_steam_user_data[merged_steam_user_data[
        'genres'] == genero]
    user_hours = df_filtered_by_genre.groupby(
        'user_id')['playtime_forever'].sum()
    max_time_played_user = user_hours.idxmax()
    max_user_df = df_filtered_by_genre[df_filtered_by_genre['user_id']
                                       == max_time_played_user]
    hours_by_year = max_user_df.groupby(
        'release_year')['playtime_forever'].sum().reset_index()
    hours_years_list = [{'Año': year, 'Horas': hours} for year, hours in zip(
        hours_by_year['release_year'], hours_by_year['playtime_forever'])]

    result = {'Usuario con mas horas jugadas para genero {}'.format(
        genero): max_time_played_user, 'Horas jugadas': hours_years_list}

    return result


@app.get("/PlayTimeGenre")
def PlayTimeGenre(genero: str = 'Action') -> dict:
    """
    Esta funcion recibe el genero de un juego y devuelve el año con
    mas horas jugadas para dicho pedido.

    Args:
        genero (str, optional): Genero de juegos. Defaults to 'Action'.

    Returns:
        dict: {"Año con más horas jugadas para {genero}}" : AAAA} 

    """
    df_filtered_play_by_genre = merged_steam_user_data[merged_steam_user_data[
        'genres'] == genero]
    user_play_hours = df_filtered_play_by_genre.groupby(
        'release_year')['playtime_forever'].sum()
    max_time_play_user = int(user_play_hours.idxmax())

    resultado = {
        "Año de lanzamiento con más horas jugadas para género {}".format(
            genero): max_time_play_user}

    return resultado


@app.get("/UsersRecommend")
def UsersRecommend(year: int = 2017) -> dict:
    """
        Esta funcion recibe un año y devuelve top 3 de juegos mas 
        recomendados para el año dado
    Args:
        year (int, optional): Año. Defaults to 2017.

    Returns:
        dict: 
            [{"Puesto 1" : {app_name}}, 
            {"Puesto 2" : {app_name}},
            {"Puesto 3" : {app_name}}]
    """
    user_reviews_recommend = merged_steam_rev_data[merged_steam_rev_data[
        'release_year'] == year]
    top_games = user_reviews_recommend.groupby(
        'app_name')['recommend'].sum().nlargest(3)
    top_games_list = top_games.index.tolist()

    top_played_ranking = {f"Puesto {i + 1}": game for i,
                          game in enumerate(top_games_list)}

    return top_played_ranking


@app.get("/UsersNotRecommend")
def UsersNotRecommend(year: int = 2017):
    """
        Esta funcion recibe un año y devuelve top 3 de juegos menos 
        recomendados para el año dado

    Args:
        year (int, optional): Año. Defaults to 2017.

    Returns:
        dict: 
            [{"Puesto 1" : {app_name}}, 
            {"Puesto 2" : {app_name}},
            {"Puesto 3" : {app_name}}]
    """
    user_reviews_recommend = merged_steam_rev_data[merged_steam_rev_data[
        'release_year'] == year]
    least_recommended = user_reviews_recommend.groupby(
        'app_name')['recommend'].sum().nsmallest(3)
    least_recommended_list = least_recommended.index.tolist()
    least_recommended_ranking = {
        f"Puesto {i + 1}": game for i, game in enumerate(
            least_recommended_list)}

    return least_recommended_ranking


@app.get("/sentiment_analysis")
def sentiment_analysis(year: int = 2017) -> dict:
    """
    Esta funcion recibe un año y devuelve una lista con el nr de
    reseñas del análisis de sentimientos

    Args:
        year (int, optional): Año. Defaults to 2017.

    Returns:
        dict: {Negativo = XX, Neutral = XX, Positiv0 = XX}
    """
    reviews_by_year = merged_steam_rev_data[merged_steam_rev_data[
        'release_year'] == year]
    counts_by_sentiment = reviews_by_year['sentiment_analysis'].value_counts(
    ).to_dict()
    result = {
        'Negativo': counts_by_sentiment.get(0, 0),
        'Neutro': counts_by_sentiment.get(1, 0),
        'Positivo': counts_by_sentiment.get(2, 0)}

    return result


@app.get("/recomendacion_juego")
def recomendacion_juego(item_id: int) -> list:
    """
    Cuando se ingresa el item_id de un producto, la función devuelve una
    lista de juegos recomendados, similares al ingresado.

    Args:
        item_id (int): ID del producto.

    Returns:
        list:
            [{"Juego recomendado 1" : {app_name}}, 
            {"Juego recomendado 2" : {app_name}},
            ...]
    """
    item_id = item_item_matrix.index.get_loc(item_id)
    similar_indices = item_similarity[item_id].argsort()[::-1][1:6]
    similar_items = [int(item_item_matrix.index[index])
                     for index in similar_indices]

    game_names = [
        f"Juego recomendado {i+1}: {merged_recommend_model.loc[
            merged_recommend_model['item_id'] == item_id,
            'app_name'].values[0]}" for i, item_id in enumerate(similar_items)]

    return game_names


@app.get("/recomendacion_usuario")
def recomendacion_usuario(user_id: str) -> list:
    """
    Cuando se ingresa el id_usuario, la funcion devuelve una lista de 5 juegos
    recomendados para dicho usuario, similares a los gustos del mismo.

    Args:
        user_id (str): user_id

    Returns:
        list: 
            [{"Juego recomendado 1" : {app_name}}, 
            {"Juego recomendado 2" : {app_name}},
            ...]
    """
    user_row = user_item_matrix.loc[user_id].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_row, user_item_matrix.values)
    similar_users_indices = similarity_scores.argsort()[0][-6:-1]
    recommended_items = user_item_matrix.iloc[similar_users_indices].sum(
    ).sort_values(ascending=False).index.tolist()[:5]

    game_names = [
        f"Juego recomendado {
            i+1}: {merged_recommend_model.loc[merged_recommend_model['item_id'] == item_id, 'app_name'].values[0]}"
        for i, item_id in enumerate(recommended_items)
    ]

    return game_names
