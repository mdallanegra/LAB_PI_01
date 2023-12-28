import numpy as np

user_ids = user_item_v_playtime['user_id'].astype(
    str)  # Convertir a string para concatenar
item_ids = user_item_v_playtime['item_name'].astype(
    str)  # Convertir a string para concatenar

# Combinar 'user_id' y 'item_id' en una lista de strings
item_names = 'Usuario: ' + user_ids + ' \n ' + 'Juego: ' + item_ids
playtime_hours = user_item_v_playtime['playtime_forever']

mask = [True if 'Usuario' in name else False for name in item_names]

# Asignar colores basados en la máscara
colors = np.where(mask, 'red', 'skyblue')

plt.figure(figsize=(10, 6))
plt.barh(item_names, playtime_hours, color=colors)

plt.title('Top 10 Mayores Jugadores x Juego')
plt.xlabel('Tiempo de Juego')
plt.ylabel('Usuario')

plt.gca().invert_yaxis()
plt.tight_layout()

plt.show()
