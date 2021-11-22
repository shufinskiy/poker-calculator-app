import time

import numpy as np
import streamlit as st
import pokercalculator as pc

hash_card = pc.download_dict_hash_cards("card_dict.dat")

cards = np.array([hash_card for hash_card in hash_card.values()])

np_key = pc.load_numpy_array("np_key.npy")
np_value = pc.load_numpy_array("np_value.npy")

power_comb = pc.create_dict_power_combinations(np_key, np_value)

player1_cards = st.text_input(label="Введите карты первого игрока", key="pl1")
player2_cards = st.text_input(label="Введите карты второго игрока", key="pl2")
board_cards = st.text_input(label="Введите карты на столе", key="board")

nsamples = st.slider("Количество прогонов", min_value=50000, max_value=1000000, value=500000, step=25000)

check = st.checkbox("Дополнительные данные", False)

if st.button("Начать расчёт"):
    if not board_cards:
        board_cards = ","

    if player1_cards and player2_cards and board_cards:
        s = time.time()

        pl1 = tuple((card.strip() for card in player1_cards.split(',')))
        pl2 = tuple((card.strip() for card in player2_cards.split(',')))
        brd = tuple((card.strip() for card in board_cards.split(',')))

        pl1 = pc.parsing_of_player_cards(pl1)
        pl2 = pc.parsing_of_player_cards(pl2)

        pl1 = pc.convert_player_card_to_int(pl1, hash_card)
        pl2 = pc.convert_player_card_to_int(pl2, hash_card)

        players_card = pc.available_comb_players_cards(pl1, pl2)

        board = pc.convert_board_card_to_int(brd, hash_card)

        players_card, board = pc.available_combinations(players_card, board)

        if board.shape[0]:
            board = cards[np.isin(cards, board)]
            cards = cards[~np.isin(cards, board)]
        else:
            board = np.array([], dtype=np.int64)

        result = pc.generate_plays(cards, players_card, power_comb, nsamples, board)
        calc_win = result[:, 0] - result[:, 1]

        keys, cnt = np.unique(calc_win, return_counts=True)

        result_dict = {key: round(value/nsamples*100, 2) for (key, value) in zip(keys, cnt)}

        f = time.time()
        st.write("Процент выигрышей первого игрока:", result_dict.get(-1, 0.00))
        st.write("Процент дележки банка:", result_dict.get(0, 0.00))
        st.write("Процент выигрышей второго игрока:", result_dict.get(1, 0.00))
        if check:
            st.write("Time: ", f-s)
    else:
        raise(ValueError("Не заданы карты"))
