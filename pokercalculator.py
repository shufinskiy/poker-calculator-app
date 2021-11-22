import itertools
import pickle
import re
from typing import Dict

from numba import njit, typed, types
import numpy as np
import streamlit as st


@st.cache
def download_dict_hash_cards(path: str) -> Dict[int, int]:
    """Downloading a dictionary of form {card: hash_cards} fron file

    Args:
        path (str): file path

    Returns:
        dict: dictionary of form {card: hash_card}
    """

    filehandler = open(path, 'rb')
    card_dict = pickle.load(filehandler)
    filehandler.close()
    return card_dict


@st.cache(hash_funcs={typed.typeddict.Dict: id})
def create_dict_power_combinations(key: np.array, value: np.array):
    """Creating a Numba dictionary of form {hash_hands: power_hands}

    Args:
        key (np.array): an array with a hash number for each 5-card combination
        value (np.array): an array with a power for each 5-card combination

    Returns:
        typed.typeddict.Dict: numba Dictionary of form {hash_hands: power_hands}
    """

    power_comb_dict = typed.Dict.empty(types.int64, types.int16)
    for key, value in zip(key, value):
        power_comb_dict[key] = value

    return power_comb_dict


@st.cache
def load_numpy_array(path: str) -> np.array:
    """Downloading a numpy array from a file

    Args:
        path (str): file path

    Returns:
        np.array
    """
    return np.load(path)


def parsing_of_player_cards(inp_cards):
    """Parsing input and creating all possible pairs of player cards

    Args:
        inp_cards (tuple): tuple from player's card range

    Returns:
        tuple: a tuple all possible pairs cards for player

    Raises:
        ValueError: if incorrect data is entered
    """
    k = list()
    for i in inp_cards:
        if re.search(r'^([0-9AKQJT][hcds]){2}$', i):
            k.append(((i[:2], i[2:]),))
        elif re.search(r'^[0-9AKQJT]{2}[so]$', i) and i[0] != i[1]:
            if i[2] == 's':
                k.append(tuple([("".join(x), "".join(y)) for (x, y) in zip(itertools.product(i[1], "hcsd"),
                                            itertools.product(i[0], "hcsd"))]))
            elif i[2] == 'o':
                k.append(tuple([(x, y) for (x, y) in (itertools.product(["".join(x) for x in itertools.product(i[1], "hcsd")],
                       ["".join(x) for x in itertools.product(i[0], "hcsd")])) if x[1] != y[1]]))
        elif re.search(r'[0-9AKQJT]{2}$', i) and i[0] == i[1]:
            k.append(tuple(map([(x, y) for (x, y) in (itertools.product(["".join(x) for x in itertools.product(i[0], "hcsd")],
                       ["".join(x) for x in itertools.product(i[1], "hcsd")])) \
                                if x != y].__getitem__, [0, 1, 2, 4, 5, 8])))
        else:
            raise ValueError('Ошибка данных')
    return tuple(itertools.chain.from_iterable(k))


def convert_player_card_to_int(parsing_cards, card_id_dict):
    """Converting player cards from a string type to integer (np.int64)

    Args:
        parsing_cards (tuple): tuple all possible pairs cards for player
        card_id_dict (dict): dictionary of generated numbers for cards

    Returns:
        np.array: an array of integers of player's cards
    """
    return np.array(tuple((card_id_dict[fst], card_id_dict[snd]) for (fst, snd) in parsing_cards)).reshape(-1, 2)


def convert_board_card_to_int(card_on_board, card_id_dict):
    """Converting cards on board from a string type to integer (np.int64)

    Args:
        card_on_board (tuple): tuple cards on board (empty if no cards on board)
        card_id_dict (dict): dictionary of generated numbers for cards

    Returns:
        np.array: array of integers cards on board (empty if no cards on board)
    """
    if len(card_on_board[0]) == 0:
        return np.array([], dtype=np.int64)
    return np.array(tuple(card_id_dict[card] for card in card_on_board), dtype=np.int64)


def available_comb_players_cards(arr_pl_one, arr_pl_two):
    """Removing intersections of player cards

    Args:
        arr_pl_one (np.array): an array of cards first player
        arr_pl_two (np.array): an array of cards second player

    Returns:
        np.array: an array of players cards

    Raises:
        ValueError: if there is no available combination
    """
    all_comb = np.hstack((arr_pl_one.repeat(arr_pl_two.shape[0], axis=0), np.tile(arr_pl_two, (arr_pl_one.shape[0], 1))))
    available_comb = all_comb[np.array([len(np.unique(i)) for i in all_comb]) == 4]
    if available_comb.shape == (0, 4):
        raise ValueError('There is not a single available combination')
    return available_comb


def available_combinations(players_cards, board_cards):
    """Removing intersections between player cards and cards on board

    Args:
        players_cards (np.array): an array of player cards without intersections
        board_cards (np.array): an array cards on board(empty if no cards on board)

    Returns:
        tuple: a tuple of two arrays: an array of players cards, an array of board cards(empty if no cards on board)

    Raises:
        ValueError: if there is no available combination
    """
    if not board_cards.shape[0]:
        return players_cards, np.array([], dtype=np.int64)
    else:
        all_comb = np.hstack((players_cards, board_cards.reshape(1, -1).repeat(players_cards.shape[0], axis=0)))
        available_comb = all_comb[np.array([len(np.unique(i)) for i in all_comb]) == 7]
        if available_comb.shape == (0, 7):
            raise ValueError('There is not a single available combination')
        return available_comb[:, :4], available_comb[0, 4:].ravel()


@njit(cache=True)
def possible_player_hands(player_hand, card_on_board):
    """Create 21 possible combinations of five cards out of seven (player's cards plus cards on board).

    Args:
        player_hand (np.array): (2, ) an array player's cards
        card_on_board (np.array): (5, ) an array cards on board

    Returns:
        np.array: (21, 5) an array of combinations of five cards out of seven
    """
    arr_seven_card = np.hstack((player_hand, card_on_board))

    five_from_seven = np.stack((
        arr_seven_card[:5],
        np.hstack((arr_seven_card[:4], arr_seven_card[5:6])),
        np.hstack((arr_seven_card[:4], arr_seven_card[6:])),
        np.hstack((arr_seven_card[:3], arr_seven_card[4:6])),
        np.hstack((arr_seven_card[:3], arr_seven_card[4:5], arr_seven_card[6:])),
        np.hstack((arr_seven_card[:3], arr_seven_card[5:])),
        np.hstack((arr_seven_card[:2], arr_seven_card[3:6])),
        np.hstack((arr_seven_card[:2], arr_seven_card[3:5], arr_seven_card[6:])),
        np.hstack((arr_seven_card[:2], arr_seven_card[3:4], arr_seven_card[5:])),
        np.hstack((arr_seven_card[:2], arr_seven_card[4:])),
        np.hstack((arr_seven_card[:1], arr_seven_card[2:6])),
        np.hstack((arr_seven_card[:1], arr_seven_card[2:5], arr_seven_card[6:])),
        np.hstack((arr_seven_card[:1], arr_seven_card[2:4], arr_seven_card[5:])),
        np.hstack((arr_seven_card[:1], arr_seven_card[2:3], arr_seven_card[4:])),
        np.hstack((arr_seven_card[:1], arr_seven_card[3:])),
        arr_seven_card[1:6],
        np.hstack((arr_seven_card[1:5], arr_seven_card[6:])),
        np.hstack((arr_seven_card[1:4], arr_seven_card[5:])),
        np.hstack((arr_seven_card[1:3], arr_seven_card[4:])),
        np.hstack((arr_seven_card[1:2], arr_seven_card[3:])),
        arr_seven_card[2:]
    ))
    return five_from_seven


@njit(cache=True)
def get_power_hands(poss_hands, power_comb_dict):
    """Getting power of all possible combinations of the player's cards

    Args:
        poss_hands (np.array): (21, 5) an array all possible combinations of the player's cards
        power_comb_dict (typed.typeddict.Dict): numba Dictionary of  form {hash_hands: power_hands}
        Hash of hand is considered as sum of all five cards

    Returns:
        np.array: (21, ) an array with power of player's possible hands
    """
    sum_comb = np.sum(poss_hands, axis=1)
    sum_comb = np.ravel(sum_comb)

    for i in range(sum_comb.shape[0]):
        sum_comb[i] = power_comb_dict[sum_comb[i]]

    return sum_comb


@njit(cache=True)
def remove_pl_cards_from_deck(cards_in_deck, players_cards):
    """Removing players' cards from deck

    Args:
        cards_in_deck (np.array): an array of cards in deck
        players_cards (np.array): an array of cards on hands of players

    Returns:
        np.array: an array cards in deck without cards on hands of players
    """

    for i in players_cards:
        cards_in_deck = cards_in_deck[cards_in_deck != i]

    return cards_in_deck


@njit(cache=True)
def generate_plays(cards_in_deck, players_cards, comb_id_dict, nsamples, board_cards):
    """Generating plays and calculating their results

    Args:
        cards_in_deck (np.array): an array of cards in deck
        players_cards (np.array): an array of cards on hands of players:
        comb_id_dict (typed.typeddict.Dict): numba Dictionary of  form {hash_hands: power_hands}
        nsamples (int): number of plays
        board_cards (np.array): an array cards on board (empty if no cards on board)

    Returns:
        np.array: (nsamples, 2) an array with results of all plays. Row [1, 0] means win of first player, [0, 1]
        win of second player, [1, 1] tie
    """
    arr_results = np.zeros((nsamples, 2), dtype=np.int8)

    cnt_residual_board = cards_in_deck.shape[0] - 47

    for i in range(nsamples):
        n = np.random.randint(0, players_cards.shape[0])
        pl1_hand = players_cards[n][:2]
        pl2_hand = players_cards[n][2:]
        pl_cards = np.hstack((pl1_hand, pl2_hand))

        possible_board_cards = remove_pl_cards_from_deck(cards_in_deck, pl_cards)
        residual_board = np.random.choice(possible_board_cards, cnt_residual_board, False)
        full_board = np.hstack((board_cards, residual_board))

        pl1 = possible_player_hands(pl1_hand, full_board)
        pl1 = np.min(get_power_hands(pl1, comb_id_dict))

        pl2 = possible_player_hands(pl2_hand, full_board)
        pl2 = np.min(get_power_hands(pl2, comb_id_dict))

        if pl1 > pl2:
            result = np.array([1, 0])
        elif pl1 == pl2:
            result = np.array([1, 1])
        else:
            result = np.array([0, 1])

        arr_results[i, :] = result

    return arr_results


if __name__ == "__main__":
    ### Загрузка словаря карта : сгенерированное число
    filehandler = open("card_dict.dat", 'rb')
    card_dict = pickle.load(filehandler)
    filehandler.close()

    ### Загрузка словаря хэш_комбинации : сила_комбинации
    filehandler = open("comb_dict.dat", 'rb')
    d = pickle.load(filehandler)
    filehandler.close()

    ### Преобразование словаря в словарь Numba
    d1 = typed.Dict.empty(types.int64, types.int64)

    for k, v in d.items():
        d1[k] = v

    ### Генерация хэша 52 карт, замена индексов карт у игроков на их хэш
    np.random.seed(100)
    m = np.unique(np.random.randint(0, 15000, 250)) ** 3
    cards = np.random.choice(m, 52, False)

    pl1 = tuple((card.strip() for card in input("Введите карты первого игрока: ").split(',')))
    pl2 = tuple((card.strip() for card in input("Введите карты второго игрока: ").split(',')))
    brd = tuple((card.strip() for card in input("Введите карты на столе: ").split(',')))

    a = parsing_of_player_cards(pl1)
    b = parsing_of_player_cards(pl2)

    a = convert_player_card_to_int(a, card_dict)
    b = convert_player_card_to_int(b, card_dict)

    players_card = available_comb_players_cards(a, b)

    board = convert_board_card_to_int(brd, card_dict)

    players_card, board = available_combinations(players_card, board)

    if board.shape[0]:
        br = cards[np.isin(cards, board)]
        cards = cards[~np.isin(cards, board)]
    else:
        br = np.array([], dtype=np.int64)

    out = generate_plays(cards, players_card, d1, 500000, br)
    f = out[:, 0] - out[:, 1]

    _, cnt = np.unique(f, return_counts=True)
    print(cnt / 5000)

