import math


def get_lists(L, N):

    num_states_lol = []

    if N == 2:
        for i in range(math.ceil(L/2)):
            estate = [0]*L
            estate[i] += 1
            estate[L-i-1] += 1
            num_states_lol.append(estate)

    elif L == 3 and N == 3:
        num_states_lol.append([1,1,1])
        num_states_lol.append([0,3,0])

    elif L == 5 and N == 3:
        num_states_lol.append([1,0,1,0,1])
        num_states_lol.append([0,1,1,1,0])
        num_states_lol.append([0,0,3,0,0])

    elif L == 7 and N == 3:
        num_states_lol.append([1,0,0,1,0,0,1])
        num_states_lol.append([0,1,0,1,0,1,0])
        num_states_lol.append([0,0,1,1,1,0,0])
        num_states_lol.append([0,0,0,3,0,0,0])

    elif L == 3 and N == 4:
        num_states_lol.append([2,0,2])
        num_states_lol.append([1,2,1])
        num_states_lol.append([0,4,0])

    elif L == 4 and N == 4:
        num_states_lol.append([2,0,0,2])
        num_states_lol.append([1,1,1,1])
        num_states_lol.append([0,2,2,0])

    elif L == 5 and N == 4:
        num_states_lol.append([1,1,0,1,1])
        num_states_lol.append([2,0,0,0,2])
        num_states_lol.append([0,2,0,2,0])
        num_states_lol.append([1,0,2,0,1])
        num_states_lol.append([0,1,2,1,0])
        num_states_lol.append([0,0,4,0,0])

    elif L == 6 and N == 4:
        num_states_lol.append([2,0,0,0,0,2])
        num_states_lol.append([0,2,0,0,2,0])
        num_states_lol.append([0,0,2,2,0,0])
        num_states_lol.append([1,1,0,0,1,1])
        num_states_lol.append([1,0,1,1,0,1])
        num_states_lol.append([0,1,1,1,1,0])

    elif L == 7 and N == 4:
        num_states_lol.append([2,0,0,0,0,0,2])
        num_states_lol.append([0,2,0,0,0,2,0])
        num_states_lol.append([0,0,2,0,2,0,0])
        num_states_lol.append([0,0,0,4,0,0,0])
        num_states_lol.append([1,1,0,0,0,1,1])
        num_states_lol.append([1,0,1,0,1,0,1])
        num_states_lol.append([0,1,1,0,1,1,0])
        num_states_lol.append([1,0,0,2,0,0,1])
        num_states_lol.append([0,1,0,2,0,1,0])
        num_states_lol.append([0,0,1,2,1,0,0])

    elif L == 8 and N == 4:
        num_states_lol.append([2,0,0,0,0,0,0,2])
        num_states_lol.append([0,2,0,0,0,0,2,0])
        num_states_lol.append([0,0,2,0,0,2,0,0])
        num_states_lol.append([0,0,0,2,2,0,0,0])
        num_states_lol.append([1,1,0,0,0,0,1,1])
        num_states_lol.append([1,0,1,0,0,1,0,1])
        num_states_lol.append([0,1,1,0,0,1,1,0])
        num_states_lol.append([1,0,0,1,1,0,0,1])
        num_states_lol.append([0,1,0,1,1,0,1,0])
        num_states_lol.append([0,0,1,1,1,1,0,0])

    elif L == 3 and N == 5:
        num_states_lol.append([1,3,1])
        num_states_lol.append([2,1,2])
        num_states_lol.append([0,5,0])

    elif L == 5 and N == 5:
        num_states_lol.append([1,1,1,1,1])
        num_states_lol.append([2,0,1,0,2])
        num_states_lol.append([0,2,1,2,0])
        num_states_lol.append([1,0,3,0,1])
        num_states_lol.append([0,1,3,1,0])
        num_states_lol.append([0,0,5,0,0])

    elif L == 7 and N == 5:
        num_states_lol.append([2,0,0,1,0,0,2])
        num_states_lol.append([0,2,0,1,0,2,0])
        num_states_lol.append([0,0,2,1,2,0,0])
        num_states_lol.append([0,0,0,5,0,0,0])
        num_states_lol.append([1,0,0,3,0,0,1])
        num_states_lol.append([0,1,0,3,0,1,0])
        num_states_lol.append([0,0,1,3,1,0,0])
        num_states_lol.append([1,1,0,1,0,1,1])
        num_states_lol.append([1,0,1,1,1,0,1])
        num_states_lol.append([0,1,1,1,1,1,0])

    else:
        raise NotImplementedError('L=', L, ' N=', N)

    return num_states_lol