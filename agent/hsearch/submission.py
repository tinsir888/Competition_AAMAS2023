# -*- coding:utf-8  -*-
# Time  : 2021/5/31 下午4:14
# Author: Yahui Cui

"""
# =================================== Important =========================================
Notes:
1. this agent is random agent , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""

"""
根据题目给出的observation，我们可以将手牌展开为一个长度为 34 的数组，
每个元素表示该牌的数量，然后再加上已经打出去的牌，就得到了整副牌的状态。
"""

import numpy as np


def is_win(hand):
    """
    判断手牌是否和牌
    
    参数:
    hand: numpy 数组，表示手牌
    
    返回值:
    bool 类型，表示手牌是否和牌
    """

    # 手牌数必须为 14 张
    if np.sum(hand) != 14:
        return False
    
    # 判断是否有雀头
    pairs = np.where(hand >= 2)[0]
    for pair in pairs:
        # 将雀头拆出来，然后判断剩下的牌能否组成顺子或刻子
        new_hand = hand.copy()
        new_hand[pair] -= 2
        if _check(new_hand, 0, 0):
            return True
    
    return False


def _check(hand, melds, idx):
    """
    判断是否能组成顺子或刻子
    Args:
        hand: numpy array，玩家手牌数组，shape为(34,)
        melds: int，已经组成的面子（刻子或顺子）的数量
        idx: int，开始搜索的位置

    Returns:
        bool，是否能组成面子
    """
    # 如果已经组成了 4 组面子，则一定能和牌
    if melds == 4:
        return True
    
    # 如果当前位置大于等于 27，则只能组成刻子
    if idx >= 27:
        for i in range(idx, 34):
            if hand[i] >= 3:
                new_hand = hand.copy()
                new_hand[i] -= 3
                if _check(new_hand, melds + 1, idx):
                    return True
    else:
        # 尝试组成顺子
        if idx % 9 <= 6 and hand[idx] >= 1 and hand[idx+1] >= 1 and hand[idx+2] >= 1:
            new_hand = hand.copy()
            new_hand[idx:idx+3] -= 1
            if _check(new_hand, melds + 1, idx):
                return True
        
        # 尝试组成刻子
        if hand[idx] >= 3:
            new_hand = hand.copy()
            new_hand[idx] -= 3
            if _check(new_hand, melds + 1, idx):
                return True
    
    return False

def give_score(hand):
    score = 0
    # 判断四条，每个记 10 分，刻子每个记 8 分，对子每个记 5 分
    for i in range(34):
        if hand[i] == 4:
            score += 23
        elif hand[i] == 3:
            score += 13
        elif hand[i] == 2:
            score += 5
    # 判断 顺子，每个记 5 分
    for idx in range(27):
        if idx % 9 <= 6 and hand[idx] >= 1 and hand[idx+1] >= 1 and hand[idx+2] >= 1:
            hand[idx:idx+3] -= min(hand[idx], hand[idx+1], hand[idx+2])
            score += 5*min(hand[idx], hand[idx+1], hand[idx+2])
    # 剩下的牌每个记 1 分
    for idx in range(27):
        score += hand[idx]
    return score

def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = []
    #print(observation['obs']['action_mask'])
    if observation['current_move_player'] == observation['controlled_player_name']:
        counts = observation['obs']['observation'][0]
        hand = np.sum(counts, axis=1)
        #print(hand)
        if (is_win(hand)) == True:
            #print("HELEHELEHELE!")
            action_ = [0]*38
            action_[37] = 1
            agent_action.append(action_)
            return agent_action
        """
        计算打出手中每一张牌后
        1.（如果有听的牌）听哪些牌
        2.（如果有听的牌）听这些牌的概率（转换为评分）之和
        3. 如果没有听牌，按照评分来打，匹配
        选取评分最大的那张牌打出。
        """
        max_score_tile_id = 0 # 记录打出之后获得评分最大的那张牌
        max_score = -100 # 记录评分的最大值
        # 计算对于每张牌，场上能出现的概率并用评分量化
        each_tile_remain = [4]*34
        for i in range(6):
            """ 
            i == 
            0: tiles on self's hand
            1: tiles on table;
            2: my chi peng gang;
            3: player0's chi peng gang;
            4: player1's chi peng gang;
            5: player2's chi peng gang;
            6: player3's chi peng gang;
            """
            tiles_on_table_counts = observation['obs']['observation'][i]
            tiles_on_table = np.sum(tiles_on_table_counts, axis=1)
            each_tile_remain -= tiles_on_table# 减去出过的牌
        #print(each_tile_remain)

        # 遍历每一张能打出的牌
        for i in range(34):
            if hand[i] == 0:
                continue
            hand_score = 0 # 对打出每一张牌之后的手牌进行评分
            new_hand = hand.copy()
            new_hand[i] -= 1
            give_score(new_hand)
            listen_tile = []
            for j in range(34):
                if i == j:
                    continue
                new_hand[j] += 1
                if is_win(new_hand) == True:
                    listen_tile.append(j)
                new_hand[j] -= 1
            cur_tile_score = sum(each_tile_remain[idx]*20 for idx in listen_tile) # 每张能和牌的牌记 20 分
            if cur_tile_score + hand_score > max_score:
                max_score = cur_tile_score + hand_score
                max_score_tile_id = i
        action_ = [0]*38
        action_[max_score_tile_id] = 1
        #print(action_)
        agent_action.append(action_)
        return agent_action
    
    agent_action = []
    for i in range(len(action_space)):
        action_ = sample_single_dim(action_space[i], is_act_continuous)
        agent_action.append(action_)
        #print(action_)
    return agent_action


def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
    return each


def sample(action_space_list_each, is_act_continuous):
    player = []
    if is_act_continuous:
        for j in range(len(action_space_list_each)):
            each = action_space_list_each[j].sample()
            player.append(each)
    else:
        player = []
        for j in range(len(action_space_list_each)):
            # each = [0] * action_space_list_each[j]
            # idx = np.random.randint(action_space_list_each[j])
            if action_space_list_each[j].__class__.__name__ == "Discrete":
                each = [0] * action_space_list_each[j].n
                idx = action_space_list_each[j].sample()
                each[idx] = 1
                player.append(each)
            elif action_space_list_each[j].__class__.__name__ == "MultiDiscreteParticle":
                each = []
                nvec = action_space_list_each[j].high
                sample_indexes = action_space_list_each[j].sample()

                for i in range(len(nvec)):
                    dim = nvec[i] + 1
                    new_action = [0] * dim
                    index = sample_indexes[i]
                    new_action[index] = 1
                    each.extend(new_action)
                player.append(each)
    return player