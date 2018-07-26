#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/25 13:18
# @Author  : Ting
from collections import Counter


class SequenceProb:
    def __init__(self, initial_prob, length):
        assert sum(initial_prob.values()) == 1, "initial probability doesn't sum to 1"
        assert len({len(key) for key in initial_prob}) == 1, "initial sequences doesn't have same length"
        assert {type(key) for key in initial_prob} == {tuple}, "key of initial distribution should be tuple"

        self.distribution = initial_prob
        self.length = length
        seq_len = len(list(initial_prob.keys())[0])
        while seq_len > self.length:
            self.remove_first()
            seq_len -= 1

        if seq_len < self.length:
            differ = self.length - seq_len
            self.distribution = {tuple('*')*differ + key: initial_prob[key] for key in initial_prob}

    def remove_first(self):
        new_distribution = Counter()
        for sequence in self.distribution:
            new_distribution[sequence[1:]] += self.distribution[sequence]
        self.distribution = new_distribution

    def append_next(self, candidates, transition):
        new_distribution = Counter()
        for can in candidates:
            for sequence in self.distribution:
                new_distribution[sequence + tuple(can)] += self.distribution[sequence] * transition(sequence, can)
        self.distribution = new_distribution

    def update(self, candidates, transition):
        assert {len(i) for i in self.distribution} == {self.length}, "length of sequence not equal to definition"
        self.remove_first()
        self.append_next(candidates, transition)

    def last_prob(self):
        distribution = {seq[-1]: .0 for seq in self.distribution}
        for sequence in self.distribution:
            distribution[sequence[-1]] += self.distribution[sequence]
        return distribution


class Transition:
    def __init__(self, conditional_prob):
        self.transition = conditional_prob

    def next_prob(self, sequence, observation):
        if observation in self.transition[sequence]:
            return self.transition[sequence][observation]
        else:
            return 0.0


class HMM:
    def __init__(self, initial_distribution, transition, emission, length=3):
        self.transition = transition
        self.emission = emission
        self.initial_distribution = initial_distribution

        self.hidden_states = emission.keys()
        self.observ_states = list(emission.values())[0].keys()

        # 定义 transition prob 是 几个元素 的联合分布
        # 一般来说，只取决于上一个元素，也就是 length = 2
        # 对于 2-gram, length = 2; 对于3-gram, length = 3
        self.length = length

    def forward(self, observations):
        # alpha[t][state] denotes P(observation, hidden state of time t = state)
        T = len(observations)
        alpha = {t: {state: 0 for state in self.hidden_states} for t in range(T)}

        prob = SequenceProb(self.initial_distribution, self.length)
        alpha[0] = prob.last_prob()
        for t in range(T-1):
            prob.update(self.hidden_states, self.transition.next_prob)
            sum_prob = prob.last_prob()
            for state in sum_prob:
                alpha[t+1][state] = sum_prob[state] * self.emission[state][observations[t+1]]
        return sum([alpha[T-1][state] for state in self.hidden_states])

    def backward(self, observations):
        # beta[t][state] denotes P(observation[t+1:], hidden state of time t = state)
        T = len(observations)
        beta = {t: {state: 0 for state in self.hidden_states} for t in range(T)}

        prob = SequenceProb(self.initial_distribution, self.length)
        beta[T-1] = prob.last_prob()
        for t in range(T-1, 0, -1):
            for state in self.hidden_states:
                for state_prev in beta[t]:
                    beta[t-1][state] = self.emission[state][observations[t]] * \
                                       self.transition.next_prob(state, state_prev) * beta[t][state_prev]
        return sum(beta[0].values())

    def viterbi(self, observations):
        T = len(observations)
        assert T >= self.length, "not enough observations"

        backpoint = ['' for _ in observations]
        # max_prob[t][state] denotes maximum probability of a hidden sequence
        # that ended with state
        max_prob = {t: {state: 0 for state in self.hidden_states} for t in range(T)}

        seq_prob = SequenceProb(self.initial_distribution, self.transition.next_prob)
        max_prob[0] = seq_prob.last_prob()
        backpoint[0] = max(max_prob[0].keys(), key=lambda x: max_prob[0][x])
        for t in range(T-self.length):
            seq_prob.update(self.hidden_states, self.transition.next_prob)
            prob = seq_prob.last_prob()
            max_prob[t+1] = max([max_prob[t][state] * prob[state] for state in prob])
            backpoint[t+1] = max(max_prob[t].keys(), key=lambda x: max_prob[t][x])[-1]
        backpoint[t+1:] = max(max_prob[t].keys(), key=lambda x: max_prob[t][x])
        return backpoint

    def baum_welch(self, observations):
        # E-step:
        # M-step:

        # the probability of being in state1 at time t and state2 at time t+1
        # P(hidden[t] = state1, hidden[t+1] = state2 | observations, model)
        prob = {state1: {state2 for state2 in self.hidden_states} for state1 in self.hidden_states}
        prob = {t: prob for t in range(T)}

        # prob[t] ∝ forward(observations) * transition * emission * backward(observations)
        prob[t] = self.forward(observations) * self.transition * self.emission[][] * self.backward(observations)
        # normalize(prob[t])
        sum_prob = sum(prob[t].values())
        prob[t] = prob[t]/sum_prob
        return

    def evaluation(self, observations):
        return self.forward(observations)

    def recognition(self, observations):
        return self.viterbi(observations)

    def training(self, observations):
        return self.baum_welch(observations)
