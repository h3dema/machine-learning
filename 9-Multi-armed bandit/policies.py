"""

implementa das politicas do MAB listadas abaixo:

- e-greedy
- e-greedy with annealing
- softmax
- softmax with annealing
- pursuit
- UCB1

refs:
    Kuleshov, Volodymyr, and Doina Precup.
    "Algorithms for multi-armed bandit problems."
    arXiv preprint arXiv:1402.6028 (2014).

"""
from random import random, randint
import numpy as np


class policy(object):
    """
        classe básica
    """

    def __init__(self, K, verbose):
        self.K = K
        self.verbose = verbose
        self.selected_arm = []

    @property
    def selected_arms(self):
        return self.selected_arm


class e_greedy(policy):
    """
        A política Greedy só toma a melhor ação aparente,
        e os empates são desfeitos por uma seleção aleatória.
        Isso pode ser visto como um caso especial de Epsilon-Greedy
        onde epsilon = 0 ou seja, deve sempre explorar.
    """

    def __init__(self, K, epsilon, verbose=False):
        super(e_greedy, self).__init__(K, verbose)
        self.epsilon = epsilon
        if self.verbose:
            print('epsilon: %6.4f' % epsilon)

    def select_arm(self, means_k):
        best_arm = means_k.index(max(means_k))
        r_eps = random()
        if self.epsilon > r_eps:
            arm = randint(1, self.K) - 1
        else:
            arm = best_arm
        self.selected_arm.append(arm)
        return arm


class e_greedy_annealing(e_greedy):
    """
        O mesmo que Epsilon Greedy, mas o valor de tau diminuiu linearmente ao longo do tempo "t" (anneling)
    """

    def __init__(self, K, epsilon, verbose=False):
        super(e_greedy_annealing, self).__init__(K, epsilon, verbose)
        self.epsilon_zero = epsilon
        self.plays = 0

    def select_arm(self, means_k):
        self.plays += 1
        self.epsilon = self.epsilon_zero / self.plays
        return super(e_greedy_annealing, self).select_arm(means_k)


class softmax(policy):
    """
        A política Softmax converte as recompensas de braço estimadas em probabilidades e,
        em seguida, busca amostras aleatórias da distribuição resultante.
    """

    def __init__(self, K, tau, verbose=False):
        super(softmax, self).__init__(K, verbose)
        self.tau = float(tau)
        if self.verbose:
            print('tau: %6.4f' % tau)

    def prob_softmax(self, mu, tau):
        m = np.array(mu)
        min_ = np.min(m)
        delta = np.max(m) - min_
        e = np.exp((m - min_) / delta / tau)
        dist = e / np.sum(e)
        return dist.tolist()

    def select_arm(self, means_k):
        p = self.prob_softmax(means_k, self.tau)
        cdf = np.cumsum(p)
        s = np.random.random()
        arm = np.where(s < cdf)[0][0]
        self.selected_arm.append(arm)
        return arm


class softmax_annealing(softmax):
    """
        É igual ao Softmax,
        mas o valor de tau diminuiu linearmente ao longo do tempo "t"
    """

    def __init__(self, K, tau, verbose=False):
        self.tau_zero = float(tau)
        super(softmax_annealing, self).__init__(K, self.tau_zero, verbose)
        self.plays = 0

    def select_arm(self, means_k):
        self.plays += 1
        self.tau = self.tau_zero / self.plays
        return super(softmax_annealing, self).select_arm(means_k)


class pursuit(policy):
    """
        Algoritmos de perseguição mantêm uma política explícita sobre os braços,
        cujas atualizações são informadas pelos meios empíricos, mas são realizadas separadamente.
        onde 0 < beta < 1 é uma taxa de aprendizado.

        ref.
        Pursuit algorithms (Thathachar & Sastry, 1985)

    """

    def __init__(self, K, beta, verbose=False):
        super().__init__(K, verbose)
        self.beta = beta
        self.p = np.array([1 / float(K) for i in range(K)])

    def select_arm(self, means_k):
        best_arm = means_k.index(max(means_k))
        self.p -= self.beta * self.p
        self.p[best_arm] += self.beta
        cdf = np.cumsum(self.p)
        s = np.random.random()
        arm = np.where(s < cdf)[0][0]
        self.selected_arm.append(arm)
        return arm


class ucb1(policy):
    """
        UCB = Upper Confidence Bound algorithm (UCB1)

        UCB1 aplica um fator de exploração ao valor esperado de cada braço
        que pode influenciar uma estratégia de seleção gananciosa (e-greedy) para
        explorar de forma mais inteligente opções menos confiáveis.

    """

    def __init__(self, K, c=2, verbose=False):
        super().__init__(K, verbose)
        self.c = c
        self.action_attempts = [1 for i in range(K)]
        self.plays = 0

    def select_arm(self, means_k):
        self.plays += 1
        exploration = np.log(self.plays) / self.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1 / self.c)
        q = np.array(means_k) + exploration
        a = np.max(q)
        check = np.where(q == a)[0]
        arm = check[0] if len(check) == 1 else np.random.choice(check)
        self.action_attempts[arm] += 1
        self.selected_arm.append(arm)
        return arm
