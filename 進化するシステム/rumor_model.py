
from dataclasses import dataclass
from enum import Enum
import numpy as np


class PDActionRule(Enum):
    RUMOR_USER = "RUMOR_USER"
    DEFECTOR = "DEFECTOR"
    TFT_LIKE = "TFT_LIKE"


class StartRumorRule(Enum):
    OWN = "OWN_RUMOR_STARTER"
    BAD = "BAD_RUMOR_STARTER"
    GOOD = "GOOD_RUMOR_STARTER"
    NONE = "NO_RUMOR_STARTER"


class SpreadRumorRule(Enum):
    ALL = "ALL_RUMOR_SPREADER"
    NONE = "NO_RUMOR_SPREADER"
    GOOD = "GOOD_RUMOR_SPREADER"
    BAD = "BAD_RUMOR_SPREADER"
    CONDITIONAL = "CONDITIONAL_RUMOR_SPREADER"

@dataclass(frozen=True)
class Strategy:
    name: str
    rule1: PDActionRule
    rule2: StartRumorRule
    rule3: SpreadRumorRule
HONEST = Strategy(
    name = "HONEST",
    rule1 = PDActionRule.RUMOR_USER,
    rule2 = StartRumorRule.OWN,
    rule3 = SpreadRumorRule.ALL
)

LIAR = Strategy(
    name = "LIAR",
    rule1 = PDActionRule.DEFECTOR,
    rule2 = StartRumorRule.OWN,
    rule3 = SpreadRumorRule.ALL
)

GOOD = Strategy(
    name = "GOOD",
    rule1 = PDActionRule.RUMOR_USER,
    rule2 = StartRumorRule.GOOD,
    rule3 = SpreadRumorRule.ALL,
)

ADVISOR = Strategy(
    name = "ADVISOR",
    rule1= PDActionRule.RUMOR_USER,
    rule2 = StartRumorRule.BAD,
    rule3=SpreadRumorRule.ALL
)

LIKE_TFT = Strategy(
    name="LIKE_TFT",
    rule1=PDActionRule.TFT_LIKE,
    rule2=StartRumorRule.NONE,
    rule3=SpreadRumorRule.NONE
)

CONDITIONAL_HONEST = Strategy(
    name = "CONDITIONAL_HONEST",
    rule1 = PDActionRule.RUMOR_USER,
    rule2 = StartRumorRule.OWN,
    rule3 = SpreadRumorRule.CONDITIONAL
)

CONDITIONAL_ADVISOR = Strategy(
    name="CONDITIONAL_ADVISOR",
    rule1= PDActionRule.RUMOR_USER,
    rule2 = StartRumorRule.BAD,
    rule3 = SpreadRumorRule.CONDITIONAL
)
@dataclass
class Agent:
    id: int
    strategy : Strategy
class RumorSimulation_fast:
    def __init__(self, initial_counts, T=100,g=5,r=1,b=5,cost=2,seed=None,score_mode = "clip_zero",reproduction_mode="deterministic"):
        self.initial_counts = initial_counts        #初期配置の集団を設定
        self.T = T                          #単位時間:1世代の繰り返しの回数
        self.g = g                          #１単位時間当たりの囚人のジレンマゲームの回数
        self.r = r                          #１単位時間当たりの噂交換フェーズの回数
        self.b = b                          #協力されたときの利得
        self.cost = cost                    #協力したときのコスト
        self.payoff_shift = self.cost
        self.rng = np.random.default_rng(seed)

        self.population = self.create_population()
        self.N = len(self.population)

        # #0より小さくなった時の処理方法を指定する
        # self.score_mode = score_mode

        #レコード保存に必要な内容を用意
        self.generation = 0
        self.history = []
        self.strategy_names = [strategy.name for strategy in self.initial_counts.keys()]

        #次世代人数の確定方法
        self.reproduction_mode = reproduction_mode

        #初期化

        self.reset_generation_state()

    def create_population(self):
        agents = []
        current_id = 0
        for strategy,count in self.initial_counts.items():
            for m in range(count):
                agent = Agent(id = current_id, strategy = strategy)
                agents.append(agent)
                current_id += 1

        return agents

    def reset_generation_state(self):
        self.C = np.zeros((self.N,self.N),dtype = int)
        self.D = np.zeros((self.N,self.N),dtype = int)
        self.pc = np.zeros((self.N,self.N),dtype = int)
        self.pd = np.zeros((self.N,self.N),dtype = int)

        self.last_action = np.zeros((self.N,self.N),dtype = int)
        self.payoff = np.zeros(self.N,dtype = float)

    def decide_action(self, i, j):
        strategy = self.population[i].strategy

        if strategy.rule1 == PDActionRule.RUMOR_USER:
            if self.C[i, j] > self.D[i, j]:
                return "C"
            elif self.C[i, j] < self.D[i, j]:
                return "D"
            elif self.C[i, j] == self.D[i, j] and self.C[i, j] > 0:
                if self.rng.random() < 0.5:
                    return "C"
                else:
                    return "D"
            else:
                return "C"

        elif strategy.rule1 == PDActionRule.DEFECTOR:
            return "D"

        elif strategy.rule1 == PDActionRule.TFT_LIKE:
            if self.last_action[i, j] == -1:
                return "D"
            else:
                return "C"

        else:
            raise ValueError(f"Unknown PDActionRule: {strategy.rule1}")

    def play_pd_pair(self,i,j):  ##任意の二点の囚人のジレンマを計算し、協力非協力を収集する
        i_action = self.decide_action(i,j)
        j_action = self.decide_action(j,i)

        #まず元の利得を計算する
        raw_payoff_i = 0
        raw_payoff_j = 0
        if i_action == "C":
            raw_payoff_i -= self.cost
            raw_payoff_j += self.b

            # j から見て、i が協力した
            self.pc[j,i] += 1
            self.last_action[j,i] = 1
        else:

            # j から見て、i が裏切った
            self.pd[j,i] += 1
            self.last_action[j,i] = -1

        if j_action == "C":
            raw_payoff_i += self.b
            raw_payoff_j -= self.cost

            # i から見て、j が協力した
            self.pc[i,j] += 1
            self.last_action[i,j] = 1
        else:
            # i から見て、j が裏切った
            self.pd[i,j] += 1
            self.last_action[i,j] = -1

        #3.固定の下駄をはかせた利得を保存する
        self.payoff[i] += raw_payoff_i + self.payoff_shift
        self.payoff[j] += raw_payoff_j + self.payoff_shift


    def choice_pair(self):
        return self.rng.choice(self.N,size=2,replace=False) #replace = Falseで重複を防ぐ

    def play_pd_once(self):
        i,j = self.choice_pair()
        self.play_pd_pair(i,j)

    ##噂交換フェーズ

    #=============================================
    #噂の利得が同じで＞0のときのランダムに噂を広める関数
    #=============================================
    def random_spread(self,listener,target,delta_C,delta_D):
        if self.rng.random() > 0.5:
            delta_C[listener,target] += 1
        else:
            delta_D[listener,target] += 1
        return delta_C,delta_D

    #==============
    ##噂を広める関数
    #==============
    def spread_existing_rumors_fast(self, speaker, listener, C_row, D_row, pc_row, pd_row):
        strategy = self.population[speaker].strategy

        if strategy.rule3 == SpreadRumorRule.NONE:
            return

        valid = np.ones(self.N, dtype=bool)
        valid[speaker] = False
        valid[listener] = False

        c = C_row
        d = D_row

        if strategy.rule3 == SpreadRumorRule.ALL:
            pos_mask = valid & (c > d)
            neg_mask = valid & (c < d)
            tie_mask = valid & (c == d) & (c > 0)

            self.C[listener, pos_mask] += 1
            self.D[listener, neg_mask] += 1

            tie_targets = np.where(tie_mask)[0]
            if len(tie_targets) > 0:
                random_values = self.rng.random(len(tie_targets))
                pos_targets = tie_targets[random_values > 0.5]
                neg_targets = tie_targets[random_values <= 0.5]

                self.C[listener, pos_targets] += 1
                self.D[listener, neg_targets] += 1

        elif strategy.rule3 == SpreadRumorRule.GOOD:
            pos_mask = valid & (c > d)
            tie_mask = valid & (c == d) & (c > 0)

            self.C[listener, pos_mask] += 1

            tie_targets = np.where(tie_mask)[0]
            if len(tie_targets) > 0:
                random_values = self.rng.random(len(tie_targets))
                pos_targets = tie_targets[random_values > 0.5]
                neg_targets = tie_targets[random_values <= 0.5]

                self.C[listener, pos_targets] += 1
                self.D[listener, neg_targets] += 1

        elif strategy.rule3 == SpreadRumorRule.BAD:
            neg_mask = valid & (c < d)
            tie_mask = valid & (c == d) & (c > 0)

            self.D[listener, neg_mask] += 1

            tie_targets = np.where(tie_mask)[0]
            if len(tie_targets) > 0:
                random_values = self.rng.random(len(tie_targets))
                pos_targets = tie_targets[random_values > 0.5]
                neg_targets = tie_targets[random_values <= 0.5]

                self.C[listener, pos_targets] += 1
                self.D[listener, neg_targets] += 1

        elif strategy.rule3 == SpreadRumorRule.CONDITIONAL:
            pos_mask = valid & (pc_row > pd_row) & (c > d)
            tie_mask = valid & (pc_row > pd_row) & (c == d) & (c > 0)

            self.C[listener, pos_mask] += 1

            tie_targets = np.where(tie_mask)[0]
            if len(tie_targets) > 0:
                random_values = self.rng.random(len(tie_targets))
                pos_targets = tie_targets[random_values > 0.5]
                neg_targets = tie_targets[random_values <= 0.5]

                self.C[listener, pos_targets] += 1
                self.D[listener, neg_targets] += 1


    def start_new_rumors_fast(self, speaker, listener, pc_row, pd_row):
        strategy = self.population[speaker].strategy

        if strategy.rule2 == StartRumorRule.NONE:
            return

        if strategy.rule2 == StartRumorRule.OWN:
            self.C[listener, speaker] += 1
            return

        valid = np.ones(self.N, dtype=bool)
        valid[speaker] = False
        valid[listener] = False

        experience_exists = (pc_row + pd_row) > 0

        if strategy.rule2 == StartRumorRule.GOOD:
            good_mask = valid & (pc_row > pd_row)
            tie_mask = valid & experience_exists & (pc_row == pd_row)

            self.C[listener, good_mask] += 1

            tie_targets = np.where(tie_mask)[0]
            if len(tie_targets) > 0:
                random_values = self.rng.random(len(tie_targets))
                selected_targets = tie_targets[random_values > 0.5]
                self.C[listener, selected_targets] += 1

        elif strategy.rule2 == StartRumorRule.BAD:
            bad_mask = valid & (pd_row > pc_row)
            tie_mask = valid & experience_exists & (pd_row == pc_row)

            self.D[listener, bad_mask] += 1

            tie_targets = np.where(tie_mask)[0]
            if len(tie_targets) > 0:
                random_values = self.rng.random(len(tie_targets))
                selected_targets = tie_targets[random_values > 0.5]
                self.D[listener, selected_targets] += 1




    #噂の内容を処理する関数
    def rumor_exchange_pair(self, i, j):
    # 噂交換開始時点の speaker 行だけコピーする
        C_i = self.C[i].copy()
        D_i = self.D[i].copy()
        pc_i = self.pc[i].copy()
        pd_i = self.pd[i].copy()

        C_j = self.C[j].copy()
        D_j = self.D[j].copy()
        pc_j = self.pc[j].copy()
        pd_j = self.pd[j].copy()

        # i -> j
        # i -> j spread
        self.spread_existing_rumors_fast(
            speaker=i,
            listener=j,
            C_row=C_i,
            D_row=D_i,
            pc_row=pc_i,
            pd_row=pd_i
        )

        # j -> i spread
        self.spread_existing_rumors_fast(
            speaker=j,
            listener=i,
            C_row=C_j,
            D_row=D_j,
            pc_row=pc_j,
            pd_row=pd_j
        )

        # i -> j start
        self.start_new_rumors_fast(
            speaker=i,
            listener=j,
            pc_row=pc_i,
            pd_row=pd_i
        )

        # j -> i start
        self.start_new_rumors_fast(
            speaker=j,
            listener=i,
            pc_row=pc_j,
            pd_row=pd_j
        )

    def rumor_exchange_once(self):
        i,j = self.rng.choice(self.N,size=2,replace = False)
        self.rumor_exchange_pair(i,j)

    def run_generation(self):
        self.reset_generation_state()

        for _ in range(self.T):
            for _ in range(self.g):
                self.play_pd_once()

            for _ in range(self.r):
                self.rumor_exchange_once()

    #================
    #世代交代を行う部分
    #================

    #戦略ごとの利得を計算する関数
    def calc_strategy_scores(self):
        strategy_scores = {}

        for index,agent in enumerate(self.population):
            strategy = agent.strategy

            if strategy not in strategy_scores:
                strategy_scores[strategy] = 0.0

            strategy_scores[strategy] += self.payoff[index]

        return strategy_scores

    # def transform_scores(self,scores):
    #     if scores.min() >= 0:
    #         return scores

    #     if self.score_mode == "clip_zero":
    #         for index,score in enumerate(scores):
    #             if score < 0:
    #                 scores[index] = 0

    #     elif self.score_mode == "shift_min":
    #         epsilon=1e-9

    #         score_min = scores.min()
    #         if score_min < 0:
    #             scores -= score_min
    #             scores += epsilon

    #     else:
    #         raise ValueError(f"Unknown score_mode: {self.score_mode}")

    #     return scores


    def scores_to_probabilities(self, strategy_scores):
        strategies = list(strategy_scores.keys())
        scores = []

        for strategy in strategies:
            scores.append(strategy_scores[strategy])

        scores = np.array(scores,dtype=float)

        total = scores.sum()

        if total == 0:
            strategy_count = {}
            for agent in self.population:
                strategy = agent.strategy

                if strategy not in strategy_count:
                    strategy_count[strategy] = 0

                strategy_count[strategy] += 1

            strategies = list(strategy_count.keys())
            counts = []

            for strategy in strategies:
                counts.append(strategy_count[strategy])

            counts = np.array(counts,dtype=float)

            probability = counts / self.N

            return strategies,probability

        elif total > 0:
            probability = scores / total

            return strategies,probability


    #次世代人数をきめる
    def make_next_count(self,probability):
        #N人を、probability の確率にしたがって各戦略へ割り振る
        if self.reproduction_mode == "multinomial":
            next_counts = self.rng.multinomial(self.N,probability)
        elif self.reproduction_mode == "deterministic":
            expected_counts = probability * self.N

            # まず小数点以下を切り捨て
            next_counts = np.floor(expected_counts).astype(int)

            # まだ足りない人数
            remaining = self.N - next_counts.sum()

            if remaining > 0:
                    # 小数部分を計算
                fractional_parts = expected_counts - next_counts

                # 小数部分が大きい順に並べる
                order = np.argsort(-fractional_parts)

                # 足りない人数ぶんだけ +1 する
                for idx in order[:remaining]:
                    next_counts[idx] += 1

            # 念のため確認
        assert next_counts.sum() == self.N

        return next_counts


    def reproduce(self):
        strategy_scores = self.calc_strategy_scores()
        strategies,probability = self.scores_to_probabilities(strategy_scores)
        next_counts = self.make_next_count(probability)

        new_population = []
        current_id = 0
        for strategy,count in zip(strategies,next_counts):
            for m in range(int(count)):
                new_agent = Agent(id = current_id,strategy=strategy)
                new_population.append(new_agent)
                current_id += 1

        self.population = new_population

        #念のため確認
        assert len(self.population) == self.N

    def count_strategies(self):
        strategy_counts = {}

        for agent in self.population:
            strategy = agent.strategy.name

            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0

            strategy_counts[strategy] += 1

        return strategy_counts

    def record_history(self):
        counts = self.count_strategies()
        strategy_scores = self.calc_strategy_scores()
        calc_strategy_total_payoffs = {}

        for strategy,score in strategy_scores.items():
            strategy_name = strategy.name
            calc_strategy_total_payoffs[strategy_name] = score

        calc_strategy_avg_payoffs = {}

        for strategy,score in calc_strategy_total_payoffs.items():
            calc_strategy_avg_payoffs[strategy] = calc_strategy_total_payoffs[strategy] / counts[strategy]

        record = {}
        record["generation"] = self.generation
        record["record_type"] = "played_generation"

        for strategy in self.strategy_names:
            count = counts.get(strategy,0)
            total_payoff = calc_strategy_total_payoffs.get(strategy,0)
            avg_payoff = calc_strategy_avg_payoffs.get(strategy,0)

            record[f"{strategy}_count"] = count
            record[f"{strategy}_freq"] = count / self.N
            record[f"{strategy}_total_payoff"] = total_payoff
            record[f"{strategy}_avg_payoff"] = avg_payoff

        self.history.append(record)


    def step_generation(self,record = True):
        #1世代のシミュレーションを実施する
        self.run_generation()
        #結果を記録する
        if record:
            self.record_history()
        #次世代の戦略とその人数を決定する
        self.reproduce()
        self.generation += 1

    #収束したかを判定する
    def is_converged(self):
        counts = self.count_strategies()
        if max(counts.values()) == self.N:
            return True
        else:
            return False

    #winnerを判別する
    def get_winner(self):
        counts = self.count_strategies()

        max_count = max(counts.values())

        winners = []
        for strategy_name,count in counts.items():
            if count == max_count:
                winners.append(strategy_name)

        return winners


    def record_final_state(self):
        counts = self.count_strategies()
        record = {}
        record["generation"] = self.generation
        record["record_type"] = "final_state"

        for strategy in self.strategy_names:
            count = counts.get(strategy,0)

            record[f"{strategy}_count"] = count
            record[f"{strategy}_freq"] = count / self.N
            record[f"{strategy}_total_payoff"] = None
            record[f"{strategy}_avg_payoff"] = None

        self.history.append(record)


    def run_until_convergence(self,max_generations = 10, record= True):
        if self.is_converged() == False:
            for _ in range(max_generations):
                self.step_generation(record = record)
                if self.is_converged():
                    break

        result = {}
        result["converged"] = self.is_converged()
        result["winner"] = self.get_winner()
        result["generation"] = self.generation
        if record:
            self.record_final_state()
            result["history"] = self.history
        else:
            result["history"] = None

        return result
