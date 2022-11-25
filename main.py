from __future__ import annotations

import numpy as np
from typing import List, Callable, Any, TypeVar, Type
import heapq
import random


T = TypeVar('T')

def nlargest(a: List[T], n: int) -> List[T]:
    return heapq.nlargest(n, a)


class Team:
    def __init__(self, elo: float, performance_sd: float):
        self.elo = elo
        self.performance_sd = performance_sd

    def play(self, other: Team):
        performance = np.random.normal(self.elo, self.performance_sd)
        other_performance = np.random.normal(other.elo, other.performance_sd)
        return self if performance > other_performance else other


class Tournament:
    def __init__(self):
        self.teams = self.generate_teams()

    def generate_teams(self) -> List[Team]:
        mu = 1000
        sd = 100
        n = 1_000
        elo_list = list(float(x) for x in np.random.normal(mu, sd, n))
        elo_list = nlargest(elo_list, 16)
        random.shuffle(elo_list)

        return [Team(elo, 0) for elo in elo_list]

    def run(self) -> Team:
        return max(self.teams, key=(lambda x: x.elo))

    def best_team(self) -> Team:
        return max(self.teams, key=lambda team: team.elo)


def bracket_once(teams: List[Team]):
    i = 0
    winning_teams = []
    while i < len(teams):
       winner = teams[i].play(teams[i+1]) 
       winning_teams.append(winner)
       i += 2
    return winning_teams


class BracketTournament(Tournament):
    def run(self) -> Team:
        teams = self.teams
        while(len(teams) > 1):
            teams = bracket_once(teams)
        return teams[0]

        
def run_simul(class_type: Type, n = 10000):
    correct = 0
    for i in range(n):
        tournament = class_type()

        winner = tournament.run()
        best = tournament.best_team()

        if winner == best:
            correct += 1

    return float(correct) / n


class RandomTournament(Tournament):
    def run(self) -> Team:
        return self.teams[0]


def main():
    experiments = [
        # Tournament,
        # RandomTournament,
        BracketTournament
    ]

    max_name_length = max([len(x.__name__) for x in experiments])

    for class_type in experiments:
        result = run_simul(class_type)
        print(f'{class_type.__name__: <{max_name_length}} | score: {result}')


if __name__ == '__main__':
    main()