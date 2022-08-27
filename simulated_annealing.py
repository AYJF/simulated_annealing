import math
import random
import matplotlib.pyplot as plt
import tsp_utils
import animate_visualizer
import numpy as np


class SimulatedAnnealing:
    def __init__(self, dist_matrix,temp, alpha, stopping_temp, stopping_iter, home =0):
        ''' animate the solution over time
            Parameters
            ----------
            dist_matrix: pandas.dataframe
                distance matrix
            temp: float
                initial temperature
            alpha: float
                rate at which temp decreases
            stopping_temp: float
                temerature at which annealing process terminates
            stopping_iter: int
                interation at which annealing process terminates
        '''


        self.sample_size = len(dist_matrix)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1
        self.home = home
        
        
        self.dist_matrix = dist_matrix.values
        
        # self.curr_solution = tsp_utils.nearestNeighbourSolution(self.dist_matrix)
        self.curr_solution = np.arange(self.sample_size)
        
        self.best_solution = self.curr_solution

        self.curr_weight = self.update_distance(self.curr_solution)
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight

        self.weight_list = [self.curr_weight]
        self.min_weight_list = []

        print(self.curr_solution)    
        print('Intial weight: ', self.curr_weight)

    def weight(self, sol):
         return sum([self.dist_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])])
    
    def update_distance(self, sol):     
        # Reset distance
        distance = 0
        # Keep track of departing city
        from_index = self.home
        # Loop all cities in the current route
        for i in range(len(sol)):
            distance += self.dist_matrix[from_index][sol[i]]
            from_index = sol[i]
        # Add the distance back to home
        distance += self.dist_matrix[from_index][self.home]
        return distance

    def acceptance_probability(self, candidate_weight):
        return math.exp(-abs(candidate_weight - self.curr_weight) / self.temp)

    def accept(self, candidate):
        candidate_weight = self.update_distance(candidate)
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.min_weight = candidate_weight
                self.best_solution = candidate

        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate

    def anneal(self):
        '''
        Annealing process with 2-opt
        described here: https://en.wikipedia.org/wiki/2-opt
        '''  
        while self.temp >= self.stopping_temp:
            for _ in range(self.stopping_iter):
                candidate = list(self.curr_solution)
                l = random.randint(2, self.sample_size - 1)
                i = random.randint(0, self.sample_size - l)

                candidate[i: (i + l)] = reversed(candidate[i: (i + l)])

                self.accept(candidate)                              
                self.weight_list.append(self.curr_weight)
                
            self.temp *= self.alpha # update temp    
        
        print('Minimum weight: ', self.min_weight)
        print('Improvement: ',
            round((self.initial_weight - self.min_weight) / (self.initial_weight), 4) * 100, '%')
        
        return self.min_weight
        




    def plotLearning(self):
        plt.plot([i for i in range(len(self.weight_list))], self.weight_list)
        line_init = plt.axhline(y=self.initial_weight, color='r', linestyle='--')
        line_min = plt.axhline(y=self.min_weight, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial weight', 'Optimized weight'])
        plt.ylabel('Weight')
        plt.xlabel('Iteration')
        plt.show()