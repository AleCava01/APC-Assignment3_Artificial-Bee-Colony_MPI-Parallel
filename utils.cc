#include <iostream>
#include <random>
#include <limits>
#include <mpi.h>
#include <functional>
#include "schwefel.cc"

int get_random_index_except_i(const int & num_bees, int i, std::mt19937 & generator) {
    
    std::uniform_int_distribution<int> distribution(0, num_bees-2);
    int idx = distribution(generator);
    if (idx >= i) 
        idx++;

    return idx;
}

void set_random_position(std::vector<double> & position,
                         std::mt19937 & generator) {
    
    std::uniform_real_distribution<double> distribution (lb,ub);

    for (std::size_t d = 0; d < position.size(); ++d)
        position[d] = distribution(generator);
}

void select_random(std::vector<int> & local_receive_bee_idx,
                   const int & local_num_bees,
                   const int & num_bees,
                   const int & rank, 
                   std::mt19937 & generator) {

    for (std::size_t local_bee = 0; local_bee < local_num_bees; ++local_bee)
        local_receive_bee_idx[local_bee] = get_random_index_except_i(num_bees, local_num_bees*rank + local_bee, generator);
}

void select_by_fitness(std::vector<int> & local_source_idx,
                       std::vector<double> & curr_OF_value,
                       const int & local_num_bees,
                       const int & num_bees,
                       const int & rank,
                       std::mt19937 & generator) {

    std::vector<double> local_fitness(local_num_bees);

    // Compute fitness for local bees
    for (std::size_t local_bee = 0; local_bee < local_num_bees; ++local_bee){
        if (curr_OF_value[local_bee] >= 0)
            local_fitness[local_bee] = 1 / (1 + curr_OF_value[local_bee]);
        else
            local_fitness[local_bee] = 1 + std::abs(curr_OF_value[local_bee]);
    }

    // Share local fitness
    std::vector<double> fitness(num_bees);
    MPI_Allgather(local_fitness.data(), local_num_bees, MPI_DOUBLE, fitness.data(), local_num_bees, MPI_DOUBLE, MPI_COMM_WORLD);

    // Compute selection probability for each bee and select another bee
    double tot_fitness = 0;
    for (const auto & fit: fitness)
        tot_fitness += fit;

    for (std::size_t local_bee = 0; local_bee < local_num_bees; ++local_bee){
        std::vector<double> fitness_without_local_bee;
        double norm_factor = tot_fitness - fitness[rank*local_num_bees + local_bee];
        for (std::size_t global_bee = 0; global_bee < num_bees; ++global_bee){
            if (global_bee != rank*local_num_bees + local_bee)
                fitness_without_local_bee.push_back(fitness[global_bee] / norm_factor);
            else
                fitness_without_local_bee.push_back(0.0);
        }

        std::discrete_distribution<int> distribution(fitness_without_local_bee.begin(), fitness_without_local_bee.end());
        local_source_idx[local_bee] = distribution(generator);
    }
}

void print(const std::vector<std::vector<double>> & current_positions,
           const std::vector<double> & current_OF_value,
           const std::vector<int> & trials,
           const int & local_num_bees,
           const int & num_bees,
           const int & rank,
           const int & size) {
    
    for (std::size_t bee = 0; bee < num_bees; ++bee){
        if (bee/local_num_bees == rank){
            std::cout << "Bee " << bee << " -> ";
            for (const auto & coord: current_positions[bee % local_num_bees])
                std::cout << coord << " ";
            std::cout << ", f(x) = " << current_OF_value[bee % local_num_bees] << ", trials: " << trials[bee % local_num_bees] << std::endl;
        }
    }
}

void print_best(const std::vector<std::vector<double>> & best_positions, 
                const std::vector<double> & local_best_OF_value, 
                const int & local_num_bees,
                const int & num_bees, 
                const int & rank) {

    std::vector<double> best_OF_values(num_bees);
    MPI_Allgather(local_best_OF_value.data(), local_num_bees, MPI_DOUBLE, best_OF_values.data(), local_num_bees, MPI_DOUBLE, MPI_COMM_WORLD);

    auto it = std::min_element(best_OF_values.begin(), best_OF_values.end());
    double global_min = *it;
    int argmin = std::distance(best_OF_values.begin(), it);

    if (argmin / local_num_bees == rank){

        std::cout << "min: f(x) = " << global_min << ", x = [";

        for (std::size_t d = 0; d < dim; ++d){
            if (d < dim-1)
                std::cout << best_positions[argmin % local_num_bees][d] << ", ";
            else
                std::cout << best_positions[argmin % local_num_bees][d] << "]" << std::endl;
        }
    }
}