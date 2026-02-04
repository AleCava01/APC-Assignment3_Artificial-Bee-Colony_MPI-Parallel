#include "utils.cc"

void initialize_positions(std::vector<std::vector<double>> & positions, 
                          std::mt19937 & generator) {

    for (std::vector<double> & position: positions){
        set_random_position(position, generator);
    }
}

void communicate_other_bees_position(const std::vector<std::vector<double>> & send_positions, 
                                     std::vector<std::vector<double>> & receive_positions,
                                     const std::vector<int> & receive_bee_idx,
                                     const int & local_num_bees, 
                                     const int & num_bees,
                                     const int & rank) {

    // This function manages the communication of other bees’ positions during both the employed and onlooker bee phases. 
    // Specifically, the vector send_positions contains the current positions of the local_bees, while receive_positions 
    // is used to store the (local) positions of the other bees. Moreover, receive_bee_idx is an $N$-dimensional vector 
    // that stores the global IDs of the bees required by each global_bee, namely index $k$ in (2). 
    // Recall that bees are block-partitioned across the ranks and that the positions of all bees should never be shared with all other bees; 
    // only the positions of the required bees must be communicated.


    // I used non-blocking communication to prevent deadlocks in irregular patterns 
    // and to overlap data transfer with computation for better parallel efficiency.

    std::vector<MPI_Request> requests;
    int size = num_bees / local_num_bees; // deduce size

    for (int i = 0; i < local_num_bees; ++i) {
        int i_global = i + rank * local_num_bees;
        int k_global = receive_bee_idx[i_global];
        int k_rank = k_global / local_num_bees;

        if (k_rank == rank) {
            receive_positions[i] = send_positions[k_global % local_num_bees];
        } 
        else {
            requests.emplace_back();
            MPI_Irecv(receive_positions[i].data(),
                dim, 
                MPI_DOUBLE, 
                k_rank, 
                k_global, // k index as tag
                MPI_COMM_WORLD, 
                &requests.back());
        }
    }

    for (int r = 0; r < size; ++r) { // iterate over ranks
        for (int i = 0; i < local_num_bees; ++i) {
            int k_global = receive_bee_idx[r * local_num_bees + i];
            int k_rank = k_global / local_num_bees;

            if (k_rank == rank && r != rank) { // if the rank contains k, ask him to send it's position.
                requests.emplace_back();
                MPI_Isend(send_positions[k_global % local_num_bees].data(), 
                    dim, 
                    MPI_DOUBLE, 
                    r, 
                    k_global, // k index as tag
                    MPI_COMM_WORLD, 
                    &requests.back());
            }
        }
    }

    if (!requests.empty()){
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    
}

void evaluate_and_update(const std::vector<std::vector<double>> & positions_to_evaluate,
                         std::vector<std::vector<double>> & current_positions,
                         std::vector<double> & current_OF_value,
                         std::vector<int> & trials,
                         const int & rank) {

    // This function computes the objective function value for each local_bee, updates the trial counter, and updates the 
    // current position and objective function value whenever the greedy selection criterion is satisfied.

    for(size_t i = 0; i < current_OF_value.size(); ++i) {
        double new_pos_OF = f(positions_to_evaluate[i]);
        if(new_pos_OF < current_OF_value[i]) {
            current_positions[i] = positions_to_evaluate[i];
            current_OF_value[i] = new_pos_OF;
            trials[i] = 0;
        } else {
            trials[i]++;
        }
    }

}

void find_new_position(const std::vector<std::vector<double>> & current_positions,
                       std::vector<std::vector<double>> & other_bee_positions,
                       const int & local_num_bees,
                       std::mt19937 & generator) {

    std::uniform_int_distribution<int> int_distribution(0, dim-1);
    std::uniform_real_distribution<double> real_distribution (-1,1);

    int update_dim;
    double rand_factor;

    for (std::size_t local_bee = 0; local_bee < local_num_bees; ++local_bee){
        update_dim = int_distribution(generator);
        rand_factor = real_distribution(generator);

        other_bee_positions[local_bee][update_dim] = current_positions[local_bee][update_dim] + rand_factor*(current_positions[local_bee][update_dim] - other_bee_positions[local_bee][update_dim]);
        other_bee_positions[local_bee][update_dim] = std::max(lb, other_bee_positions[local_bee][update_dim]);
        other_bee_positions[local_bee][update_dim] = std::min(ub, other_bee_positions[local_bee][update_dim]);

        for (std::size_t d = 0; d < dim; ++d){
            if (d != update_dim)
                other_bee_positions[local_bee][d] = current_positions[local_bee][d];
        }
    }
}

void share_other_bees_position(const std::vector<std::vector<double>> & current_positions,
                               const std::vector<int> & local_receive_bee_idx,
                               std::vector<std::vector<double>> & receive_positions,
                               const int & local_num_bees,
                               const int & num_bees,
                               const int & rank) {

    // This function manages the retrieval of the other bees’ indices and the communication of their positions, 
    // given the indices $k$ selected by the local_bees.
    
    // Collect all IDs of required bees' positions
    
    std::vector<int> all_receive_bee_idx(num_bees);

    MPI_Allgather(local_receive_bee_idx.data(), // Allgather to collect all receive_bee_idx from all ranks
        local_num_bees, MPI_INT, 
        all_receive_bee_idx.data(), 
        local_num_bees, 
        MPI_INT, 
        MPI_COMM_WORLD);

    // Communicate other bees' positions

    communicate_other_bees_position(current_positions, receive_positions, all_receive_bee_idx, local_num_bees, num_bees, rank);
}


void employed_bee_phase(std::vector<std::vector<double>> & current_positions, 
                        std::vector<double> & curr_OF_value, 
                        std::vector<int> & trials,
                        const int & local_num_bees,
                        const int & num_bees, 
                        const int & rank, 
                        std::mt19937 & generator){

    // This function implements the Employed Bee Phase. Specifically, 
    //      (i) each local_bee randomly selects another bee; 
    //      (ii) the positions of the selected bees are communicated; 
    //      (iii) new candidate positions are generated and evaluated, followed by the application of the greedy update mechanism.

    // For each bee, select another bee

    std::vector<int> local_neighbor_idx(local_num_bees);
    for(int i = 0; i < local_num_bees; ++i) {
        local_neighbor_idx[i] = get_random_index_except_i(num_bees, rank * local_num_bees + i, generator);
    }

    std::vector<std::vector<double>> neighbor_positions(local_num_bees, std::vector<double>(dim));
    share_other_bees_position(current_positions, local_neighbor_idx, neighbor_positions, local_num_bees, num_bees, rank);

    // find_new_position usa neighbor_positions come input E output
    find_new_position(current_positions, neighbor_positions, local_num_bees, generator);

    evaluate_and_update(neighbor_positions, current_positions, curr_OF_value, trials, rank);
    
}

void onlookers_bee_phase(std::vector<std::vector<double>> & current_positions, 
                        std::vector<double> & curr_OF_value, 
                        std::vector<int> & trials,
                        const int & local_num_bees,
                        const int & num_bees, 
                        const int & rank, 
                        std::mt19937 & generator){

    // This function implements the Onlooker Bee Phase. Specifically, it 
    //      (i) computes the fitness of each bee and selects the source bee using the roulette-wheel mechanism; 
    //      (ii) shares the positions of the selected source bees; 
    //      (iii) performs the same steps as in the Employed Bee Phase.

    // Compute fitness and find the source
    std::vector<int> local_source_idx(local_num_bees);
    select_by_fitness(local_source_idx, curr_OF_value, local_num_bees, num_bees, rank, generator); // updates local source idx with bees selected with fitness roulette system.

    // Share source information
    std::vector<std::vector<double>> source_positions(local_num_bees, std::vector<double>(dim));
    share_other_bees_position(current_positions, local_source_idx, source_positions, local_num_bees, num_bees, rank);
    
    
    // Select another bee randomly
    std::vector<int> local_neighbor_idx(local_num_bees);
    for(int i = 0; i < local_num_bees; ++i) {
        local_neighbor_idx[i] = get_random_index_except_i(num_bees, local_source_idx[i], generator);
    }

    // Share the information
    std::vector<std::vector<double>> neighbor_positions(local_num_bees, std::vector<double>(dim));
    share_other_bees_position(current_positions, local_neighbor_idx, neighbor_positions, local_num_bees, num_bees, rank);

    // Find the new position from source
    find_new_position(source_positions, neighbor_positions, local_num_bees, generator);

    // Update info
    evaluate_and_update(neighbor_positions, current_positions, curr_OF_value, trials, rank);
    
}


void scout_bee_phase(std::vector<std::vector<double>> & current_positions,
                     std::vector<std::vector<double>> & best_positions, 
                     std::vector<double> & curr_OF_value,
                     std::vector<double> & best_OF_value, 
                     std::vector<int> & trials,
                     const int & local_num_bees,
                     const int & max_trials,
                     std::mt19937 & generator){

    // This function implements the Scout Bee Phase. Specifically, if trials >= max_trials for a bee, its position is randomly 
    // reinitialized and its trial counter is reset to zero.

    for(int i = 0; i<local_num_bees; ++i){
        if(trials[i]>=max_trials){
            set_random_position(current_positions[i], generator);
            curr_OF_value[i] = f(current_positions[i]);
            trials[i] = 0;
        }
    }
}

void abc_algorithm(const int & num_bees, 
                   const int & iter_num, 
                   const int & max_trials, 
                   const int & rank, 
                   const int & size) {

    // Initialization phase
    int local_num_bees = num_bees / size;
    std::mt19937 generator(rank);
    
    std::vector<double> init_list(dim);
    std::vector<std::vector<double>> local_current_positions(local_num_bees, init_list);
    initialize_positions(local_current_positions, generator);

    std::vector<double> local_curr_OF_value(local_num_bees, std::numeric_limits<double>::max());
    std::vector<int> trials(local_num_bees, 0);

    std::vector<std::vector<double>> local_best_positions(local_num_bees, init_list);
    std::vector<double> local_best_OF_value(local_num_bees, std::numeric_limits<double>::max());

    evaluate_and_update(local_current_positions, local_current_positions, local_curr_OF_value, trials, rank);

    // Main loop
    for (std::size_t iter = 0; iter < iter_num; ++iter){

        // Employed bees phase
        employed_bee_phase(local_current_positions, local_curr_OF_value, trials, local_num_bees, num_bees, rank, generator);

        // Onlookers bees phase
        onlookers_bee_phase(local_current_positions, local_curr_OF_value, trials, local_num_bees, num_bees, rank, generator);
        
        // Check of the best position
        for (std::size_t local_bee = 0; local_bee < local_num_bees; ++local_bee){
            if (local_curr_OF_value[local_bee] < local_best_OF_value[local_bee]){
                local_best_positions[local_bee] = local_current_positions[local_bee];
                local_best_OF_value[local_bee] = local_curr_OF_value[local_bee];
            }
        }

        // Scout bees phase
        scout_bee_phase(local_current_positions, local_best_positions, local_curr_OF_value, local_best_OF_value, trials, local_num_bees, max_trials, generator);
    }

    // Print the best
    print_best(local_best_positions, local_best_OF_value, local_num_bees, num_bees, rank);
}