#include "algorithm.cc"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initial setup
    int num_bees;
    int iter_num;
    int max_trials;
    
    if (argc == 4){
        num_bees = std::stoi(argv[1]);
        iter_num = std::stoi(argv[2]);
        max_trials = std::stoi(argv[3]);

        abc_algorithm(num_bees, iter_num, max_trials, rank, size);
    }
    else {
        if (rank == 0)
            std::cerr << "Please, provide 1. number of bees, 2. iterations, 3. max_trials... " << std::endl;
    }

    MPI_Finalize();
    return 0;
}

