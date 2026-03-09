#pragma once

#include <vector>

class Dataframe;

namespace Kmeans { 

    // Struct for Kmeans methods, k for nb_clusters used, labels_ for the clusters
    // If method Optimal_Kmeans was used, opt_k for the Optimal k by using elbow method
    // inertia_ to check values 
    struct KmeansRes {
        int k;                        // Choosen k      
        std::vector<double> labels_;  // Clusters
        std::vector<double> cluster_centers_;   

        // If method Optimal_Kmeans was used
        int opt_k;                      // Optimal k by using elbow method  
        std::vector<double> inertia_;   // To get optimal k using elbow method
       
    };

    // Kmeans function using initialisation "Kmeans++" to have a better start and results
    // n_init to perform multiple times Kmeans, will choose the best one
    // optimal_k to run function Optimal_Kmeans to have the best k using elbow method (will overwrite your nb_clusters param)
    KmeansRes Kmeans(const Dataframe& x, int nb_clusters, int n_init = 10, bool optimal_k = false);

    // Function to have the best k using elbow method
    // n_init to perform multiple times Kmeans, will choose the best one
    void Optimal_Kmeans(const Dataframe& x, int n_init = 10);
}