#pragma once

#include <vector>
#include "Data/Data.hpp"

namespace Split {

    struct TrainTestSplit {
        Dataframe X_train;
        Dataframe X_test;
        Dataframe y_train;
        Dataframe y_test;        
    };

    struct TrainTestValidSplit {
        Dataframe X_train;
        Dataframe X_valid;
        Dataframe X_test;
        Dataframe y_train;
        Dataframe y_valid;
        Dataframe y_test; 
    };

    // Classical Train Test split
    TrainTestSplit train_test_split(const Dataframe& x, const Dataframe& y, int proportion = 80);

    // Classical Train Test Valid split, proportion pair take % of Train and Valid 
    TrainTestValidSplit train_test_split(const Dataframe& x, const Dataframe& y, const std::pair<int, int> proportion = {60, 20});
    
    // Stratified split on y 
    TrainTestSplit stratified_split(const Dataframe& x, const Dataframe& y, int proportion = 80);
    
    // Stratified split on y, proportion pair take % of Train and Valid 
    TrainTestValidSplit stratified_split(const Dataframe& x, const Dataframe& y, const std::pair<int, int> proportion = {60, 20});
}