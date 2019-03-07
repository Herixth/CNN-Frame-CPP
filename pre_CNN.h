#pragma once
#ifndef PRE_CNN
#define PRE_CNN

#include "Feature_Filter.h"

class CNN_Part {
public:
    CNN_Part();
    ~CNN_Part();

    //< read input
    void read_input(std::ifstream&);

    //< add filters or pools
    //< <em> false </em> is_pool   int size int N
    //< <em> true </em> is_pool   int size  N = 0
    void add_filter_pool(bool, int, int = 0, int = 0);

    //< get last map
    Maps& get_last_map();

    //< Forward stride
    void Forward_CNN(int);

    //< Backward
    void Backward_CNN(double);
private:
    //< only input layer
    Maps input_layer;
    //< size of filter_pools and size of featureMaps is the same
    std::vector<Filters> filter_pools;
    std::vector<Maps> featureMaps;
    //< the size of filter_pools
    int fp_num;
};

#endif // !PRE_CNN
