#pragma once
#ifndef PRE_CNN
#define PRE_CNN

#include "Feature_Filter.h"

class CNN_Part {
public:
    CNN_Part();
    ~CNN_Part();

    //< read input
    //< re write
    void read_input(std::ifstream&);

    //< add input layer
    //< H W
    void add_input_layer(int, int);

    //< add filters or pools
    //< <em> false </em> is_pool   int size int D int N
    //< <em> true </em> is_pool   int size int D = 0, N = 0
    void add_filter_pool(bool, int, int = 0, int = 0);

    //< get last map
    Maps& get_last_map();

    //< set last map
    void set_last_map(Maps&);

    //< Forward stride
    void Forward_CNN(int = 1);

    //< Backward
    void Backward_CNN(double);

    //< param saving 
    void save_param(std::ofstream&);

    //< read param
    void read_param(std::ifstream&);
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
