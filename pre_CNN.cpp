// !start
// implement pre_CNN
#include "pre_CNN.h"

// @brief   default constructor function
CNN_Part::CNN_Part() {
    this->fp_num = 0;
    this->featureMaps.clear();
    this->filter_pools.clear();
}

// @brief   destruction function
CNN_Part::~CNN_Part() {
    this->featureMaps.clear();
    this->filter_pools.clear();
}

// @brief   read input file
void CNN_Part::read_input(std::ifstream& inFile) {
    assert(inFile.good());
    this->input_layer.inputFile(inFile);
}

// @brief   add layer
void CNN_Part::add_filter_pool(bool is_pool, int size, int D, int N) {
    //< check
    assert(size <= MAX_WH);
    this->filter_pools.push_back(Filters());
    (*this->filter_pools.rbegin()).is_pool = is_pool;
    if (is_pool) {
        (*this->filter_pools.rbegin()).pool_size = size;
        (*this->filter_pools.rbegin()).fils.clear();
    }
    else {
        (*this->filter_pools.rbegin()).N = N;
        For_(iter, 0, N) {
            (*this->filter_pools.rbegin()).fils.push_back(Maps(D, size));
        }
    }

    //< add featureMaps
    this->featureMaps.push_back(Maps());
    this->fp_num++;
}

// @brief   get last featureMap
Maps& CNN_Part::get_last_map() {
    assert(fp_num);
    return (*this->featureMaps.rbegin());
}

// @brief   forward CNN
void CNN_Part::Forward_CNN(int stride) {
    //< check
    assert(this->featureMaps.size() == this->filter_pools.size());
    assert(this->featureMaps.size() == this->fp_num);

    For_(iter, 0, this->fp_num) {
        if (this->filter_pools[iter].is_pool) {
            this->featureMaps[iter].Mean_pooling(this->featureMaps[iter - 1], this->filter_pools[iter].pool_size);
        }
        else {
            this->featureMaps[iter].Forward_(this->featureMaps[iter - 1], this->filter_pools[iter], stride);
        }
    }
}

// @brief   backward CNN
void CNN_Part::Backward_CNN(double rate) {
    //< call this function after Forward_CNN
    assert(this->featureMaps.size() == this->filter_pools.size());
    assert(this->featureMaps.size() == this->fp_num);
    
    _For(iter, this->fp_num - 2, 0) {
        if (this->filter_pools[iter + 1].is_pool) {
            int size = this->filter_pools[iter + 1].pool_size;
            this->featureMaps[iter].Mean_pooling_delta(this->featureMaps[iter + 1], size);
        }
        else {
            this->featureMaps[iter].calc_gradient(this->featureMaps[iter + 1], this->filter_pools[iter + 1]);
        }
    }
    //< input layer
    this->input_layer.calc_gradient(this->featureMaps[0], this->filter_pools[0]);

    //< update weight
    For_(iter, 0, this->fp_num) {
        if (this->filter_pools[iter].is_pool)
            continue;
        this->filter_pools[iter].update(rate);
    }
}