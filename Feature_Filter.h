#pragma once
#ifndef FEATURE_FILTER
#define FEATURE_FILTER
//base
#include "Matrix.h"

struct Filters {
    Filters();
    ~Filters();
    //< read layer
    void readFile(std::ifstream&);

    //< save layer
    void saveFile(std::ofstream&);

    //< not read file
    //< N D F
    void rand_create(int, int, int);

    //< just for debug
    void print() const;

    //< update after calc delta
    void update(double);

    //< elements
    std::vector<Maps> fils;
    int N;
    
    int pool_size;
    bool is_pool;
};

class Maps {
public:
    Maps();
    Maps(int, int);
    Maps(int, int, int);
    Maps(const Maps&);
    ~Maps();

    //< file operation
    void inputFile(std::ifstream&);
    void paramFile(std::ifstream&);

    //< param saving
    void save_param(std::ofstream&);

    //< get elements
    int get_D() const;
    int get_H() const;
    int get_W() const;
    double get_bias() const;
    //< D
    Matrix& get_depth(int);
    //< get value
    double get_value_fir(int, int, int) const;
    double get_value_sec(int, int, int) const;

    //< set value
    void set_value_fir(int, int, int, double);

    //< for init input layer in CNN_Part
    void set_H_W(int, int);

    //< set bias
    void set_bias(double);

    //< copy map 
    //< -#param depth[int] for sec
    void copy_map(Matrix&, int);
    
    //< feature maps -> filters[N, D, F]
    //< @param stride[int]
    //< @param zero_padding[int]: default 0
    void Forward_(Maps&, Filters&, int, int = 0);

    //< mean_pooling
    //< @param size[int] the square size
    void Mean_pooling(Maps&, int);

    //< transform from featureMaps to Layer
    void fp_transform(Maps&);
    void bp_transform(Maps&);

    //< mean_pooling delta
    void Mean_pooling_delta(Maps&, int);

    //< backward pre
    //< @param stride default 1
    void calc_gradient(Maps&, Filters&, double, int = 1);

    //< backward update weight
    //< just for filter
    //< rate
    void update_weight(double);

    //< for DNN
    //< -#param Front
    //< -#param Back
    void update_weight_DNN(Maps&, Maps&, double);

    //< { all for Layer function
        //< append
    void append(bool = false, double = 1.0);

        //< -#param Weight
        //< -#param Aput
    void Forward_DNN(Maps&, Maps&);

        //< calc delta only for output layer
        //< -#param target[double*]
        //< -#param len[int] for target number and check
    void calc_delta_DNN_output(double*, int, double&);
        //< calc delta in hidden layer
        //< -#param Weight
        //< -#param Layer
    void calc_delta_DNN_hidden(Maps&, Maps&);
    //< }

    //< print maps for debug
    void print() const;
private:
    std::vector<Matrix> maps;
    int __D;
    int __H;
    int __W;
    double __bias;
};
#endif // !FEATURE_FILTER
