#pragma once
#ifndef DNN
#define DNN
#include "pre_CNN.h"

class DNN_Part {
public:
    DNN_Part();
    ~DNN_Part();

    //< file read
    void read_param(std::ifstream&);
    //< file save
    void save_param(std::ofstream&);

    //< start
    void set_input(Maps&);
    //< add layer nodes int and weight maps
    void add_layer(int);

    //< read target on handwritten digit
    void set_tar(int);
    //< get target after Forward DNN
    int get_res();
    int get_tar() const;

    //< call after set all
    void Forward_DNN();

    //< call after get tar and Forward_DNN
    //< calc delta and update weights
    void Backward_DNN(double);

    //< get first map
    Maps& get_first_map();

private:
    std::vector<Maps> __layers;
    std::vector<Maps> __weights;
    int layer_num;

    //< for handwriten digit rec, __tar means the target digit from 0 to 9
    int __tar; 

    //< Ed
    double __Ed;
};
#endif // !DNN

