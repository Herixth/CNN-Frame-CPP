// !start
// implement DNN_.h
#include "DNN_.h"

// @brief   default constructor function
DNN_Part::DNN_Part() {
    this->layer_num = 0;
    this->__Ed = 0;
    this->__layers.clear();
    this->__weights.clear();
}

// @brief   destructor function
DNN_Part::~DNN_Part() {
    this->__layers.clear();
    this->__weights.clear();
}

// @brief   read file
void DNN_Part::read_param(std::ifstream& inFile) {
    //< after set and add
    For_(iter, 0, this->layer_num - 1) {
        this->__weights[iter].paramFile(inFile);
    }
}

// @brief   save file
void DNN_Part::save_param(std::ofstream& outFile) {
    //< after train
    For_(iter, 0, this->layer_num - 1) {
        this->__weights[iter].save_param(outFile);
    }
}

// @brief   get input from CNN part
void DNN_Part::set_input(Maps& featureMap) {
    assert(this->layer_num);
    //< transform
    (*this->__layers.begin()).fp_transform(featureMap);
}

// @brief    add layer and weight
//           weight[front + 1, curr]
void DNN_Part::add_layer(int num) {
    //< new weight
    if (this->layer_num) {
        int new_row = (*this->__layers.rbegin()).get_H() + 1;
        this->__weights.push_back(Maps(1, num, new_row));
    }
    
    //< new layer
    this->__layers.push_back(Maps(1, num, 1));
    this->layer_num++;
}   

// @brief   read digit
void DNN_Part::set_tar(int tar) {
    this->__tar = tar;
}

// @brief   get tar
int DNN_Part::get_tar() {
    //< be sure this function called after Forward_DNN
    int tar = 0;
    int num_out = (*this->__layers.rbegin()).get_H();

#ifdef DEBUG
    printf("tar:\t");
    For_(inc, 0, num_out) {
        printf("%d%c", inc, "\t\n"[inc == num_out - 1]);
    }
    printf("val: ");
#endif // DEBUG


    For_(iter, 0, num_out) {
        double val = (*this->__layers.rbegin()).get_value_fir(0, iter, 0);
        double rem = (*this->__layers.rbegin()).get_value_fir(0, tar, 0);
        
#ifdef DEBUG
        printf("%.05lf%c", val, " \n"[iter == num_out - 1]);
#endif // DEBUG

        if (val > rem) {
            tar = iter;
        }
    }

#ifdef DEBUG
    printf("Ed>> %.3lf\n", this->__Ed);
#endif // DEBUG


    return tar;
}

// @brief   just go forward
void DNN_Part::Forward_DNN() {
    //< call this function after set_input() and set_tar()
    assert(this->layer_num && this->__weights.size() + 1 == this->__layers.size());
    For_(curr, 0, this->layer_num - 1) {
        this->__layers[curr + 1].Forward_DNN(this->__weights[curr], this->__layers[curr]);
    }
}

// @brief   calc delta and update weights
void DNN_Part::Backward_DNN(double rate) {
    //< calc delta
    int num_out = this->__layers[this->layer_num - 1].get_H();
    double *targ_ = new double[num_out];
    For_(iter, 0, num_out) {
        targ_[iter] = (iter == this->__tar ? 1.0 : 0.0);
    }
    //< calc output layer
    this->__layers[this->layer_num - 1].calc_delta_DNN_output(targ_, num_out, this->__Ed);

    delete targ_;

    //< calc hidden layer
    _For(iter, this->layer_num - 2, 0) {
        this->__layers[iter].calc_delta_DNN_hidden(this->__weights[iter], this->__layers[iter + 1]);
    }

    // update weight
    For_(iter, 0, this->layer_num - 1) {
        this->__weights[iter].update_weight_DNN(this->__layers[iter], this->__layers[iter + 1], rate);
    }
}

// @brief   get first map
Maps& DNN_Part::get_first_map() {
    assert(this->layer_num);
    return this->__layers[0];
}