// !start
// implement "Feature_Filter.h"
#include "Feature_Filter.h"

// @brief   default constructor function
Maps::Maps() {
    this->maps.clear();
    this->__D = this->__H = 0;
    this->__W = 0;
    this->__bias = 0.0;
}

// @brief   other constructor function for filters
Maps::Maps(int D, int F) {
    this->__D = D;
    this->__H = this->__W = F;
    this->__bias = 0;
    this->maps.clear();
    For_(iter, 0, D) {
        this->maps.push_back(Matrix(F, F));
    }
}

// @brief   D H W
Maps::Maps(int D, int H, int W) {
    this->__D = D;
    this->__H = H;
    this->__W = W;
    this->__bias = 0;
    this->maps.clear();
    For_(iter, 0, D) {
        this->maps.push_back(Matrix(H, W));
    }
}

// @brief   copy constructor function
Maps::Maps(const Maps& obj) {
    this->maps.clear();
    For_(iter, 0, obj.__D) {
        this->maps.push_back(obj.maps[iter]);
    }
    this->__D = obj.__D;
    this->__H = obj.__H;
    this->__W = obj.__W;
    this->__bias = obj.__bias;
}

// @brief   destructor function
Maps::~Maps() {
    this->maps.clear();
}

// @brief   read input layer
// @format  
//  H W
// x x x for W
// x x x ...
// . . .
// for H
void Maps::inputFile(std::ifstream& inFile) {
    this->__D = 1;
    this->maps.clear();
    //< check file open
    assert(inFile.good());
    //< use Matrix::read_file
    this->maps.push_back(Matrix());
    (*maps.rbegin()).read_file(inFile);
    this->__H = (*maps.rbegin()).get_H();
    this->__W = (*maps.rbegin()).get_W();
}

// @brief   read param for filters
// @format 
//  D bias
//  H W
//  xxx...
//  H W
//  xxx...
void Maps::paramFile(std::ifstream& inFile) {
    this->maps.clear();
    //< check file open
    assert(inFile.good());
    //< read Depth
    inFile >> this->__D >> this->__bias;
    For_(dep, 0, this->__D) {
        this->maps.push_back(Matrix());
        (*maps.rbegin()).read_file(inFile);
    }
    this->__H = (*maps.rbegin()).get_H();
    this->__W = (*maps.rbegin()).get_W();
}

// @brief   param saving
void Maps::save_param(std::ofstream& outFile) {
    //< check
    assert(outFile.good());
    outFile << this->__D << " " << this->__bias << std::endl;
    For_(Dd, 0, this->__D) {
        this->maps[Dd].save_file(outFile);
    }
}

// @brief   copy map for sec
void Maps::copy_map(Matrix& mat, int depth) {
    //< check
    assert(this->__H == mat.get_H() && this->__W == mat.get_W() && depth < this->__D && depth >= 0);
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            double val = mat.get_value_sec(row, col);
            this->maps[depth].set_value_sec(row, col, val);
        }
    }
}

// @brief   Forward
// @param   featureMap from n-1 layer
// @param   filters    N filters
void Maps::Forward_(Maps& featureMap, Filters& filters, int stride, int zero_padding) {
    //< check operation valid
    assert(featureMap.get_D() == (*filters.fils.begin()).get_D());
    int depth = featureMap.get_D();
    int num = filters.N;
    //< clear
    this->maps.clear();
    For_(iter, 0, num) {
        //default stride = 1
        this->maps.push_back(Matrix());
        //< for each input map
        //< first different from other
        For_(dp, 0, depth) {
            bool mode = (dp != 0);
            //< zero padding
            Matrix feat = featureMap.get_depth(dp);
            Matrix filt = filters.fils[iter].get_depth(dp);

            feat.padding_0(zero_padding);
            (*this->maps.rbegin()).cross_correlation(feat, filt, stride, filters.fils[iter].get_bias(), mode, false, false, false);
        }
        //< go through Relu
        (*this->maps.rbegin()).filt_Relu();
    }
    this->__D = num;
    this->__H = this->maps[0].get_H();
    this->__W = this->maps[0].get_W();
}

// @brief   mean pooling layer
void Maps::Mean_pooling(Maps& featureMap, int size) {
    //< check operation valid
    assert(!(featureMap.get_H() % size) && !(featureMap.get_W() % size));
    
    //< clear and reset
    this->maps.clear();
    this->__D = featureMap.get_D();
    this->__H = featureMap.get_H() / size;
    this->__W = featureMap.get_W() / size;

    //< begin pooling
    For_(dp, 0, this->__D) {
        this->maps.push_back(Matrix());
        (*this->maps.rbegin()).mean_pooling(featureMap.get_depth(dp), size);
    }
}

// @brief   for forward process transform
void Maps::fp_transform(Maps& featureMap) {
    //< clear and reset
    this->maps.clear();
    int D = featureMap.get_D();
    int H = featureMap.get_H();
    int W = featureMap.get_W();
    this->__D = 1;
    this->__W = 1;
    this->__H = D * H * W;
    this->maps.push_back(Matrix());
    (*this->maps.rbegin()).set_W(this->__W);
    (*this->maps.rbegin()).set_H(this->__H);

    For_(dd, 0, D) {
        For_(dh, 0, H) {
            For_(dw, 0, W) {
                this->maps[0].set_value_fir(dd * H * W + dh * W + dw, 0, \
                    featureMap.get_value_fir(dd, dh, dw));
            }
        }
    }
}

// @brief   for back process transform
void Maps::bp_transform(Maps& layer) {
    //< check
    assert(layer.get_D() == 1 && layer.get_W() == 1 && layer.get_H() - 1 == this->__D * this->__H * this->__W);
    
    For_(dd, 0, this->__D) {
        For_(dh, 0, this->__H) {
            For_(dw, 0, this->__W) {
                this->maps[dd].set_value_sec(dh, dw, \
                    layer.get_value_sec(0, dd * this->__H * this->__W + dh * this->__W + dw, 0));
            }
        }
    }
}

// @brief   mean pooling delta
//          featureMap is the next layer
void Maps::Mean_pooling_delta(Maps& featureMap, int size) {
    //< check
    assert(!(this->__H % featureMap.get_H()) && (this->__H / featureMap.get_H() == size));
    assert(!(this->__W % featureMap.get_W()) && (this->__W / featureMap.get_W() == size));
    assert(this->__D == featureMap.get_D());

    int featH = featureMap.get_H();
    int featW = featureMap.get_W();
    For_(Dd, 0, this->__D) {
        //< for each in featureMap
        For_(row, 0, featH) {
            For_(col, 0, featW) {
                int tarRow = row * size;
                int tarCol = col * size;
                double val = featureMap.get_value_sec(Dd, row, col);
                For_(dr, tarRow, tarRow + size) {
                    For_(dc, tarCol, tarCol + size) {
                        this->maps[Dd].set_value_sec(dr, dc, val / (size * size));
                    }
                }
            }
        }
    }
}

// @brief   backward one convertion layer
void Maps::calc_gradient(Maps& featureMap, Filters& filters, int stride) {
    //< check
    assert(this->__D == filters.fils[0].get_D());
    assert(filters.N == featureMap.get_D());

    //< calc P
    int Padding = filters.fils[0].get_H() - 1;

    //< for each map on featureMap
    int lay = featureMap.get_D();
    For_(Dn, 0, lay) {
        Matrix curr_map = featureMap.get_depth(Dn);
        //< padding
        curr_map.padding_0(Padding);
        For_(Dd, 0, this->__D) {
            bool mode = (Dd != 0);
            Matrix filt = filters.fils[Dn].get_depth(Dd);
            filt.rotate_180();
            this->maps[Dd].cross_correlation(curr_map, filt, stride, 0, mode, true, false, true);
        }
    }
    //< mul f^{l-1}()
    For_(Dd, 0, this->__D) {
        For_(Dr, 0, this->__H) {
            For_(Dc, 0, this->__W) {
                if (!this->maps[Dd].get_value_fir(Dr, Dc)) {
                    this->maps[Dd].set_value_sec(Dr, Dc, 0.0);
                }
            }
        }
    }
    
    //< calc grandient
        //< Wb
    int featH = featureMap.get_H();
    int featW = featureMap.get_W();
    For_(Dn, 0, lay) {
        double sum = 0;
        For_(Dr, 0, featH) {
            For_(Dc, 0, featW) {
                sum += featureMap.get_value_sec(Dn, Dr, Dc);
            }
        }
        filters.fils[Dn].set_bias(sum);
    }

        //< Wi,j
        //< every filter
    For_(Dn, 0, lay) {
        Matrix curr_map = featureMap.get_depth(Dn);
        For_(Dd, 0, this->__D) {
            Matrix res_map;
            res_map.cross_correlation(this->maps[Dd], curr_map, stride, 0, false, false, true, true);
            filters.fils[Dn].copy_map(res_map, Dd);
        }
    }
}

// @brief   update weight w.fir -= rate * w.sec
void Maps::update_weight(double rate) {
    For_(Dd, 0, this->__D) {
        For_(Dr, 0, this->__H) {
            For_(Dc, 0, this->__W) {
                double fir_val = this->maps[Dd].get_value_fir(Dr, Dc);
                double sec_val = this->maps[Dd].get_value_sec(Dr, Dc);
                this->maps[Dd].set_value_fir(Dr, Dc, fir_val - rate * sec_val);
            }
        }
    }
}

// @brief   update weight for DNN
void Maps::update_weight_DNN(Maps& Front, Maps& Back, double rate) {
    //< make sure this->maps is weight between Front and Back
    //< check
    //assert(Back.get_H() == this->__W);

    For_(row, 0, this->__H) {
        double delta = Back.get_value_sec(0, row, 0);
        For_(col, 0, this->__W) {
            double Xput = Front.get_value_fir(0, col, 0);
            //< update
            double origin = this->maps[0].get_value_fir(row, col);
            origin += rate * Xput * delta;
            this->maps[0].set_value_fir(row, col, origin);
        }
    }
}

// @brief   append 1 line on height
void Maps::append(bool signal, double val) {
    //< check
    assert(this->__D == 1 && this->__W == 1);
    this->maps[0].append(signal, val);
    (signal ? this->__W++ : this->__H++);
}

// @brief   for DNN forward
void Maps::Forward_DNN(Maps& Weight, Maps& Aput) {
    //< check
    assert(Weight.get_D() == 1 && Aput.get_D() == 1 && Aput.get_W() == 1);
    assert(Weight.get_W() == Aput.get_H() + 1);
    //< clear and reset
    this->__D = 1;
    this->__W = 1;
    this->__H = Weight.get_H();
    if (this->maps.size() != 1) {
        this->maps.clear();
        this->maps.push_back(Matrix());
    }
    //< append
    Aput.append();
    //< product and sigmoid
    this->maps[0].mult(Weight.get_depth(0), Aput.get_depth(0));
    //< go through sigmoid
    this->maps[0].filt_sigmoid();
}

// @brief   calc delta for output layer
void Maps::calc_delta_DNN_output(double* target, int len, double& Ed) {
    //< check
    assert(len == this->__H && this->__D == 1 && this->__W == 1);
    Ed = 0;
    For_(row, 0, this->__H) {
        double Yval = this->maps[0].get_value_fir(row, 0);
        Ed += pow(Yval - target[row], 2) / 2.0;
        this->maps[0].set_value_sec(row, 0, Yval * (1 - Yval) * (target[row] - Yval));
    }
}

// @brief   calc delta for hidden layer
void Maps::calc_delta_DNN_hidden(Maps& Weight, Maps& Layer) {
    //< check
    assert(this->__W == 1 && this->__D == 1 && this->__H == Weight.get_W());
    assert(Layer.get_W() == 1 && Layer.get_D() == 1);
    //< math:
    int Weight_H = Weight.get_H();
    For_(row, 0, this->__H - 1) {
        double Alit = this->maps[0].get_value_fir(row, 0);
        double sum = 0;
        For_(row_in_weight, 0, Weight_H) {
            sum += Weight.get_value_fir(0, row_in_weight, row) * Layer.get_value_sec(0, row_in_weight, 0);
        }
        this->maps[0].set_value_sec(row, 0, Alit * (1 - Alit) * sum);
    }
}

// @brief   print information for debug
void Maps::print() const {
    printf("\nthis->__D = %d, this->__H = %d, this->__W = %d\n", \
        this->__D, this->__H, this->__W);
    For_(dep, 0, this->__D) {
        printf("depth: %d\n", dep);
        this->maps[dep].print();
    }
}

// @brief   get elements
int Maps::get_D() const {
    return this->__D;
}

int Maps::get_H() const {
    return this->__H;
}

int Maps::get_W() const {
    return this->__W;
}

double Maps::get_bias() const {
    return this->__bias;
}

void Maps::set_bias(double bias) {
    this->__bias = bias;
}

Matrix& Maps::get_depth(int depth) {
    //< check
    assert(depth < this->__D && depth >= 0);
    return this->maps[depth];
}

double Maps::get_value_fir(int dep, int hei, int wid) const {
    //< check
    assert(dep < this->__D && hei < this->__H && wid < this->__W);
    assert(dep >= 0 && hei >= 0 && wid >= 0);
    return this->maps[dep].get_value_fir(hei, wid);
}

double Maps::get_value_sec(int dep, int hei, int wid) const {
    //< check
    assert(dep < this->__D && hei < this->__H && wid < this->__W);
    assert(dep >= 0 && hei >= 0 && wid >= 0);
    return this->maps[dep].get_value_sec(hei, wid);
}

void Maps::set_value_fir(int D, int H, int W, double value) {
    assert(D < this->__D && H < this->__H && W < this->__W);
    assert(D >= 0 && H >= 0 && W >= 0);
    this->maps[D].set_value_fir(H, W, value);
}

void Maps::set_H_W(int H, int W) {
    this->maps.clear();
    this->__D = 1;
    this->__H = H;
    this->__W = W;
}

//***********************************//
Filters::Filters() {
    this->fils.clear();
    this->N = 0;
    this->pool_size = 0;
    this->is_pool = false;
}

Filters::~Filters() {
    this->fils.clear();
}

// @brief   for a layer
// @format 
// N
// D bias
// H W
// ...
// H W
// ...
// D bias
// ...
void Filters::readFile(std::ifstream& inFile) {
    this->fils.clear();
    //< check file open
    assert(inFile.good());
    inFile >> this->N;
    //< for every maps
    For_(iter, 0, this->N) {
        this->fils.push_back(Maps());
        (*this->fils.rbegin()).paramFile(inFile);
    }
}

// @brief   print
void Filters::print() const {
    printf("\nthis layer has %d filters\n", this->N);
    For_(iter, 0, this->N) {
        printf("No.%d filter:\n", iter);
        this->fils[iter].print();
    }
}

// @brief   update
void Filters::update(double rate) {
    For_(Dn, 0, this->N) {
        this->fils[Dn].update_weight(rate);
    }
}

// @brief   for Filters init
void Filters::rand_create(int N, int D, int F) {
    this->N = N;
    this->fils.clear();
    For_(iter, 0, N) {
        this->fils.push_back(Maps(D, F));
    }
}