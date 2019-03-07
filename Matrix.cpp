// !start
// implement "Matrix.h"
#include "Matrix.h"

// @brief   default constructor function
Matrix::Matrix() {
    this->matr = new Dpd*[MAX_WH];
    For_(iter, 0, MAX_WH) {
        this->matr[iter] = new Dpd[MAX_WH];
    }
    //< __H and __W
    this->__H = this->__W = MAX_WH;

    std::srand(unsigned int(std::time(NULL)));

    For_(inc, 0, this->__H) {
        For_(snc, 0, this->__W) {
            this->matr[inc][snc] = Dpd(double(std::rand()) / RAND_MAX - 0.5, 0.0);
        }
    }
}

// @brief   other constructor function
Matrix::Matrix(int mH, int mW) {
    this->matr = new Dpd*[MAX_WH];
    For_(iter, 0, MAX_WH) {
        this->matr[iter] = new Dpd[MAX_WH];
    }
    //< __H and __W
    this->__H = mH;
    this->__W = mW;

    For_(inc, 0, this->__H) {
        For_(snc, 0, this->__W) {
            this->matr[inc][snc] = Dpd(double(std::rand()) / RAND_MAX - 0.5, 0.0);
        }
    }
}

// @brief   copy constructor function
Matrix::Matrix(const Matrix& obj) {
    //< delete
    this->matr = new Dpd*[MAX_WH];
    For_(iter, 0, MAX_WH) {
        this->matr[iter] = new Dpd[MAX_WH];
    }
    //< __H and __W
    this->__H = obj.get_H();
    this->__W = obj.get_W();
    //< do a copy
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            this->matr[row][col] = \
                Dpd(obj.get_value_fir(row, col), obj.get_value_sec(row, col));
        }
    }
}

// @brief   destructor function
Matrix::~Matrix() {
    For_(iter, 0, MAX_WH) {
        delete[] this->matr[iter];
    }
    delete[] this->matr;
}

// @brief   get row_max
int Matrix::get_H() const {
    return this->__H;
}

// @brief   get col_max
int Matrix::get_W() const {
    return this->__W;
}

// @brief   set row_max
void Matrix::set_H(int H) {
    //< check
    assert(H <= MAX_WH && H > 0);
    this->__H = H;
}

// @brief   set col_max
void Matrix::set_W(int W) {
    assert(W <= MAX_WH && W > 0);
    this->__W = W;
}

// @brief   get value fir
double Matrix::get_value_fir(int row, int col) const {
    assert(row < this->__H && col < this->__W && row >= 0 && col >= 0);
    return this->matr[row][col].first;
}

// @brief   get value sec
double Matrix::get_value_sec(int row, int col) const {
    assert(row < this->__H && col < this->__W && row >= 0 && col >= 0);
    return this->matr[row][col].second;
}

// @brief   set value fir
void Matrix::set_value_fir(int row, int col, double val) {
    assert(row < this->__H && col < this->__W && row >= 0 && col >= 0);
    this->matr[row][col].first = val;
}

// @brief   set value sec
void Matrix::set_value_sec(int row, int col, double val) {
    assert(row < this->__H && col < this->__W && row >= 0 && col >= 0);
    this->matr[row][col].second = val;
}

// @brief   rotate matr 180бу
void Matrix::rotate_180() {
    int half = int(std::floor(1.0 * this->__H * this->__W / 2));
    For_(cnt, 0, half) {
        int row = cnt / this->__W;
        int col = cnt % this->__W;
        std::swap(this->matr[row][col], this->matr[this->__H - 1 - row][this->__W - 1 - col]);
    }
}

// @brief   padding zeros
void Matrix::padding_0(int P) {
    const Dpd Zero_DPD = Dpd(0.0, 0.0);
    assert(2 * P + this->__H <= MAX_WH && 2 * P + this->__W <= MAX_WH && P >= 0);
    this->__H += 2 * P;
    this->__W += 2 * P;
    //< begin moving
    _For(row, this->__H - 1, P) {
        _For(col, this->__W - 1, P) {
            this->matr[row][col] = this->matr[row - P][col - P];
        }
    }
    //< begin padding
    For_(row, 0, this->__H) {
        if (row < P || row > this->__H - P) {
            For_(col, 0, this->__W) {
                this->matr[row][col] = Zero_DPD;
            }
        }
        else {
            For_(col, 0, P) {
                this->matr[row][col] = Zero_DPD;
            }
            For_(col, this->__W - P, this->__W) {
                this->matr[row][col] = Zero_DPD;
            }
        }
    }
}

// @brief   depadding around matrix
void Matrix::depadding(int P) {
    assert(2 * P < this->__H && 2 * P < this->__W && P >= 0);
    this->__H -= 2 * P;
    this->__W -= 2 * P;

    //< begin moving
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            this->matr[row][col] = this->matr[row + P][col + P];
        }
    }
}

// @brief   convolution operation
// @param   mode
//          -<em>false</em>     add bias, replace matrix
//          -<em>true</em>      do not add bias, append value in matrix
// @param   *_tar
//          -<em>false</em>     use pair.first
//          -<em>false</em>     use pair.second
void Matrix::cross_correlation(const Matrix& feature_map, const Matrix& filter, int stride, int bias, bool mode,\
                                bool feat_tar, bool filt_tar, bool recv_tar) {
    //< filter must be squard
    assert(filter.get_H() == filter.get_W());
    int feam_H = feature_map.get_H();
    int feam_W = feature_map.get_W();
    int filt_F = filter.get_H();
    //< operation must be valid
    assert(!((feam_H - filt_F) % stride) && !((feam_W - filt_F) % stride) && stride > 0);
    this->__H = (feam_H - filt_F) / stride + 1;
    this->__W = (feam_W - filt_F) / stride + 1;
    //< notice overflow
    assert(this->__H > 0 && this->__W > 0 && this->__H <= MAX_WH && this->__W <= MAX_WH);

    //< for next feature map
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            int tar_row = stride * row;
            int tar_col = stride * col;
            int sum = 0;
            //< calc convolution
            For_(inc, 0, filt_F) {
                For_(snc, 0, filt_F) {
                    int tar_r = tar_row + inc;
                    int tar_c = tar_col + snc;
                    sum += int((feat_tar ? feature_map.get_value_sec(tar_r, tar_c) : feature_map.get_value_fir(tar_r, tar_c)) *\
                        (filt_tar ? filter.get_value_sec(inc, snc) : filter.get_value_fir(inc, snc)));
                }
            }
            if (mode) {
                (recv_tar ? this->matr[row][col].second : this->matr[row][col].first) += sum;
            }
            else {
                (recv_tar ? this->matr[row][col].second : this->matr[row][col].first) = sum + bias;
            }
        }
    }
}

// @brief   do mean pooling
void Matrix::mean_pooling(const Matrix& mat, const int size) {
    //< already checked
    assert(!(mat.get_H() % size) && !(mat.get_W() % size));
    //< clear and reset
    this->__H = mat.get_H() / size;
    this->__W = mat.get_W() / size;

    //< do pooling
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            //< clearfy tar
            int tar_r = row * size;
            int tar_c = col * size;
            double sum = 0;
            For_(dr, tar_r, tar_r + size) {
                For_(dc, tar_c, tar_c + size) {
                    sum += mat.get_value_fir(dr, dc);
                }
            }
            this->matr[row][col] = Dpd(sum / (size * size), 0.0);
        }
    }
}

// @brief   matrix product
void Matrix::mult(const Matrix& matA, const Matrix& matB) {
    //< check valid
    assert(matA.get_W() == matB.get_H());
    this->__H = matA.get_H();
    this->__W = matB.get_W();

    int mat_s = matA.get_W();
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            double sum = 0.0;
            For_(rd, 0, mat_s) {
                sum += matA.get_value_fir(row, rd) * matB.get_value_fir(rd, col);
            }
            this->matr[row][col] = Dpd(sum, 0.0);
        }
    }
}

// @brief   append line
void Matrix::append(bool signal, double value) {
    //< check
    assert((this->__H + 1 <= MAX_WH && !signal) || (this->__W + 1 <= MAX_WH && signal));
    signal ? this->__W++ : this->__H++;
    For_(inc, 0, (signal ? this->__H : this->__W)) {
        (signal ? this->matr[inc][this->__W - 1] : this->matr[this->__H - 1][inc]) = Dpd(value, 0.0);
    }
}

// @brief   sigmoid on matrix
void Matrix::filt_sigmoid() {
    //< check
    assert(this->__W == 1);
    For_(row, 0, this->__H) {
        this->matr[row][0].first = sigmoid(this->matr[row][0].first);
    }
}

// @brief   Relu on matrix
void Matrix::filt_Relu() {
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            this->matr[row][col].first = Relu(this->matr[row][col].first);
        }
    }
}

// @brief   read file for first value, just for debug
void Matrix::read_file(std::ifstream& inFile) {
    //< make sure file open successfully
    assert(inFile.good());
    // read H and W
    inFile >> this->__H >> this->__W;
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            inFile >> matr[row][col].first;
#ifdef DEBUG
            matr[row][col].first /= 128;
#endif // DEBUG

        }
    }
}

// @brief   save file
void Matrix::save_file(std::ofstream& outFile) {
    //< check
    assert(outFile.good());
    outFile << this->__H << " " << this->__W << std::endl;
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            outFile << this->matr[row][col].first << " \n"[col == this->__W - 1];
        }
    }
}

// @brief   print all matrix for debug
void Matrix::print() const {
    For_(row, 0, this->__H) {
        For_(col, 0, this->__W) {
            std::printf("%.04lf|%.02lf%c",\
                matr[row][col].first, matr[row][col].second, "\t\n"[col == this->__W - 1]);
        }
    }
}

// @brief   sigmoid function
inline double sigmoid(double val) {
    return 1.0 / (1 + std::exp(-val));
}

// @brief   Relu function
inline double Relu(double val) {
    return std::max(0.0, val);
}
