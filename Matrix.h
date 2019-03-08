#pragma once
#ifndef MATRIX
#define MATRIX
// base head file
#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

typedef std::pair<double, double> Dpd;

/**
 * @name 矩阵的最大宽/高 MAX_WH
 * @name 全连接矩阵的最大宽高 MAX_WH_DNN
 * @{
 */
#define MAX_WH 122
/** @} */

/**
 * @name 宏函数, 简化for循环
 * @{
 */
#define For_(T, B, E) for (int T = B; T < E; T ++)
#define _For(T, B, E) for (int T = B; T >= E; T --)
/** @} */


/**
 * @name all class and struct
 * @{
 */
struct Filters;
class Matrix;
class Maps;
class DNN_Part;
class CNN_Part;
class CNN_Frame;
/** @}*/

/**
 * @name sigmoid function
 * @{
 */
inline double sigmoid(double);
/** @} */

/**
 * @name Relu function
 * @{
 */
inline double Relu(double);
/** @} */

class Matrix {
public:
    Matrix();
    Matrix(int, int);
    Matrix(const Matrix&);
    ~Matrix();

    //< get __H/__W
    int get_H() const;
    int get_W() const;
    //< set __H/__W
    void set_H(int);
    void set_W(int);

    //< get value (fir/sec) at matr[row][col]
    double get_value_fir(int, int) const;
    double get_value_sec(int, int) const;
    
    //< set value (fir/sec) at matr[row][col]
    void set_value_fir(int, int, double);
    void set_value_sec(int, int, double);
    
    //< make the matrix rotate 180°
    void rotate_180();

    //< padding zero
    void padding_0(int);
    //< depadding
    void depadding(int);
    
    //< cross_correlation operation| feature map and filter
    //< -#param: stride(int)
    //< -#param: bias(int)
    //< -#param: mode(bool)
    //< -#param: feat_tar(bool)
    //< -#param: filt_tar(bool)
    //< -#param: recv_tar(bool)
    void cross_correlation(const Matrix&, const Matrix&, int, int, bool, bool, bool, bool);

    //< mean_pooling
    void mean_pooling(const Matrix&, const int size);
    
    //< matrix product
    void mult(const Matrix&, const Matrix&);

    //< append one line filled by element
    //< <em>false</em>   row
    //< <em>true</em>    col
    void append(bool = false, double = 1.0);

    //< go through sigmoid function only when DNN forward
    void filt_sigmoid();

    //< go through Relu function only when CNN forward
    void filt_Relu();

    //< read file for debug
    void read_file(std::ifstream&);

    //< save file
    void save_file(std::ofstream&);

    //< print matrix for debug
    void print() const;
private:
    Dpd** matr;
    int __W;
    int __H;
};

#endif // !MATRIX
