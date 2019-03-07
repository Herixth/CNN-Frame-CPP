/*****************************************************************************
*  Convolutional Neural Networks Frame(CNN-Frame)                            *
*  Copyright (C) 2019 Xiang.He  herixth@outlook.com.                         *
*                                                                            *
*  This file is part of CNN-Frame.                                           *
*                                                                            *
*  This program is free software; you can redistribute it and/or modify      *
*  it under the terms of the GNU General Public License version 3 as         *
*  published by the Free Software Foundation.                                *
*                                                                            *
*  You should have received a copy of the GNU General Public License         *
*  along with CNN-Frame. If not, see <http://www.gnu.org/licenses/>.         *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*  @file     run_main.cpp                                                    *
*  @brief    main function entry file                                        *
*  Details.                                                                  *
*                                                                            *
*  @author   Xiang.He                                                        *
*  @email    herixth@outlook.com                                             *
*  @version  1.0.0.1(°æ±¾ºÅ)                                                 *
*  @date     2019/03/01                                                      *
*  @license  GNU General Public License (GPL)                                *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         : Description                                              *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/03/01 | 1.0.0.1   | Xiang.He       | Create file                     *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#include <cstdlib>
#include <string>

#include "CNN.h"

using namespace std;

const char* trainfile = "mnist_train.txt";
const char* testfile = "mnist_test.txt";
const char* paramfile = "param.txt";
const char* resultfile = "res.txt";
const double rate = 0.35;

#ifdef DEBUG
inline void save_param(DNN_Part& DNN_) {
    ofstream outFile(paramfile);
    
    DNN_.save_param(outFile);

    outFile.close();
}

inline void read_param(DNN_Part& DNN_) {
    ifstream inFile(paramfile);

    DNN_.read_param(inFile);

    inFile.close();
}
#endif // DEBUG



int main(int argc, char* argv[]) {
    srand(unsigned int(time(NULL)));
    ifstream inFile(trainfile);
    int digit = 0;
    
    DNN_Part DNN_test;

    DNN_test.add_layer(784);
    DNN_test.add_layer(400);
    DNN_test.add_layer(10);
    
    Maps fet;

    read_param(DNN_test);


    For_(iter, 0, 60000) {
        inFile >> digit;


        fet.inputFile(inFile);

        DNN_test.set_tar(digit);
        DNN_test.set_input(fet);

        //< train
        DNN_test.Forward_DNN();
        DNN_test.Backward_DNN(rate);
        

        cout << "order: " << iter << endl;
        int tar = DNN_test.get_tar();
        cout << "ans: " << tar <<  "     tar: " << digit;
        if (tar == digit)
            cout << "    correct";
        cout << endl << endl;
        
        if (iter % 200 == 0)
            save_param(DNN_test);
    }

    inFile.close();
    inFile.open(testfile);
    int cnt = 0;
    For_(iter, 0, 10000) {
        inFile >> digit;
        fet.inputFile(inFile);

        DNN_test.set_tar(digit);
        DNN_test.set_input(fet);

        //< train
        DNN_test.Forward_DNN();

        cout << "test: " << iter << endl;
        int tar = DNN_test.get_tar();
        cout << "ans: " << tar <<  "     tar: " << digit;
        if (tar == digit)
            cnt++;
        cout << endl << endl;
    }

    ofstream res(resultfile, ios::app);
    res << "rate: " << 1.0 * cnt / 10000 << endl;
    inFile.close();
    res.close();
#ifdef VISUAL_STUDIO
    system("pause");
#endif
    return 0;
}