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
const char* cfgfile = "construct.cfg";
const char* paramfile = "param.txt";
const char* resultfile = "res.txt";
const double rate = 0.35;


int main(int argc, char* argv[]) {
    srand(unsigned int(time(NULL)));

    CNN_Frame cnn_frame(trainfile, cfgfile, resultfile);

    cnn_frame.read_cfg();

    cnn_frame.read_input();

    cnn_frame.Forward();

    cnn_frame.Backward();


#ifdef VISUAL_STUDIO
    system("pause");
#endif
    return 0;
}