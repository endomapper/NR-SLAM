/*
 * This file is part of NR-SLAM
 *
 * Copyright (C) 2022-2023 Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * NR-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "color_factory.h"

using namespace std;

ColorFactory::ColorFactory() {
    unique_colors_.resize(100);
    unique_colors_[0] = cv::Scalar(255,  255,  255);	//White                    #FFFFFF
    unique_colors_[1] = cv::Scalar(255,    0,    0);	//Red                      #FF0000
    unique_colors_[2] = cv::Scalar(  0,  255,    0);	//Green                    #00FF00
    unique_colors_[3] = cv::Scalar(  0,    0,  255);	//Blue                     #0000FF
    unique_colors_[4] = cv::Scalar(255,    0,  255);	//Magenta                  #FF00FF
    unique_colors_[5] = cv::Scalar(  0,  255,  255);	//Cyan                     #00FFFF
    unique_colors_[6] = cv::Scalar(255,  255,    0);	//Yellow                   #FFFF00
    unique_colors_[7] = cv::Scalar(  0,    0,    0);	//Black                    #000000
    unique_colors_[8] = cv::Scalar(112,  219,  147);	//Aquamarine               #70DB93
    unique_colors_[9] = cv::Scalar( 92,   51,   23);	//Baker's Chocolate        #5C3317
    unique_colors_[10] = cv::Scalar(159,   95,  159);	//Blue Violet              #9F5F9F
    unique_colors_[11] = cv::Scalar(181,  166,   66);	//Brass                    #B5A642
    unique_colors_[12] = cv::Scalar(217,  217,   25);	//Bright Gold              #D9D919
    unique_colors_[13] = cv::Scalar(166,   42,   42);	//Brown                    #A62A2A
    unique_colors_[14] = cv::Scalar(140,  120,   83);	//Bronze                   #8C7853
    unique_colors_[15] = cv::Scalar(166,  125,   61);	//Bronze II                #A67D3D
    unique_colors_[16] = cv::Scalar( 95,  159,  159);	//Cadet Blue               #5F9F9F
    unique_colors_[17] = cv::Scalar(217,  135,   25);	//Cool Copper              #D98719
    unique_colors_[18] = cv::Scalar(184,  115,   51);	//Copper                   #B87333
    unique_colors_[19] = cv::Scalar(255,  127,    0);	//Coral                    #FF7F00
    unique_colors_[20] = cv::Scalar( 66,   66,  111);	//Corn Flower Blue         #42426F
    unique_colors_[21] = cv::Scalar( 92,   64,   51);	//Dark Brown               #5C4033
    unique_colors_[22] = cv::Scalar( 47,   79,   47);	//Dark Green               #2F4F2F
    unique_colors_[23] = cv::Scalar( 74,  118,  110);	//Dark Green Copper        #4A766E
    unique_colors_[24] = cv::Scalar( 79,   79,   47);	//Dark Olive Green         #4F4F2F
    unique_colors_[25] = cv::Scalar(153,   50,  205);	//Dark Orchid              #9932CD
    unique_colors_[26] = cv::Scalar(135,   31,  120);	//Dark Purple              #871F78
    unique_colors_[27] = cv::Scalar(107,   35,  142);	//Dark Slate Blue          #6B238E
    unique_colors_[28] = cv::Scalar( 47,   79,   79);	//Dark Slate Grey          #2F4F4F
    unique_colors_[29] = cv::Scalar(151,  105,   79);	//Dark Tan                 #97694F
    unique_colors_[30] = cv::Scalar(112,  147,  219);	//Dark Turquoise           #7093DB
    unique_colors_[31] = cv::Scalar(133,   94,   66);	//Dark Wood                #855E42
    unique_colors_[32] = cv::Scalar( 84,   84,   84);	//Dim Grey                 #545454
    unique_colors_[33] = cv::Scalar(133,   99,   99);	//Dusty Rose               #856363
    unique_colors_[34] = cv::Scalar(209,  146,  117);	//Feldspar                 #D19275
    unique_colors_[35] = cv::Scalar(142,   35,   35);	//Firebrick                #8E2323
    unique_colors_[36] = cv::Scalar(245,  204,  176);	//Flesh                    #F5CCB0
    unique_colors_[37] = cv::Scalar( 35,  142,   35);	//Forest Green             #238E23
    unique_colors_[38] = cv::Scalar(205,  127,   50);	//Gold                     #CD7F32
    unique_colors_[39] = cv::Scalar(219,  219,  112);	//Goldenrod                #DBDB70
    unique_colors_[40] = cv::Scalar(192,  192,  192);	//Grey                     #C0C0C0
    unique_colors_[41] = cv::Scalar( 82,  127,  118);	//Green Copper             #527F76
    unique_colors_[42] = cv::Scalar(147,  219,  112);	//Green Yellow             #93DB70
    unique_colors_[43] = cv::Scalar( 33,   94,   33);	//Hunter Green             #215E21
    unique_colors_[44] = cv::Scalar( 78,   47,   47);	//Indian Red               #4E2F2F
    unique_colors_[45] = cv::Scalar(159,  159,   95);	//Khaki                    #9F9F5F
    unique_colors_[46] = cv::Scalar(192,  217,  217);	//Light Blue               #C0D9D9
    unique_colors_[47] = cv::Scalar(168,  168,  168);	//Light Grey               #A8A8A8
    unique_colors_[48] = cv::Scalar(143,  143,  189);	//Light Steel Blue         #8F8FBD
    unique_colors_[49] = cv::Scalar(233,  194,  166);	//Light Wood               #E9C2A6
    unique_colors_[50] = cv::Scalar( 50,  205,   50);	//Lime Green               #32CD32
    unique_colors_[51] = cv::Scalar(228,  120,   51);	//Mandarian Orange         #E47833
    unique_colors_[52] = cv::Scalar(142,   35,  107);	//Maroon                   #8E236B
    unique_colors_[53] = cv::Scalar( 50,  205,  153);	//Medium Aquamarine        #32CD99
    unique_colors_[54] = cv::Scalar( 50,   50,  205);	//Medium Blue              #3232CD
    unique_colors_[55] = cv::Scalar(107,  142,   35);	//Medium Forest Green      #6B8E23
    unique_colors_[56] = cv::Scalar(234,  234,  174);	//Medium Goldenrod         #EAEAAE
    unique_colors_[57] = cv::Scalar(147,  112,  219);	//Medium Orchid            #9370DB
    unique_colors_[58] = cv::Scalar( 66,  111,   66);	//Medium Sea Green         #426F42
    unique_colors_[59] = cv::Scalar(127,    0,  255);	//Medium Slate Blue        #7F00FF
    unique_colors_[60] = cv::Scalar(127,  255,    0);	//Medium Spring Green      #7FFF00
    unique_colors_[61] = cv::Scalar(112,  219,  219);	//Medium Turquoise         #70DBDB
    unique_colors_[62] = cv::Scalar(219,  112,  147);	//Medium Violet Red        #DB7093
    unique_colors_[63] = cv::Scalar(166,  128,  100);	//Medium Wood              #A68064
    unique_colors_[64] = cv::Scalar( 47,   47,   79);	//Midnight Blue            #2F2F4F
    unique_colors_[65] = cv::Scalar( 35,   35,  142);	//Navy Blue                #23238E
    unique_colors_[66] = cv::Scalar( 77,   77,  255);	//Neon Blue                #4D4DFF
    unique_colors_[67] = cv::Scalar(255,  110,  199);	//Neon Pink                #FF6EC7
    unique_colors_[68] = cv::Scalar(  0,    0,  156);	//New Midnight Blue        #00009C
    unique_colors_[69] = cv::Scalar(235,  199,  158);	//New Tan                  #EBC79E
    unique_colors_[70] = cv::Scalar(207,  181,   59);	//Old Gold                 #CFB53B
    unique_colors_[71] = cv::Scalar(255,  127,    0);	//Orange                   #FF7F00
    unique_colors_[72] = cv::Scalar(255,   36,    0);	//Orange Red               #FF2400
    unique_colors_[73] = cv::Scalar(219,  112,  219);	//Orchid                   #DB70DB
    unique_colors_[74] = cv::Scalar(143,  188,  143);	//Pale Green               #8FBC8F
    unique_colors_[75] = cv::Scalar(188,  143,  143);	//Pink                     #BC8F8F
    unique_colors_[76] = cv::Scalar(234,  173,  234);	//Plum                     #EAADEA
    unique_colors_[77] = cv::Scalar(217,  217,  243);	//Quartz                   #D9D9F3
    unique_colors_[78] = cv::Scalar( 89,   89,  171);	//Rich Blue                #5959AB
    unique_colors_[79] = cv::Scalar(111,   66,   66);	//Salmon                   #6F4242
    unique_colors_[80] = cv::Scalar(140,   23,   23);	//Scarlet                  #8C1717
    unique_colors_[81] = cv::Scalar( 35,  142,  104);	//Sea Green                #238E68
    unique_colors_[82] = cv::Scalar(107,   66,   38);	//Semi-Sweet Chocolate     #6B4226
    unique_colors_[83] = cv::Scalar(142,  107,   35);	//Sienna                   #8E6B23
    unique_colors_[84] = cv::Scalar(230,  232,  250);	//Silver                   #E6E8FA
    unique_colors_[85] = cv::Scalar( 50,  153,  204);	//Sky Blue                 #3299CC
    unique_colors_[86] = cv::Scalar(  0,  127,  255);	//Slate Blue               #007FFF
    unique_colors_[87] = cv::Scalar(255,   28,  174);	//Spicy Pink               #FF1CAE
    unique_colors_[88] = cv::Scalar(  0,  255,  127);	//Spring Green             #00FF7F
    unique_colors_[89] = cv::Scalar( 35,  107,  142);	//Steel Blue               #236B8E
    unique_colors_[90] = cv::Scalar( 56,  176,  222);	//Summer Sky               #38B0DE
    unique_colors_[91] = cv::Scalar(219,  147,  112);	//Tan                      #DB9370
    unique_colors_[92] = cv::Scalar(216,  191,  216);	//Thistle                  #D8BFD8
    unique_colors_[93] = cv::Scalar(173,  234,  234);	//Turquoise                #ADEAEA
    unique_colors_[94] = cv::Scalar( 92,   64,   51);	//Very Dark Brown          #5C4033
    unique_colors_[95] = cv::Scalar(205,  205,  205);	//Very Light Grey          #CDCDCD
    unique_colors_[96] = cv::Scalar( 79,   47,   79);	//Violet                   #4F2F4F
    unique_colors_[97] = cv::Scalar(204,   50,  153);	//Violet Red               #CC3299
    unique_colors_[98] = cv::Scalar(216,  216,  191);	//Wheat                    #D8D8BF
    unique_colors_[99] = cv::Scalar(153,  204,   50);	//Yellow Green             #99CC32
}

std::vector<cv::Scalar> ColorFactory::GetUniqueColors(const int n) {
    return vector(unique_colors_.begin(), unique_colors_.begin() + n);
}
