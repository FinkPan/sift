#include <sift.hpp>


void sift::LoadImage(const string &file_path)
{
    grayimage_ = imread(file_path,0);
    if(grayimage_.empty())
    {
        cout << "无法读取图片: " << file_path << ".\n";
        exit(-1);
    }
}

void sift::AssignOrientations()
{
    cout << "\nAssigning Orientations to keypoints...\n\n";
    vector<double> hist;

    //对于在DOG金字塔中检测出的关键点点，采集其所在高斯金字塔图像3σ邻域窗口内像素的梯度和方向分布特征
    //梯度的模值m(x,y)按：1.5*sigma 的高斯分布加成，即：ORI_SIGMA_TIMES*extrema[i].octave_scale
    //ORI_HIST_BINS=36 //梯度直方图将0~360度的方向范围分为36个柱(bins)，其中每柱10度
    //ORI_SIGMA_TIME = 1.5
    //ORI_WINDOW_RADIUS = 3.0*ORI_SIGMA_TIME
    CalculateOrientationHistogram(grayimage_,
        cvRound(descriptor_.x), cvRound(descriptor_.y), 
        36,            //36 bins
        4, 
        3, 
        hist);                    //返回的是36大小的数组指针
    for(int j = 0; j < 2; j++)
        GaussSmoothOriHist(hist, 36);
    //随着距中心点越远的领域其对直方图的贡献也响应减小.Lowe论文中还提到要使用高斯函数对直方图进行平滑，减少突变的影响。
    //由于SIFT算法只考虑了尺度和旋转不变性，并没有考虑仿射不变性。
    //通过高斯加权，使特征点附近的梯度幅值有较大的权重，这样可以部分弥补因没有仿射不变性而产生的特征点不稳定的问题。

    descriptor_.orientation = DominantDirection(hist, 36);//直方图的峰值代表该特征点处邻域内图像梯度的主方向，也即该特征点的主方向
}

void sift::CalculateOrientationHistogram(const Mat& gauss, int x, int y, int bins, int radius, double sigma, vector<double>& hist)
{	//radius就是3sigma窗口
    hist.assign(bins,0.0);

    double mag, ori;
    double weight;
    int bin;
    const double PI2 = 2.0*CV_PI;
    double econs = -1.0/(2.0*sigma*sigma);
    for(int i = -radius; i <= radius; i++)
    {
        for(int j = -radius; j <= radius; j++)//在取得的区域中统计直方图
        {
            if(CalcGradMagOri(gauss, x+i, y+j, mag, ori))//获得位置和方向
            {
                weight = exp((i*i+j*j)*econs);//计算权重

                //使用Pi-ori将ori转换到[0,2*PI]之间
                bin = cvRound(bins * (CV_PI - ori)/PI2);
                bin = bin < bins ? bin : 0;

                hist[bin] += mag * weight;

                cout <<  "hist[" << bin << "] = " << hist[bin] << endl;


            }
        }
    }
}

//计算模值和方向：计算在gauss图像中坐标为xy处的模值和方向
bool sift::CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori)
{
    //x 为 Mat中的 col; y 为 Mat中的 row
    if(x > 0 && x < gauss.cols-1 && y > 0 && y < gauss.rows -1)
    {
          //*(img.data + img.step[0]*y + img.step[1]*x) = 255;
          //  首地址        行row             列col
          double dx = *(gauss.data + gauss.step[0] * (y+1) + gauss.step[1]*x) -
                      *(gauss.data + gauss.step[0] * (y-1) + gauss.step[1]*x);

          double dy = *(gauss.data + gauss.step[0] * y + gauss.step[1]*(x+1)) -
                      *(gauss.data + gauss.step[0] * y + gauss.step[1]*(x-1));

        mag = sqrt(dx*dx + dy*dy);
        ori = atan2(dx, dy);//atan2返回[-Pi, -Pi]的弧度值
        return true;
    }
    else
        return false;
}

void sift::DescriptorRepresentation()
{
    double ***hist;

    hist = CalculateDescrHist(grayimage_,
        descriptor_.x, descriptor_.y, 1, descriptor_.orientation, 8, 4);

    
    int  width = 4;
    int bins = 8;

    int int_val, i, r, c, o, k = 0;

    for( r = 0; r < width; r++ )
        for( c = 0; c < width; c++ )
            for( o = 0; o < bins; o++ )
            {
                descriptor_.descriptorvector[k++] = hist[r][c][o];//转换为一维向量了
            }

    //feature.descr_length = k;//共有k个特征向量

    NormalizeDescr(descriptor_);//特征矢量形成后，为了去除光照变化的影响，需要对它们进行归一化处理。

    for( i = 0; i < k; i++ )
    {
        if( descriptor_.descriptorvector[i] > 0.2 )//在归一化处理后，对于特征矢量中值大于0.2的要进行截断，即大于0.2的值只取0.2
            descriptor_.descriptorvector[i] = 0.2;
    }
    NormalizeDescr(descriptor_);//然后，再进行一次归一化处理，其目的是提高特征的鉴别性。


    /* convert floating-point descriptor to integer valued descriptor */
    for( i = 0; i < k; i++ )
    {
        int_val = 512 * descriptor_.descriptorvector[i];      //INT_DESCR_FCTR=512.0
        descriptor_.descriptorvector[i] = min( 255, int_val );//不能超过255
    }



    for(int j = 0; j < 4; j++)
    {

        for(int k = 0; k < 4; k++)
        {
            delete[] hist[j][k];
        }
        delete[] hist[j];
    }
    delete[] hist;
}

void sift::NormalizeDescr(Descriptor& feat)//归一化处理
{
    double cur, len_inv, len_sq = 0.0;
    int i, d = 128;

    for( i = 0; i < d; i++ )
    {
        cur = feat.descriptorvector[i];
        len_sq += cur*cur;
    }
    len_inv = 1.0 / sqrt( len_sq );
    for( i = 0; i < d; i++ )
        feat.descriptorvector[i] *= len_inv;
}

/************************************************************************/
/*5.3 对关键点周围图像区域分块，计算块内梯度直方图，生成具有独特性的向量*/
/* 	(1)确定描述子直方图的邻域的半径;									*/
/*	(2)将坐标轴旋转到关键点方向;										*/
/*	(3)将邻域内的采样点分配到对应的子区域内，							*/
/*	   将子区域内的梯度值分配到8个方向上，计算其权值。				    */
/************************************************************************/
double*** sift::CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width)
{
    double ***hist = new double**[width];

    //申请空间并初始化
    for(int i = 0; i < width; i++)
    {
        hist[i] = new double*[width];
        for(int j = 0; j < width; j++)
        {
            hist[i][j] = new double[bins];
        }
    }

    for(int r = 0; r < width; r++ )
        for(int c = 0; c < width; c++ )
            for(int o = 0; o < bins; o++ )
                hist[r][c][o]=0.0;


    double cos_ori = cos(ori);
    double sin_ori = sin(ori);

    double sigma = 0.5 * width;//Lowe建议子区域的像素的梯度大小按子窗口宽度的一半进行高斯加权计算

    double conste = -1.0/(2*sigma*sigma);
    double PI2 = CV_PI * 2;
    double sub_hist_width = 3 * octave_scale;//DESCR_SCALE_ADJUST=3，即3sigma半径

    //实际计算领域半径时，需要采用双线性插值，所需图像窗口边长为3*sigma*(d+1)
    //在考虑到旋转因素(方便下一步将坐标轴旋转到关键点的方向)，半径变为:r*(sqrt(2.0)/2);
    //实际计算所需的图像区域半径为：(3sigma*(d+1)*pow(2,0.5))/2
    int radius = (sub_hist_width*sqrt(2.0)*(width+1))/2.0 + 0.5; //+0.5取四舍五入

    double grad_ori, grad_mag;
    for(int i = -radius; i <= radius; i++)
    {
        for(int j =-radius; j <= radius; j++)
        {
            //将坐标轴旋转为关键点的方向，以确保旋转不变性
            double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
            double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;


            //xbin,ybin为落在4*4窗口中的下标值
            double xbin = rot_x + width/2 - 0.5;
            double ybin = rot_y + width/2 - 0.5;

            //
            if(xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width)
            {
                if(CalcGradMagOri(gauss, x+j, y + i, grad_mag, grad_ori))//计算gauss中位于(x+j, y + i)处的权重
                {
                    grad_ori = (CV_PI-grad_ori) - ori;
                    while(grad_ori < 0.0)
                        grad_ori += PI2;
                    while(grad_ori >= PI2)
                        grad_ori -= PI2;

                    double obin = grad_ori * (bins/PI2);

                    double weight = exp(conste*(rot_x*rot_x + rot_y * rot_y));

                    InterpHistEntry(hist, xbin, ybin, obin, grad_mag*weight, bins, width);//插值计算每个种子点八个方向的梯度
                }
            }
        }
    }
    return hist;
}

/************************************************************************/
/*        		 5.1 插值计算每个种子点八个方向的梯度  				    */
/************************************************************************/
void sift::InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d)
{
	double d_r, d_c, d_o, v_r, v_c, v_o;
	double** row, * h;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor( ybin );
	c0 = cvFloor( xbin );
	o0 = cvFloor( obin );
	d_r = ybin - r0;
	d_c = xbin - c0;
	d_o = obin - o0;

	/*
		做插值：
		xbin,ybin,obin:种子点所在子窗口的位置和方向
		所有种子点都将落在4*4的窗口中
		r0,c0取不大于xbin，ybin的正整数
		r0,c0只能取到0,1,2
		xbin,ybin在(-1, 2)

		r0取不大于xbin的正整数时。
		r0+0 <= xbin <= r0+1
		mag在区间[r0,r1]上做插值

		obin同理
	*/

	for( r = 0; r <= 1; r++ )
	{
		rb = r0 + r;
		if( rb >= 0  &&  rb < d )
		{
			v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
			row = hist[rb];
			for( c = 0; c <= 1; c++ )
			{
				cb = c0 + c;
				if( cb >= 0  &&  cb < d )
				{
					v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
					h = row[cb];
					for( o = 0; o <= 1; o++ )
					{
						ob = ( o0 + o ) % bins;
						v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
						h[ob] += v_o;
					}
				}
			}
		}
	}
}

//高斯平滑，模板为{0.25, 0.5, 0.25}
void sift::GaussSmoothOriHist(vector<double>& hist, int n)
{
    double prev = hist[n-1], temp, h0=hist[0];


    for(int i = 0; i < n; i++)
    {
        temp = hist[i];
        hist[i] = 0.25 * prev + 0.5 * hist[i] + 
            0.25 * (i+1 >= n ? h0 : hist[i+1]);
        prev = temp;
    }
}

/************************************************************************/
/*        		 		4.3 直方图的极值查找	    				    */
/************************************************************************/
double sift::DominantDirection(vector<double>& hist, int n)
{
    double maxd = hist[0];
    for(int i = 1; i < n; i++)
    {
        if(hist[i] > maxd)
            maxd = hist[i];
    }
    return maxd;
}

void sift::write_features(string &file)
{
    ofstream dout(file);
    dout << "the demain is:" << 128 <<endl<<endl;

    dout<<"(x,y):"<<"("<<descriptor_.x << ", " << descriptor_.y  << ")" << endl;//dout<<"关键点坐标(x,y):"<<"("features[i].dx<<", "<<features[i].dy<<")"<<endl;
    dout<<"scale:"<<descriptor_.scale  << endl << "orientation:" << descriptor_.orientation << endl;//dout<<"尺度(scale):"<<features[i].scale<<endl<<"关键点的梯度方向(orientation):"<<features[i].ori<<endl;
    dout<<"vector is:"<<endl;//dout<<"16个种子点的8个方向向量的信息共128个信息:"<<endl;
    for(int j = 0; j < 128; j++)
    {
        if(j % 20 == 0)
            dout<<endl;
        dout << descriptor_.descriptorvector[j]<<" "; 
    }
    dout<<endl<<endl<<endl;
    dout.close();
}