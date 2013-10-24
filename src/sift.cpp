#include <sift.hpp>


void sift::LoadImage(const string &file_path)
{
    grayimage_ = imread(file_path,0);
    if(grayimage_.empty())
    {
        cout << "�޷���ȡͼƬ: " << file_path << ".\n";
        exit(-1);
    }
}

void sift::AssignOrientations()
{
    cout << "\nAssigning Orientations to keypoints...\n\n";
    vector<double> hist;

    //������DOG�������м����Ĺؼ���㣬�ɼ������ڸ�˹������ͼ��3�����򴰿������ص��ݶȺͷ���ֲ�����
    //�ݶȵ�ģֵm(x,y)����1.5*sigma �ĸ�˹�ֲ��ӳɣ�����ORI_SIGMA_TIMES*extrema[i].octave_scale
    //ORI_HIST_BINS=36 //�ݶ�ֱ��ͼ��0~360�ȵķ���Χ��Ϊ36����(bins)������ÿ��10��
    //ORI_SIGMA_TIME = 1.5
    //ORI_WINDOW_RADIUS = 3.0*ORI_SIGMA_TIME
    CalculateOrientationHistogram(grayimage_,
        cvRound(descriptor_.x), cvRound(descriptor_.y), 
        36,            //36 bins
        4, 
        3, 
        hist);                    //���ص���36��С������ָ��
    for(int j = 0; j < 2; j++)
        GaussSmoothOriHist(hist, 36);
    //���ž����ĵ�ԽԶ���������ֱ��ͼ�Ĺ���Ҳ��Ӧ��С.Lowe�����л��ᵽҪʹ�ø�˹������ֱ��ͼ����ƽ��������ͻ���Ӱ�졣
    //����SIFT�㷨ֻ�����˳߶Ⱥ���ת�����ԣ���û�п��Ƿ��䲻���ԡ�
    //ͨ����˹��Ȩ��ʹ�����㸽�����ݶȷ�ֵ�нϴ��Ȩ�أ��������Բ����ֲ���û�з��䲻���Զ������������㲻�ȶ������⡣

    descriptor_.orientation = DominantDirection(hist, 36);//ֱ��ͼ�ķ�ֵ����������㴦������ͼ���ݶȵ�������Ҳ�����������������
}

void sift::CalculateOrientationHistogram(const Mat& gauss, int x, int y, int bins, int radius, double sigma, vector<double>& hist)
{	//radius����3sigma����
    hist.assign(bins,0.0);

    double mag, ori;
    double weight;
    int bin;
    const double PI2 = 2.0*CV_PI;
    double econs = -1.0/(2.0*sigma*sigma);
    for(int i = -radius; i <= radius; i++)
    {
        for(int j = -radius; j <= radius; j++)//��ȡ�õ�������ͳ��ֱ��ͼ
        {
            if(CalcGradMagOri(gauss, x+i, y+j, mag, ori))//���λ�úͷ���
            {
                weight = exp((i*i+j*j)*econs);//����Ȩ��

                //ʹ��Pi-ori��oriת����[0,2*PI]֮��
                bin = cvRound(bins * (CV_PI - ori)/PI2);
                bin = bin < bins ? bin : 0;

                hist[bin] += mag * weight;

                cout <<  "hist[" << bin << "] = " << hist[bin] << endl;


            }
        }
    }
}

//����ģֵ�ͷ��򣺼�����gaussͼ��������Ϊxy����ģֵ�ͷ���
bool sift::CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori)
{
    //x Ϊ Mat�е� col; y Ϊ Mat�е� row
    if(x > 0 && x < gauss.cols-1 && y > 0 && y < gauss.rows -1)
    {
          //*(img.data + img.step[0]*y + img.step[1]*x) = 255;
          //  �׵�ַ        ��row             ��col
          double dx = *(gauss.data + gauss.step[0] * (y+1) + gauss.step[1]*x) -
                      *(gauss.data + gauss.step[0] * (y-1) + gauss.step[1]*x);

          double dy = *(gauss.data + gauss.step[0] * y + gauss.step[1]*(x+1)) -
                      *(gauss.data + gauss.step[0] * y + gauss.step[1]*(x-1));

        mag = sqrt(dx*dx + dy*dy);
        ori = atan2(dx, dy);//atan2����[-Pi, -Pi]�Ļ���ֵ
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
                descriptor_.descriptorvector[k++] = hist[r][c][o];//ת��Ϊһά������
            }

    //feature.descr_length = k;//����k����������

    NormalizeDescr(descriptor_);//����ʸ���γɺ�Ϊ��ȥ�����ձ仯��Ӱ�죬��Ҫ�����ǽ��й�һ������

    for( i = 0; i < k; i++ )
    {
        if( descriptor_.descriptorvector[i] > 0.2 )//�ڹ�һ������󣬶�������ʸ����ֵ����0.2��Ҫ���нضϣ�������0.2��ֵֻȡ0.2
            descriptor_.descriptorvector[i] = 0.2;
    }
    NormalizeDescr(descriptor_);//Ȼ���ٽ���һ�ι�һ��������Ŀ������������ļ����ԡ�


    /* convert floating-point descriptor to integer valued descriptor */
    for( i = 0; i < k; i++ )
    {
        int_val = 512 * descriptor_.descriptorvector[i];      //INT_DESCR_FCTR=512.0
        descriptor_.descriptorvector[i] = min( 255, int_val );//���ܳ���255
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

void sift::NormalizeDescr(Descriptor& feat)//��һ������
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
/*5.3 �Թؼ�����Χͼ������ֿ飬��������ݶ�ֱ��ͼ�����ɾ��ж����Ե�����*/
/* 	(1)ȷ��������ֱ��ͼ������İ뾶;									*/
/*	(2)����������ת���ؼ��㷽��;										*/
/*	(3)�������ڵĲ�������䵽��Ӧ���������ڣ�							*/
/*	   ���������ڵ��ݶ�ֵ���䵽8�������ϣ�������Ȩֵ��				    */
/************************************************************************/
double*** sift::CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width)
{
    double ***hist = new double**[width];

    //����ռ䲢��ʼ��
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

    double sigma = 0.5 * width;//Lowe��������������ص��ݶȴ�С���Ӵ��ڿ�ȵ�һ����и�˹��Ȩ����

    double conste = -1.0/(2*sigma*sigma);
    double PI2 = CV_PI * 2;
    double sub_hist_width = 3 * octave_scale;//DESCR_SCALE_ADJUST=3����3sigma�뾶

    //ʵ�ʼ�������뾶ʱ����Ҫ����˫���Բ�ֵ������ͼ�񴰿ڱ߳�Ϊ3*sigma*(d+1)
    //�ڿ��ǵ���ת����(������һ������������ת���ؼ���ķ���)���뾶��Ϊ:r*(sqrt(2.0)/2);
    //ʵ�ʼ��������ͼ������뾶Ϊ��(3sigma*(d+1)*pow(2,0.5))/2
    int radius = (sub_hist_width*sqrt(2.0)*(width+1))/2.0 + 0.5; //+0.5ȡ��������

    double grad_ori, grad_mag;
    for(int i = -radius; i <= radius; i++)
    {
        for(int j =-radius; j <= radius; j++)
        {
            //����������תΪ�ؼ���ķ�����ȷ����ת������
            double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
            double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;


            //xbin,ybinΪ����4*4�����е��±�ֵ
            double xbin = rot_x + width/2 - 0.5;
            double ybin = rot_y + width/2 - 0.5;

            //
            if(xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width)
            {
                if(CalcGradMagOri(gauss, x+j, y + i, grad_mag, grad_ori))//����gauss��λ��(x+j, y + i)����Ȩ��
                {
                    grad_ori = (CV_PI-grad_ori) - ori;
                    while(grad_ori < 0.0)
                        grad_ori += PI2;
                    while(grad_ori >= PI2)
                        grad_ori -= PI2;

                    double obin = grad_ori * (bins/PI2);

                    double weight = exp(conste*(rot_x*rot_x + rot_y * rot_y));

                    InterpHistEntry(hist, xbin, ybin, obin, grad_mag*weight, bins, width);//��ֵ����ÿ�����ӵ�˸�������ݶ�
                }
            }
        }
    }
    return hist;
}

/************************************************************************/
/*        		 5.1 ��ֵ����ÿ�����ӵ�˸�������ݶ�  				    */
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
		����ֵ��
		xbin,ybin,obin:���ӵ������Ӵ��ڵ�λ�úͷ���
		�������ӵ㶼������4*4�Ĵ�����
		r0,c0ȡ������xbin��ybin��������
		r0,c0ֻ��ȡ��0,1,2
		xbin,ybin��(-1, 2)

		r0ȡ������xbin��������ʱ��
		r0+0 <= xbin <= r0+1
		mag������[r0,r1]������ֵ

		obinͬ��
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

//��˹ƽ����ģ��Ϊ{0.25, 0.5, 0.25}
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
/*        		 		4.3 ֱ��ͼ�ļ�ֵ����	    				    */
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

    dout<<"(x,y):"<<"("<<descriptor_.x << ", " << descriptor_.y  << ")" << endl;//dout<<"�ؼ�������(x,y):"<<"("features[i].dx<<", "<<features[i].dy<<")"<<endl;
    dout<<"scale:"<<descriptor_.scale  << endl << "orientation:" << descriptor_.orientation << endl;//dout<<"�߶�(scale):"<<features[i].scale<<endl<<"�ؼ�����ݶȷ���(orientation):"<<features[i].ori<<endl;
    dout<<"vector is:"<<endl;//dout<<"16�����ӵ��8��������������Ϣ��128����Ϣ:"<<endl;
    for(int j = 0; j < 128; j++)
    {
        if(j % 20 == 0)
            dout<<endl;
        dout << descriptor_.descriptorvector[j]<<" "; 
    }
    dout<<endl<<endl<<endl;
    dout.close();
}