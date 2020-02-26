
//	Copyright © 2020 Yashkir Consulting
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <time.h>
#include "f.h"
#include <armadillo>
using namespace std;
//
arma::Row <double> get_col(arma::Mat <double> W, int i, arma::Row <double> v)
{
	int j;
	int n = (int)v.n_elem;
	for (j = 0; j < n; j++)
		v(j) = W(j, i);
	return v;
}
arma::Mat<double> generator(arma::Mat<double> matrix, int rank, int expansion_number)
{/*
	matrix Q (generator) is calculated as log of the credit rating transition matrix,
	by Taylor expansion series
*/
	int  j, k, s;
	arma::Mat<double> Q(rank, rank);
	arma::Mat<double> R(rank, rank);
	arma::Mat <double> D(rank, rank); 
	D = matrix;
	for (j = 0; j < rank; j++) D(j, j) = D(j,j) -  1.;
	Q = D;

	for (k = 2;k <= expansion_number;k++)
	{
		s = int(pow(-1, k + 1));
		R = powm(D, rank, k);
		Q = Q + R * s / k;
	}
	return Q;
}
//
arma::Mat<double> mpower(arma::Mat<double> Q, int rank, double q, int expansion_number)
{
//	Returns the matrix B calculated as exp(q * Q) which is matrix^q (by Taylor expansion)
	int  j, k;
	double kfact, v, qk;
	arma::Mat<double> B(rank, rank);
	arma::Mat<double> R (rank, rank);

	B.fill(0);
	for (j = 0; j < rank; j++) B(j, j) = 1.;

	kfact = 1.;
	qk = 1;
	for (k = 1;k <= expansion_number;k++)
	{
		kfact = kfact * k;
		qk = qk * q;
		v = qk / kfact;
		R = powm(Q, rank, k);
		B = B + v * R;
	}
	return B;
}
//
arma::Mat<double> powm(arma::Mat<double> D, int rank, int k)
{
//	returns power of a matrix D as T = D^k (k is integer > 0)
	int p;
	arma::Mat<double> T(rank, rank);
	T = D;
	for (p = 1; p < k; p++)
	{
		T = T * D;
	}
	return T;
}
//
double interpolation(arma::Mat<double> y0, int n, int rank, double t)
{
	// interpolation for value 'y' for tenor t using data y0
	int p;
	double y;
	for (p = 1; p < n; p++)
	{
		if (t >= y0(p-1,0) && t <= y0(p,0))	break;
	}
	if (t <= y0(0,0) ) p = 0;
	if (t >= y0(n-1,0))  p = n - 2;
	y = y0(p, rank) + (y0(p + 1,rank) - y0(p,rank)) * (t - y0(p,0)) / (y0(p + 1,0) - y0(p,0));
	return y;
}
//
double inter(arma::Mat<double> M, int n_rows, int n_col, double t)
{
	int k;
	if (t > M(n_rows - 1, 0))
		return M(n_rows - 1, n_col); // if t > the last value in the 1st column then return this last value
	for (k = 0; k < n_rows - 1; k++)
	{
		if (M(k, 0) <= t && t <= M(k + 1, 0))
			break;		// found row number "k" where "t" lies between row "k" and row "k+1" of the 1st column values 
	}
	double b = M(k, n_col) + (M(k + 1, n_col) - M(k, n_col))*(t - M(k, 0)) / (M(k + 1, 0) - M(k, 0));
	return b;
}
//
arma::Mat<double> default_calc_v0(arma::Mat<double> defaults, arma::Mat<double> spreads, int numb_input_spreads, arma::Row<double> pay_times, int number_periods, int rank, arma::Mat<double> recovery_rates)
{/*
	default probabilities for periods from t=0 to payment date k
	calculated using credit spreads and recovery rates for all ratings
*/
	int k, j;
	double tenor, spread, defaut_probability;
	for (k = 0; k < number_periods + 1; k++)
	{
		tenor = pay_times(k);
		for (j = 0; j < rank - 1; j++)
		{
			spread = interpolation(spreads, numb_input_spreads, j + 1, tenor);
			defaut_probability = (1. - exp(-spread * tenor)) / (1. - recovery_rates(j, 1));
			defaults(k, j) = defaut_probability;
		}
	}
	return defaults;
}
//
arma::Mat<double> calc_marginal_matrix(arma::Cube<double> matrix_set, int payoff_numb, int rank)
{
	// marginal credit rating transition matrix  for given swap time period
	int j, i;
	// M[k-1][][] x Y[][] = M[k][][]
	// marginal matrix Y for period k-1 to k is: Y[][] = inverse(M[k-1][][]) x M[k][][]
	arma::Mat<double> M_k_1(rank, rank);
	arma::Mat<double> M_k(rank, rank);
	arma::Mat<double> inverse_matrix(rank, rank);
	arma::Mat<double> marginal_matrix(rank, rank);

	M_k_1 = matrix_set.slice(payoff_numb - 1);
	M_k   = matrix_set.slice(payoff_numb);
	inverse_matrix = inv(M_k_1);
	marginal_matrix = inverse_matrix * M_k;
	for (j = 0; j < rank; j++)for (i = 0; i < rank; i++)marginal_matrix(j, i) = max(0, marginal_matrix(j, i));

	return marginal_matrix;
}
//
arma::Mat<double> matrix_power_q(arma::Mat<double> matrix, int rank, double q, int expansion_number)
{
//	raising matrix to arbitrary power
	double dm;
	arma::Mat<double> B;
	arma::Mat<double> Q;
	arma::Mat<double> M;
	int m;
	// splitting q into an integer "m" and a fraction (dm < 1)
	m = int(q);
	dm = q - m;
	if (q >= 1 && dm == 0)		//when the power is an integer:  1,2,3,4,...
	{
		B = powm(matrix, rank, m);
		B = normalize_matrix(B, rank);
		return B;
	}
	else if (q < 1)				// power is less than 1  (0.5  or 0.25 or anything < 1 , but > 0)
	{
		Q = generator(matrix, rank, expansion_number); // Q= log matrix  using "Tailor series"
		B = mpower(Q, rank, q, expansion_number);	   // B=matrix^q as B=exp(q Q) using "Tailor series"	
		B = normalize_matrix(B, rank);
		return B;
	}
	else if (q > 1)				// power value = integer m  + fraction dm
	{
		M = powm(matrix, rank, m);						// matrix^m by multiplication
		M = normalize_matrix(M, rank);
		Q = generator(matrix, rank, expansion_number);	// Q= log (matrix)  using "Tailor series"
		B = mpower(Q, rank, dm, expansion_number);		// matrix^dm = exp(Q) using "Tailor series"
		B = normalize_matrix(B, rank);
		B =  M % B;						// matrix^q = (matrix^m) x (matrix^dm)
		B = normalize_matrix(B, rank);
		return(B);
	}
	else {
		cout<<"Error: matrix power < 0";
		return matrix;
	}
}
//
arma::Mat<double> normalize_matrix(arma::Mat<double> matr, int rank)
{
//	rescaling each row to have sum of this row equal to 1
	int j, i;
	double row_sum;
	for (j = 0; j < rank; j++)
	{
		row_sum = 0;
		for (i = 0; i < rank; i++) row_sum = row_sum + matr(j,i);
		for (i = 0; i < rank; i++)
		{
			matr(j,i) = matr(j,i) / row_sum;
		}
	}
	return matr;
}
//
string date_now()
{
	time_t _tm = time(NULL);
	struct tm *curtime = localtime(&_tm);
	string s = "";
	s.append(asctime(curtime));
	return(s);
}
//

//
arma::Row<double> pfe(arma::Row<double> U, double confidence, arma::Row<double> pfe)
{
	// 3rd argument pfe is the two-element vector for percentiles: confidence and (1-confidence)
	double w, u, dcp;
	int jmax = (int)U.n_elem - 1;
	if (jmax == 0)
	{
		pfe(0) = pfe(1) = U(0);
		return pfe;
	}
	dcp = 1.0 / jmax;
	U = sort(U);
	int j1 = (int)(confidence * jmax);
	int j2 = (int)((1.-confidence) * jmax);
	if (j1 >= jmax) j1 = jmax - 1;
	if (j2 <= 0)    j2 = 0;
	w = U(j1) + (U(j1 + 1) - U(j1)) * (confidence - (1.0 * j1) / jmax) / dcp;
	u = U(j2) + (U(j2 + 1) - U(j2)) * ((1.-confidence) - (1.0 * j2) / jmax) / dcp;
	pfe(0) = w;
	pfe(1) = u;
	return pfe;
}
//
void save_text(string txt, string filename)
{
	ofstream out;
	out.open(filename);
	out << txt ;
	out.close();
}
double max(double a, double b)
{
	if (a >= b)return a;
	else return b;
}
//
int sign(double x)
{
	if (x > 0) return 1;
	if (x < 0) return -1;
	else return 0;
}
//
double cdf(double x)
{
	// cumulative distribution function
	double	A1 = 0.31938153,
		A2 = -0.356563782,
		A3 = 1.781477937,
		A4 = -1.821255978,
		A5 = 1.330274429,
		RSQRT2PI = 0.39894228040143267793994605993438;
	double  K = 1.0 / (1.0 + 0.2316419 * fabs(x));
	double  cnd = RSQRT2PI * exp(-0.5 * x * x) *
		(K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
	if (x > 0)
		cnd = 1.0 - cnd;
	return cnd;
}
//
double d1fwd(double F, double K, double T, double sigma)
{
	return  (log(F / K) + 0.5*sigma*sigma*T) / (sigma * sqrt(T));
}
//
double d2fwd(double d1, double T, double sigma)
{
	return  d1 - sigma * sqrt(T);
}
//
double fwdBS(int cp, double F, double K, double T, double sigma)
{	// cp=1 for call, cp=-1 for put
	//F is forward price, K is strike, T is maturity, sigma is volatility
	if (F <= 0 && cp > 0) return 0;
	if (F <= 0 && cp < 0) return K;
	double d1 = d1fwd(F, K, T, sigma);
	double d2 = d2fwd(d1, T, sigma);
	return cp * (F * cdf(cp*d1) - K * cdf(cp*d2) );
}
//
double BS(int call_put, double spot_price, double strike_price, double interest_rate, double dividend_rate, double time_to_maturity, double volatility)
{
	int eta=call_put;
	double 	S = spot_price,
			K = strike_price,
			r = interest_rate,
			q = dividend_rate,
			tau = time_to_maturity,
			sigma = volatility;
	
	double d1 = (log(S/K) + (r - q + sigma*sigma/2.) * tau) / (sigma*sqrt(tau));
	double d2 = d1 - sigma * sqrt(tau);
	double v = eta * S * exp(-q*tau) * cdf(eta*d1)   -       eta * K * exp(-r*tau) * cdf(eta*d2) ;
	return v;
}
//
double volatility(double tenor, double StoK_ratio, arma::Mat<double> vol_surf, int numb_t, int numb_ratios)
{
	// returns volatility for S/K ratio and tenor, using 2D interpolation
	double vol = 0, x, y, x1 = 0, y1, x2 = 0, y2; // tenor y , ratio S/K  x
	double vol11, vol12, vol21, vol22, R1, R2;
	int i1 = 0, i2 = 0, j1, j2;
	x = StoK_ratio;
	y = tenor;
	/*
	x1,y2		x2,y2
	x,y
	x1,y1		x2,y2
	*/
	int j, i;
	for (i = 1; i < numb_ratios; i++)
	{
		if (x >= vol_surf(0,i) && x <= vol_surf(0,i + 1))
		{
			x1 = vol_surf(0, i);
			x2 = vol_surf(0, i+1);
			i1 = i;
			i2 = i + 1;
		}
	}
	if (x < vol_surf(0, 1))
	{
		x1 = vol_surf(0, 1);
		x2 = vol_surf(0, 2);
		i1 = 1;
		i2 = 2;
	}
	if (x > vol_surf(0,numb_ratios))
	{
		x1 = vol_surf(0, numb_ratios-1);
		x2 = vol_surf(0, numb_ratios);
		i1 = numb_ratios - 1;
		i2 = numb_ratios;
	}
	for (j = 1; j < numb_t; j++)
	{
		if (y >= vol_surf(j,0) && y <= vol_surf(j+1, 0))
		{
			y1 = vol_surf(j, 0);
			y2 = vol_surf(j+1, 0);
			j1 = j;
			j2 = j + 1;
		}
	}
	if (y < vol_surf(1, 0))
	{
		y1 = vol_surf(1, 0);
		y2 = vol_surf(2, 0);
		j1 = 1;
		j2 = 2;
	}
	if (y > vol_surf(numb_t,0))
	{
		y1 = vol_surf(numb_t-1, 0);
		y2 = vol_surf(numb_t, 0);
		j1 = numb_t - 1;
		j2 = numb_t;
	}
	vol11 = vol_surf(j1,i1);
	vol12 = vol_surf(j1, i2);
	vol21 = vol_surf(j2, i1);
	vol22 = vol_surf(j2, i2);
	R1 = ((x2 - x) / (x2 - x1)) * vol11 + ((x - x1) / (x2 - x1)) * vol21;
	R2 = ((x2 - x) / (x2 - x1)) * vol12 + ((x - x1) / (x2 - x1)) * vol22;
	vol = ((y2 - y) / (y2 - y1)) * R1 + ((y - y1) / (y2 - y1)) * R2;
	return vol;
}
//
void plot2graphs(string out_file, int col_x, int col_y1, int col_y2, string xlabel, string ylabel, int show_graph)
{
	stringstream plot_string, show_string;
	string plot;
	if (show_graph != 1)
	{
		cout << endl << "Results of simulation recorded in the output file:  " << out_file <<endl;
	}
	else
	{
		plot_string << "set terminal png;"
			<< endl << "set output " << "'out_" << col_x << col_y1 << col_y2 << ".png';"
			<< endl << "set grid;"
			<< endl << "set key autotitle columnhead;"
			<< endl << "set xlabel '" << xlabel << "';"
			<< endl << "set ylabel '" << ylabel << "';"
			<< endl << "plot '" << out_file << "' using " << col_x << ":" << col_y1 << " with points, '"
			<< out_file << "' using " << col_x << ":" << col_y2 << " with points;";
		plot = plot_string.str();
		ofstream plot_command_file;
		plot_command_file.open("plotcommand");
		plot_command_file << plot;
		plot_command_file.close();
		int ret = system("gnuplot plotcommand");
		if (ret == 0)
		{
			show_string << "eog " << "out_" << col_x << col_y1 << col_y2 << ".png";
			string tmp = show_string.str();
			const char *show_graph = tmp.c_str();
			int retp = system(show_graph);
			if (retp !=0)
				cout << endl << "eog.exe failure"<<endl;
		}
		else
		{
			cout<<endl<<"gnuplot failure\n";
		}
	}
}
//
void plot3graphs(string out_file, int col_x, int col_y1, int col_y2, int col_y3, string xlabel, string ylabel, string title, int show_graph)
{
	stringstream plot_string, show_string;
	string plot;
	if (show_graph != 1)
	{
		cout << endl << "Results of simulation recorded in the output file:  " << out_file << endl;
	}
	else
	{
		plot_string << "set terminal png;"
			<< endl << "set output " << "'out_" << col_x << col_y1 << col_y2 << col_y3 << ".png';"
			<< endl << "set grid;"
			<< endl << "set key autotitle columnhead;"
			<< endl << "set xlabel '" << xlabel << "';"
			<< endl << "set ylabel '" << ylabel << "';"
			<< endl << "set title '" << title << "';"
			<< endl << "plot '"	
			<< out_file << "' using " << col_x << ":" << col_y1 << "  with points pt 6, '"			//   
			<< out_file << "' using " << col_x << ":" << col_y2 << " with points pt 6, '"
			<< out_file << "' using " << col_x << ":" << col_y3 << " with points pt 6;";
		plot = plot_string.str();
		ofstream plot_command_file;
		plot_command_file.open("plotcommand");
		plot_command_file << plot;
		plot_command_file.close();
		int ret = system("gnuplot plotcommand");
		if (ret == 0)
		{
			show_string << "eog " << "out_" << col_x << col_y1 << col_y2 << col_y3 << ".png";
			string tmp = show_string.str();
			const char *show_graph = tmp.c_str();
			int retp = system(show_graph);
			if (retp != 0)
				cout << endl << "eog.exe failure" << endl;
		}
		else
		{
			cout << endl << "gnuplot failure\n";
		}
	}
}
//
void plotgraphs(string data_file, arma::Row<int> cols, string xlabel, double xmin, double xmax, string ylabel,
	double ymin, double ymax, string title, string linesorpoints, int show_graph)
{
	int k;
	stringstream plot_string;
	string show_string;
	int n = (int)cols.n_elem;
	string plot;
	string pngfile;
	pngfile = data_file + ".png";
	if (show_graph != 1)
	{
		cout << endl << "No graph for:  " << data_file << endl;
	}
	else
	{
		plot_string << "set terminal png;"
			<< endl << "set output " << "'" << pngfile << "';"
			<< endl << "set grid;"
			<< endl << "set key autotitle columnhead;"
			<< endl << "set xlabel '" << xlabel << "';"
			<< endl << "set ylabel '" << ylabel << "';"
			<< endl << "set title '" << title << "';"
			<< endl << "set xrange [" << xmin << ":" << xmax << "];"
			<< endl << "set yrange [" << ymin << ":" << ymax << "];"
			<< endl << "plot ";

		for (k = 1; k < n - 1; k++)
			plot_string << "'" << data_file << "' using " << cols(0) << ":" << cols(k) << "  with " << linesorpoints << ", ";
		plot_string << "'" << data_file << "' using " << cols(0) << ":" << cols(n - 1) << "  with " << linesorpoints<< "; ";

		plot = plot_string.str();
		ofstream plot_command_file;
		plot_command_file.open("plotcommand");
		plot_command_file << plot;
		plot_command_file.close();
		int ret = system("gnuplot plotcommand");
		if (ret == 0)
		{
			show_string = "mspaint " + pngfile;
			const char *show_graph = show_string.c_str();
			int retp = system(show_graph);
			if (retp != 0)
				cout << endl << "mspaint failure" << endl;
		}
		else
		{
			cout << endl << "gnuplot failure\n";
		}
	}
}
//	Copyright © 2020 Yashkir Consulting

