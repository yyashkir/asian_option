//	Copyright © 2019 Yashkir Consulting

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <sstream>
#include "f.h"
#include <armadillo>
using namespace std;

string strike_type;
string out_file;
double strike_fixed;
double spot_price;
double interest_rate;
double dividend_rate;
double sigma;
double maturity;
double time_step;
double confidence_level;	// for pfe calculation ( 0<...<1)
double u;
double pu;
double pm;
double pd;

int time_points_number;
int callput;

arma::Row<double> t;
arma::Mat<double> sji;		//node prices
arma::Mat<double> pji;		//node probabilities
arma::Mat<double> eap;	    //expected average prices
arma::Mat<double> vji;		//option node values
arma::Mat<double> Qj;		//quantiles ()


void  read_input(string data_file)
{
	int  k;
	string delimiter="::";
	string s;
	ifstream infile(data_file);
	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ; // substring after delimiter
	callput=stoi(s);

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	strike_type = s;

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	strike_fixed = atof(s.c_str());

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	spot_price = atof(s.c_str());

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	interest_rate = atof(s.c_str());

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	dividend_rate = atof(s.c_str());

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	sigma = atof(s.c_str());

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	maturity= atof(s.c_str());

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	time_step= atof(s.c_str());

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	confidence_level= atof(s.c_str());

	infile >> s ;
	s = s.substr(s.find(delimiter)+delimiter.length()) ;
	out_file= s;

	infile.close();
}

void setting_variables()
{
	int k;
	time_points_number = (int)( maturity /  time_step) + 1;
	time_step =  maturity / (time_points_number  - 1);

	u = exp(sigma * sqrt(2 * time_step));  cout<<"u="<<u<<endl;

	double r = interest_rate;
	double q = dividend_rate;
	double dt = time_step;
	pu = pow( ( sqrt(u) * exp((r - q) * dt / 2) - 1 )  / ( u - 1. )   ,2);
	pd = pow( ( sqrt(u) * exp((r - q) * dt / 2) - u )  / ( u - 1. )   ,2);
	pm = 1. -  pu -  pd;
	cout<<"pu="<<pu<<" pm="<<pm<<"  pd="<<pd<<endl;
	if (dt * pow(r - q,2) >= 2 * sigma*sigma )
	{
		cout<<endl<<"ERROR: time step is too large"<<endl;
		exit(0);
	}

	int n = time_points_number;
	sji.set_size(n,2*n-1);
	sji.fill(0);

	pji.set_size(n,2*n-1);
	pji.fill(0);

	eap.set_size(n,2*n-1);
	eap.fill(0);

	vji.set_size(n,2*n-1);
	vji.fill(0);
}

void trinomial_tree()
{
	int i, j;
	double S = spot_price;
	int n = time_points_number;
	sji(0, 0) = S;
	for (j = 0; j < n; j++)
	{
		for (i = 0; i <= 2*j; i++)
		{
			sji(j, i) = S * pow(u, i - j);
		}
	}
//	sji.raw_print("sji");

	for (j = 0; j < n; j++)
	{
		for (i = 0; i <= 2*j; i++)
		{
			if(j==0 && i == 0)
				pji(j, i) = 1;
			if(j==1 && i == 1)
				pji(j, i) = pm;
			if(j>=1 && i == 2*j)
				pji(j,i) = pji(j-1,i-2) * pu;
			if(j>=1 && i == 0)
				pji(j,i) = pji(j-1,0) * pd;
			if(j>=2 && i == 2*j-1 )
				pji(j,i) = pji(j-1,i-1) * pm + pji(j-1,i-2) * pu ;
			if(j>=2 && i == 1)
				pji(j,i) = pji(j-1,0) * pm + pji(j-1,1) * pd ;
			if(j>=2 && i >=2 && i <= 2*j-2)
				pji(j,i) = pji(j-1,i-1) * pm + pji(j-1,i) * pd + pji(j-1,i-2) * pu ;
		}
	}
//	pji.raw_print("pji");
}

double calc_eap(int jnode, int inode)
{
	int j, i;
	int n = time_points_number;
	double av = 0;
	for (j = 0; j <= jnode; j++)
	{
		for (i = 0; i <= 2*j; i++)
		{
			if( j <= jnode && i<=inode && i <= 2*j && i>= inode-2*(jnode-j) && i >=0   )
			{
				av = av + sji(j,i) * pji(j,i);
			}
		}
	}
	for (j = jnode+1; j <n; j++)
	{
		for (i = inode; i <= inode + 2*(j - jnode) ; i++)
		{
				av = av + sji(j,i) * pji(j,i);
		}
	}
	av = av / n;
	return av;
}

void save_eap(string out_file)
{
	int j, i;
	int n = time_points_number;
	ofstream myfile;
	myfile.open (out_file);
	myfile << "node numbers:";
	for(j=0;j<n;j++)
	{
        myfile<<endl<<" ";
        for(i=0;i<2*j+1;i++)
            myfile<<eap(j,i)<<" ";
    }
	myfile.close();
}

void expected_average_prices()
{
	int j, i;
	int n = time_points_number;
	for (j = 0; j < n; j++)
	{
		for (i = 0; i <= 2*j; i++)
		{
			eap(j,i) = calc_eap(j,i);
		}
	}
//	eap.raw_print("eap");
	save_eap("eap");
}



void option_vji_fixed_strike()		// or "average rate": payoff = [average - strike]+
{
	int j, i;
	int n = time_points_number;
	double tau;
	for (j = 0; j < n; j++)
	{
		tau = (n - j - 1) * time_step;
		for (i = 0; i <= 2*j; i++)
		{
			vji(j,i) = BS(callput, eap(j,i), strike_fixed, interest_rate, dividend_rate, tau, sigma);
			
		}
	}
//	vji.raw_print("strike fixed");
}

void option_vji_float_strike()		// or "floating rate": payoff = [price - average]+
{
	int j, i;
	int n = time_points_number;
	double tau;
	for (j = 0; j < n; j++)
	{
		tau = (n - j - 1) * time_step;
		for (i = 0; i <= 2*j; i++)
		{
			vji(j,i) = BS(callput, sji(j,i), eap(j,i), interest_rate, dividend_rate, tau, sigma);
		}
	}
//	vji.raw_print("strike float vji");
}

double c_pfe (arma::Mat<double> vp, int m, double c)
{
	int i, k;
	arma::vec pc(m);
	for(i=1;i<=m;i++)
		pc(i-1) = (i *1.0)/ (1.0*m);
	k = int(c * (m-1));
	arma::vec w(m);
	w= vp.unsafe_col(1);	
	w = sort(w);	
	double pfe = w(k) + (c - pc(k)) * (w(k+1) - w(k) ) / (pc(k+1) -pc(k) );		// interpolated pfe value
	return max(0,pfe);
}

void calc_pfe (int j)
{
	int i, m;
	arma::Mat<double> vp;
	arma::vec v;
	m = 2*j+1;
	vp.set_size(m,2);
	v.set_size(m);
	for(i=0;i<m;i++)
	{
		v(i) = vji(j,i);
		vp(i,0) = pji(j,i);
		vp(i,1) = vji(j,i);
	}
	Qj(j,0) = j*time_step;
	Qj(j,1) = c_pfe(vp,m,1.-confidence_level);
	Qj(j,2) = mean(v);
	Qj(j,3) = c_pfe(vp,m,confidence_level);
}

void ee_pfe()
{
	int j, i;
	int n = time_points_number;
	Qj.set_size(n,4);
	Qj.fill(0);
	Qj(0,1) = Qj(0,2) = Qj(0,3) = vji(0,0);

	for(j=1; j< n; j++)
			calc_pfe(j);
//	Qj.raw_print("\npfe");
}

void save_pfe(string out_file)
{
	int j, i;
	int n = time_points_number;
	ofstream myfile;
	myfile.open (out_file);
	myfile << "time pfe-dn ee pfe-up";
	for(j=0;j<n;j++)
		myfile<<endl<<Qj(j,0)<<" "<<Qj(j,1)<<" "<<Qj(j,2)<<" "<<Qj(j,3);
	myfile.close();
}

int main(int argc, char **argv)
{
	read_input(string(argv[1]));

	setting_variables();

	trinomial_tree();

	expected_average_prices();

	if(strike_type == "fixed")
	option_vji_fixed_strike();

	if(strike_type != "fixed")
	option_vji_float_strike();

	ee_pfe();

	save_pfe(out_file);

	Qj.shed_col(0);
	plotgraphs(out_file, { 1,2,3,4 }, "years", 0, maturity, "Asian option PFE & EE", 0, 1.1 * Qj.max(), "Price: " + to_string(Qj(0,1)), "linespoints pt 7 lw 2", 1);

	return 0;
}
//	Copyright © 2019 Yashkir Consulting
// http://www.coggit.com/freetools
