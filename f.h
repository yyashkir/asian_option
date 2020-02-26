#pragma once

using namespace std;
#include <armadillo>

string date_now();
int sign(double x);

double cdf(double x);
double d1fwd(double F, double K, double T, double sigma);
double d2fwd(double d1, double T, double sigma);
double fwdBS(int cp, double F, double K, double T, double sigma);
double BS(int call_put, double spot_price, double strike_price, double interest_rate, double dividend_rate, double time_to_maturity, double volatility);
double interpolation(arma::Mat<double> y0, int n, int rank, double t);
double inter(arma::Mat<double> M, int n_rows, int n_col, double t);
double max(double a, double b);
double volatility(double tenor, double StoK_ratio, arma::Mat<double> vol_surf, int numb_t, int numb_ratios);
arma::Row<double> pfe(arma::Row<double> U, double confidence, arma::Row<double> wu);
arma::Mat<double> default_calc_v0(arma::Mat<double> defaults, arma::Mat<double> spreads, int numb_input_spreads, arma::Row<double> pay_times, int number_periods, int rank, arma::Mat<double> recovery_rates);
arma::Mat<double> matrix_power_q(arma::Mat<double> matrix, int rank, double q, int expansion_number);
arma::Mat<double> calc_marginal_matrix(arma::Cube<double> matrix_set, int payoff_numb, int rank);
arma::Mat<double> generator(arma::Mat<double> matrix, int rank, int expansion_number);
arma::Mat<double> powm(arma::Mat<double> D, int rank, int k);
arma::Mat<double> normalize_matrix(arma::Mat<double> matr, int rank);
arma::Row <double> get_col(arma::Mat <double> W, int i, arma::Row <double> v);

void plot2graphs(string out_file, int col_x, int col_y1, int col_y2, string xlabel, string ylabel, int show_graph);
void plot3graphs(string out_file, int col_x, int col_y1, int col_y2, int col_y3, string xlabel, string ylabel, string title, int show_graph);
void plotgraphs(string data_file, arma::Row<int> cols, string xlabel, double xmin, double xmax, string ylabel,
																double ymin, double ymax, string title, string linesorpoints, int show_graph);
void save_text(string txt, string filename);

//	Copyright © 2019 Yashkir Consulting
