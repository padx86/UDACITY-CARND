#ifndef PPHELPERS_H
#define PPHELPERS_H

#include <math.h>	//min
#include <vector>	//vector
#include <stdlib.h>	//abs

using std::vector;
using std::min;
using std::pair;

double deg2rad(double x);
double rad2deg(double x);
double mph2mps(double x);
double mps2mph(double x);
double distance(double x1, double y1, double x2, double y2);
double frenetDistance(double car_s, double traffic_s);
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y);
int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y);
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y);
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y);
pair<vector<double>,vector<double>> getLocalTransformation(double global_x, double global_y, double global_yaw, vector<double> ptsx, vector<double> ptsy);

#endif // !PPHELPERS_H
