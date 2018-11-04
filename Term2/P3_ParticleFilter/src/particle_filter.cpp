/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	this->num_particles = 150;									//Set number of particles
	this->particles.resize(this->num_particles);				//resize particles vector
	this->weights.resize(this->num_particles);					//resize weights vector
	std::fill(this->weights.begin(), this->weights.end(), 1.);	//Fill weights

	default_random_engine gen;									//Initialize default random engine
	normal_distribution<double> dist_x(x, std[0]);				//Initialize normal distribution in x, take mean of gps x coordinate and std dev in x
	normal_distribution<double> dist_y(y, std[1]);				//Initialize normal distribution in y, take mean of gps y coordinate and std dev in y
	normal_distribution<double> rot_theta(theta, std[2]);		//Initialize normal distribution in x, take mean of gps x coordinate

	//Create initial particles
	unsigned int id_counter = 0;
	for (vector<Particle>::iterator it = this->particles.begin(); it != this->particles.end(); it++) {
		it->x = dist_x(gen);
		it->y = dist_y(gen);
		it->theta = rot_theta(gen);
		it->weight = 1.;
		it->id = id_counter++;
		//std::cout << "Initialized particle id: " << it->id << ", x: " << it->x << ", y:" << it->y << endl;
	}
	
	this->is_initialized = true;			//Initialization compltete
	this->denums_initialized = false;		//Initialization for constant multiv. gaussian dist. incomplete
	return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	//Update all particles
	default_random_engine gen;
	
	double dyaw_dt = yaw_rate*delta_t;

	for (vector<Particle>::iterator it = this->particles.begin(); it != this->particles.end(); it++) {
		if(abs(yaw_rate)>.00001){
			it->x += (velocity / yaw_rate)*(sin(it->theta + dyaw_dt)-sin(it->theta));
			it->y += (velocity / yaw_rate)*(cos(it->theta) - cos(it->theta + dyaw_dt));
			it->theta += dyaw_dt;
		}
		else {
			it->x += cos(it->theta)*dyaw_dt;
			it->y += sin(it->theta)*dyaw_dt;
		}
		//Add noise
		it->x = normal_distribution<double>(it->x, std_pos[0])(gen);
		it->y = normal_distribution<double>(it->y, std_pos[1])(gen);
		it->theta = normal_distribution<double>(it->theta, std_pos[2])(gen);

		//std::cout << "x=" << it->x << " ,y=" << it->y << "theta=" << it->theta << ", yaw rate=" << yaw_rate << ", velocity=" << velocity << endl;
	}
	return;
}

vector<LandmarkObs> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double min_dist;
	double cur_dist;
	LandmarkObs lm;
	vector<LandmarkObs> associated;
	//Cycle observed landmarks and find corresponding map landmarks
	for (vector<LandmarkObs>::iterator it = observations.begin(); it != observations.end(); it++) {
		min_dist = 1000000.;
		for (vector<LandmarkObs>::iterator it2 = predicted.begin(); it2 != predicted.end(); it2++) {
			cur_dist = dist(it->x,it->y,it2->x,it2->y);
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				lm = (*it2);
			}
		}
		associated.push_back(lm);
	}
	return associated;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	//Initialize constant terms of P(x,y) once as no change in std_landmark
	if (!this->denums_initialized) {
		this->denum_x = 2.*std_landmark[0] * std_landmark[0];					//Initialize first denumerator inside exponent for P(x,y)
		this->denum_y = 2.*std_landmark[1] * std_landmark[1];					//Initialize second denumerator inside exponent for P(x,y)
		this->gaus_norm = 1. / (2.*M_PI*std_landmark[0] * std_landmark[1]);		//Initialize normalization term for P(x,y)
		this->denums_initialized = true;										//Constant parts of gaussian dist initialized
		std::cout << "Constant gaus terms initialized" << endl;
	}

	//Initialize vectors
	vector<LandmarkObs> map_observed(observations.size());
	vector<LandmarkObs> predicted;
	vector<LandmarkObs> associated;

	//Single Landmark
	LandmarkObs lm;

	unsigned int w_counter = 0;	//weights counter
	unsigned int lm_counter;	//landmark counter

	for (vector<Particle>::iterator it = this->particles.begin(); it != this->particles.end(); it++) {

		//hgt coordinates
		lm_counter = 0;
		for (vector<LandmarkObs>::const_iterator it2 = observations.begin(); it2!=observations.end(); it2++) {
			lm.x = cos(it->theta)*it2->x - sin(it->theta)*it2->y + it->x;
			lm.y = sin(it->theta)*it2->x + cos(it->theta)*it2->y + it->y;
			lm.id = it2->id;
			map_observed[lm_counter++] = lm;
		}

		//Get Landmarks within sensor Range
		predicted.clear();		//clear vector before reusing
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			if (dist(it->x, it->y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) < sensor_range) {
				LandmarkObs lm;
				lm.x = map_landmarks.landmark_list[j].x_f;
				lm.y = map_landmarks.landmark_list[j].y_f;
				lm.id = map_landmarks.landmark_list[j].id_i;
				predicted.push_back(lm);
			}
		}
		
		//Get associated landmarks
		associated = dataAssociation(predicted, map_observed);
		
		//Recalc weights
		it->weight = 1.;
		for (unsigned int j = 0; j < associated.size(); j++) {
			it->weight *= this->gaus_norm*exp(-(pow((map_observed[j].x - associated[j].x), 2) / this->denum_x + pow((map_observed[j].y - associated[j].y), 2) / this->denum_y));
		}
		weights[w_counter++] = it->weight; //add weights
	}
	return;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	discrete_distribution<int> distrib(weights.begin(),weights.end());
	default_random_engine gen;

	vector<Particle> resampled(this->num_particles);
	unsigned int idx;
	for (unsigned int i = 0; i < this->num_particles; i++) {
		idx = distrib(gen);
		resampled[i] = this->particles[idx];
		//std::cout << "Particle Nr:" << i << " replaced by Particle Nr:" << idx << endl;
	}
	this->particles = resampled;
	return;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
