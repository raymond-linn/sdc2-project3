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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// initialize number of particles
	num_particles = 300;

	// generate normal gaussian distribution to all particles
	// from lesson 14 lecture 3, 4, 5
	default_random_engine gen;

	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);

	for (int i = 0; i < num_particles; i++) {
		double sample_x, sample_y, sample_theta;

		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		// prepare particle here
		Particle particle;
		particle.id = i;
		particle.x = x + sample_x;
		particle.y = y + sample_y;
		particle.theta = theta + sample_theta;
		// weights to 1
		particle.weight = 1.0;

		// push weights and particles to current particle list and weights
		weights.push_back(particle.weight);
		particles.push_back(particle);
	}

	// set is_initialized
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// normal gaussian distribution generator
	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++){

		// check if yaw_rate is less than or equal to zero
		// from lesson 14 Lecture 6, 7, 8
		if (fabs(yaw_rate) > 0.001){
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		else{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}

		// add the guassian noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// Nearest neighbor as lesson 14 lecture 9, 10
	// loop through measurements
	for (int i = 0; i < observations.size(); i++){
		double min_distance = numeric_limits<double>::max();
		int min_index = -1;

		// loop predicted to calculate min distance from observations
		for (int j = 0; j < predicted.size(); j++){
			double cur_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (cur_distance < min_distance){
				min_distance = cur_distance;
				min_index = j;
			}
		}
		observations[i].id = predicted[min_index].id;
		observations[i].x -= predicted[min_index].x;
		observations[i].y -= predicted[min_index].y;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html


	// 0) initialize the weights
	weights.clear();

  // 1) update particles
	for (int i = 0; i < num_particles; i++) {
    // to store and calculate data Association
    vector<LandmarkObs> landmarks;
    vector<LandmarkObs> obs;
		// 2) transform coordinates
		// get landmarks in given sensor range
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++){

      double cur_distance = dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, particles[j].x, particles[j].y);
			if (cur_distance <= sensor_range){
				// populate landmarks
				LandmarkObs lm;
				lm.id = map_landmarks.landmark_list[j].id_i;
				lm.x = map_landmarks.landmark_list[j].x_f;
				lm.y = map_landmarks.landmark_list[j].y_f;
				// push it to landmarks
				landmarks.push_back(lm);
			}
		}

		// go through to get observations
		for (int j = 0; j < observations.size(); j++ ){
			double cur_distance = dist(observations[j].x, observations[j].y, 0, 0);
			if(cur_distance <= sensor_range){
				LandmarkObs lm;
				lm.id = -1;
				lm.x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
				lm.y = particles[i].y + observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta);
				// push it to observations
				obs.push_back(lm);
			}
		}

		// 3) associate data
		dataAssociation(landmarks, obs);

		// 4) update weights
		// calculate weight
		double weight = 1.0;
		double sigma_x = std_landmark[0];
		double sigma_y = std_landmark[1];

		// x - mu_x = d_x which is obs.x as Bivariate case so is to y - mu_y = d_y
		for (int j = 0; j < obs.size(); j++){
			double d_x = obs[j].x;
			double d_y = obs[j].y;

			// from lesson 14 lecture 13 transformation and association
			// weight *=  (c1 * exp(-0.5 * (c2 + c3)));
      weight *= (1.0 / (2.0 * M_PI * sigma_x * sigma_y)) * exp(-0.5 * ((d_x * d_x) / (sigma_x * sigma_x) + (d_y * d_y) / (sigma_y * sigma_y)));
		}

		// update weights
		weights.push_back(weight);
		particles[i].weight = weight;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// set discrete distribution
	default_random_engine gen;
	discrete_distribution<int> dist(weights.cbegin(), weights.cend());

	// Copy current particles
	vector<Particle> particles_tmp = particles;
	// prepare the particles list
	particles.clear();

	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		int index = dist(gen);
		particle = particles_tmp[index];
		particles.push_back(particle);
	}

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
