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

        num_particles = 100;

        std::default_random_engine gen;

        std::normal_distribution<double> N_x(x, std[0]);
        std::normal_distribution<double> N_y(y, std[1]);
        std::normal_distribution<double> N_theta(theta, std[2]);

        for (int i = 0; i < num_particles; i++) 
        {
            Particle particle;
            particle.id = i;
            particle.x = N_x(gen);
            particle.y = N_y(gen);
            particle.theta = N_theta(gen);
            particle.weight = 1;

            particles.push_back(particle);
            weights.push_back(1);
        }

        is_initialized = true; 
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  for (int i = 0; i < num_particles; i++)
  {
    
    double new_x;
    double new_y;
    double new_theta;

    if (fabs(yaw_rate) < 0.0001)
    {
      new_theta = particles[i].theta;
      new_x = particles[i].x + (velocity * delta_t * cos(new_theta));
      new_y = particles[i].y + (velocity * delta_t * sin(new_theta));
    } 
    else 
    {
      new_theta = particles[i].theta + yaw_rate * delta_t;
      new_x = particles[i].x + (velocity / yaw_rate) * (sin(new_theta) - sin(particles[i].theta));
      new_y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(new_theta));
    }

    normal_distribution<double> N_x(new_x, std_pos[0]);
    normal_distribution<double> N_y(new_y, std_pos[1]);
    normal_distribution<double> N_theta(new_theta, std_pos[2]);

    particles[i].x =  N_x(gen);
    particles[i].y = N_y(gen);
    particles[i].theta = N_theta(gen);
  } 

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned int i = 0; i < observations.size(); i++) 
  {   
    LandmarkObs o = observations[i];
    double min_dist = numeric_limits<double>::max();
    int map_id = -1; 

    for (unsigned int j = 0; j < predicted.size(); j++) 
    {
      LandmarkObs p = predicted[j];
      double cur_dist = dist(o.x, o.y, p.x, p.y);

      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        map_id = p.id;
      }
    }

    observations[i].id = map_id;
  }


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

  double weights_sum = 0.0;

  for (int i=0; i<num_particles; i++) 
  {      
    std::vector<LandmarkObs> observations_particle_space;   

    const double cos_particle_theta = std::cos(particles[i].theta);
    const double sin_particle_theta = std::sin(particles[i].theta);   

    for (int i_obs=0; i_obs<observations.size(); i_obs++)
    {
      LandmarkObs landmark;
      landmark.x = particles[i].x + observations[i_obs].x*cos_particle_theta - observations[i_obs].y*sin_particle_theta;
      landmark.y = particles[i].y + observations[i_obs].x*sin_particle_theta + observations[i_obs].y*cos_particle_theta;
      observations_particle_space.push_back(landmark);
    }  

    std::vector<LandmarkObs> landmarks_to_observed;
    double weight = 1.0;   

    for (int j=0; j<observations_particle_space.size(); j++)
    {
      double min_distance = sensor_range;
      const Map::single_landmark_s* closest_lm = nullptr;
      double obs_x = observations_particle_space[j].x;
      double obs_y = observations_particle_space[j].y;     

      for (int k=0; k<map_landmarks.landmark_list.size(); k++)
      {
        const double lm_x = map_landmarks.landmark_list[k].x_f;
        const double lm_y = map_landmarks.landmark_list[k].y_f;
        const double distance = dist(obs_x, obs_y, lm_x,  lm_y);
        if (distance < min_distance)
        {
          min_distance = distance;
          closest_lm = &map_landmarks.landmark_list[k];
        }
      }

      if (closest_lm) 
      {
        const double mvg_part1 = 1/(2.0*M_PI*std_landmark[0]*std_landmark[1]);
        const double x_dist    = obs_x - closest_lm->x_f;
        const double y_dist    = obs_y - closest_lm->y_f;
        const double mvg_part2 = ((x_dist*x_dist) / (2*std_landmark[0]*std_landmark[0])) + ((y_dist*y_dist) / (2*std_landmark[1]*std_landmark[1]));
        const double mvg       = mvg_part1 * std::exp(-mvg_part2);
        weight *= mvg;
      }
    }   
    weights[i] = weight;
    particles[i].weight = weight;
    weights_sum += weight;
  }    

  for (int i=0; i<num_particles; i++)
      particles[i].weight /= weights_sum;

}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
    uniform_real_distribution<double> dis(0, 1);
    vector<Particle> resample_particles;

    resample_particles.reserve(particles.size());

    for (int i=0; i<particles.size(); i++)
    {
        double rand_num = dis(gen);
        double particle_weights = 0.0;
        int j = 0;
        while (particle_weights < rand_num)
        {
            if (j >= particles.size())
              j = 0;
            particle_weights += particles[j].weight;
            j++;
        }

        resample_particles.push_back(particles[j-1]);
    }
    particles = resample_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
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
