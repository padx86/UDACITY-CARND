#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 10;
double dt = .1;

const unsigned int n_states = 6; //[x,y,psi,v,cte,epsi]
const unsigned int n_controls = 2; //[steer angle: delta, accelaration/brake request: a]
const double Lf = 2.67;
double v_ref = 100;

//Create starting index for all states and controls used in the variable vector vars
size_t x_start = 0;
size_t y_start = N;
size_t psi_start = 2 * N;
size_t v_start = 3 * N;
size_t cte_start = 4 * N;
size_t epsi_start = 5 * N;
size_t delta_start = 6 * N;
size_t a_start = 7* N - 1;


class FG_eval {
public:
	// Fitted polynomial coefficients
	Eigen::VectorXd coeffs;
	double velocity;			//Add velocity for adaptive control
	FG_eval(Eigen::VectorXd coeffs, double velocity) { 
		this->coeffs = coeffs; 
		if (velocity < 10.) {
			this->velocity = 10.;
		}
		else {
			this->velocity = velocity;
		}
		 
	}

	typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;

	void operator()(ADvector& fg, const ADvector& vars) {
		fg[0] = 0; //Cost stored in fg(0)

		//Add penalties for crosstrack error, orientation error and velocity;
		for (unsigned int t = 0; t < N; t++) {
			fg[0] += 4.*CppAD::pow(vars[v_start + t] - v_ref, 2);
			fg[0] += 7000.*CppAD::pow(vars[cte_start + t], 2);
			fg[0] += 6500.*CppAD::pow(vars[epsi_start + t], 2);
		}

		//Add penalties to prevent massive actuator usage
		for (unsigned int t = 0; t < N - 1; t++) {
			fg[0] += 40.*this->velocity*CppAD::pow(vars[delta_start + t], 2); 
			fg[0] += 1.*CppAD::pow(vars[a_start + t], 2);
		}

		//Add penalties to prevent massive changes in actuator usage 
		for (unsigned int t = 0; t < N - 2; t++) {
			fg[0] += 150.*this->velocity*CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
			fg[0] += 1.*CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
			//fg[0] += 1.*this->velocity*CppAD::pow(vars[epsi_start + t + 1] - vars[epsi_start + t],2);
		}

		//Setup rest of vector
		//Use starting values
		fg[1 + x_start] = vars[x_start];
		fg[1 + y_start] = vars[y_start];
		fg[1 + psi_start] = vars[psi_start];
		fg[1 + v_start] = vars[v_start];
		fg[1 + cte_start] = vars[cte_start];
		fg[1 + epsi_start] = vars[epsi_start];
		
		//Setup further values using model
		for (unsigned int t = 0; t < N - 1; t++) {
			//Values at t
			AD<double> x_0 = vars[x_start + t];
			AD<double> y_0 = vars[y_start + t];
			AD<double> psi_0 = vars[psi_start + t];
			AD<double> v_0 = vars[v_start + t];
			AD<double> cte_0 = vars[cte_start + t];
			AD<double> epsi_0 = vars[epsi_start + t];
			//Actuation at t
			AD<double> delta_0 = vars[delta_start + t];
			AD<double> a_0 = vars[a_start + t];
			

			//Values at t+1
			AD<double> x_1 = vars[x_start + t + 1];
			AD<double> y_1 = vars[y_start + t + 1];
			AD<double> psi_1 = vars[psi_start + t + 1];
			AD<double> v_1 = vars[v_start + t + 1];
			AD<double> cte_1 = vars[cte_start + t + 1];
			AD<double> epsi_1 = vars[epsi_start + t + 1];
			//Don't consider actuation at t+1

			//Evaluate f(x=t) and atan(df/dx(x=t))
			AD<double> f_0 = coeffs[0] + coeffs[1] * x_0 + coeffs[2] * CppAD::pow(x_0, 2) + coeffs[3] * CppAD::pow(x_0, 3);
			AD<double> atandf_0 = CppAD::atan(coeffs[1] + 2. * coeffs[2] * x_0 + 3. * coeffs[3] * CppAD::pow(x_0, 2));

			//Add model contraints
			fg[2 + x_start + t] = x_1 - (x_0 + v_0*CppAD::cos(psi_0)*dt);
			fg[2 + y_start + t] = y_1 - (y_0 + v_0*CppAD::sin(psi_0)*dt);
			fg[2 + psi_start + t] = psi_1 - (psi_0 + v_0*(-delta_0)*dt / Lf);
			fg[2 + v_start + t] = v_1 - (v_0 + a_0*dt);
			fg[2 + cte_start + t] = cte_1 - ((f_0-y_0) + (v_0*CppAD::sin(epsi_0)*dt));
			fg[2 + epsi_start + t] = epsi_1 - ((psi_0 - atandf_0) + v_0*(-delta_0)*dt/Lf);
		}
	}
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
	bool ok = true;
	size_t i;
	typedef CPPAD_TESTVECTOR(double) Dvector;

	// TODO: Set the number of model variables (includes both states and inputs).
	// For example: If the state is a 4 element vector, the actuators is a 2
	// element vector and there are 10 timesteps. The number of variables is:
	//
	// 4 * 10 + 2 * 9

	size_t n_vars = N*n_states+(N-1)*n_controls;
	// TODO: Set the number of constraints
	size_t n_constraints = n_states*N;

	// Initial value of the independent variables.
	// SHOULD BE 0 besides initial state.

	Dvector vars(n_vars);

	for (i = 0; i < n_vars; i++) {
		vars[i] = 0;
	}

	Dvector vars_lowerbound(n_vars);
	Dvector vars_upperbound(n_vars);
	// TODO: Set lower and upper limits for variables.
	// Lower and upper limits for the constraints
	// Should be 0 besides initial state.
	Dvector constraints_lowerbound(n_constraints);
	Dvector constraints_upperbound(n_constraints);

	//variable bounds for x position
	for (i = x_start; i < y_start; i++) {
		vars_lowerbound[i] = -1000.;
		vars_upperbound[i] = -vars_lowerbound[i];
	}
	//variable bounds for y position
	for (i = y_start; i < psi_start; i++) {
		vars_lowerbound[i] = -1000.;
		vars_upperbound[i] = -vars_lowerbound[i];
	}
	//variable bounds for yaw angle
	for (i = psi_start; i < v_start; i++) {
		vars_lowerbound[i] = -1000.;
		vars_upperbound[i] = -vars_lowerbound[i];
	}
	//variable bounds for velocity
	for (i = v_start; i < cte_start; i++) {
		vars_lowerbound[i] = -50.;
		vars_upperbound[i] = 100.;
	}
	//variale bounds for cross track error
	for (i = cte_start; i < epsi_start; i++) {
		vars_lowerbound[i] = -100.;
		vars_upperbound[i] = -vars_lowerbound[i];
	}
	//variable bounds for yaw angle error
	for (i = epsi_start; i < a_start; i++) {
		vars_lowerbound[i] = -100.;
		vars_upperbound[i] = -vars_lowerbound[i];
	}
	//variable bounds for steer angle
	for (i = delta_start; i < a_start; i++) {
		vars_lowerbound[i] = -.43633231299; //-25deg in radians
		vars_upperbound[i] = -vars_lowerbound[i];
	}
	//Variable bounds for accelaration
	for (i = a_start; i < n_vars; i++) {
		vars_lowerbound[i] = -1.;
		vars_upperbound[i] = -vars_lowerbound[i];
	}

	//Set constraints
	for (i = 0; i < n_constraints; i++) {
		constraints_lowerbound[i] = 0.;
		constraints_upperbound[i] = 0.;
	}
	constraints_lowerbound[x_start] = state[0];
	constraints_lowerbound[y_start] = state[1];
	constraints_lowerbound[psi_start] = state[2];
	constraints_lowerbound[v_start] = state[3];
	constraints_lowerbound[cte_start] = state[4];
	constraints_lowerbound[epsi_start] = state[5];

	constraints_upperbound[x_start] = state[0];
	constraints_upperbound[y_start] = state[1];
	constraints_upperbound[psi_start] = state[2];
	constraints_upperbound[v_start] = state[3];
	constraints_upperbound[cte_start] = state[4];
	constraints_upperbound[epsi_start] = state[5];

	// object that computes objective and constraints
	FG_eval fg_eval(coeffs,state[3]);

	//
	// NOTE: You don't have to worry about these options
	//
	// options for IPOPT solver
	std::string options;
	// Uncomment this if you'd like more print information
	options += "Integer print_level  0\n";
	// NOTE: Setting sparse to true allows the solver to take advantage
	// of sparse routines, this makes the computation MUCH FASTER. If you
	// can uncomment 1 of these and see if it makes a difference or not but
	// if you uncomment both the computation time should go up in orders of
	// magnitude.
	options += "Sparse  true        forward\n";
	options += "Sparse  true        reverse\n";
	// NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
	// Change this as you see fit.
	options += "Numeric max_cpu_time          0.5\n";

	// place to return solution
	CppAD::ipopt::solve_result<Dvector> solution;

	// solve the problem
	CppAD::ipopt::solve<Dvector, FG_eval>(
		options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
		constraints_upperbound, fg_eval, solution);

	// Check some of the solution values
	ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

	// Cost
	auto cost = solution.obj_value;
	std::cout << "Cost " << cost << std::endl;

	// TODO: Return the first actuator values. The variables can be accessed with
	// `solution.x[i]`.
	//
	// {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
	// creates a 2 element double vector.
	vector<double> res;
	
	res.push_back(solution.x[delta_start]);	//Set first element of resulting vector to steer angle
	res.push_back(solution.x[a_start]);		//set second element of resulting vector to throttle/brake

	//Append mpc x & y vals
	for (unsigned int i = 0; i < N; i++) {
		res.push_back(solution.x[x_start + i]);
		res.push_back(solution.x[y_start + i]);
	}
	return res;
}