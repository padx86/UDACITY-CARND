# PID Controller project
#### Project submission
## Remarks
* src and build directories containing files 
* CMakeLists.txt is included and unchanged
* Description can be found below

## Model description
#### Content of main.cpp
As the model has a latency of 0.1 seconds, the first step was to predict the vehicle states(without errors) using the bicycle kinematics model 0.1 seconds in the future in the global coordinate system.
```
px += v*cos(psi)*dt;
py += v*sin(psi)*dt;
psi += v*(-steer_value)*dt / Lf;
v += throttle*dt;
```
With predicted vehicle states a homogenous transformation was performed to transform the waypoints retrieved from the simulator to the vehicle coordinate system.
```
//Using cos(-psi)=cos(psi); sin(-psi) = -sin(psi)
           [cos(psi)  sin(psi) 0][ptsx[i]-px]
x_local =  [-sin(psi) cos(psi) 0][ptsy[i]-py]
  		   [0         0        1][0         ]
``` 
After preforming coordinate transformation the polynomial was fitted using:
```
Eigen::VectorXd coeffs = polyfit(x_pts, y_pts, 3);
```
Having the coeffs of the polynomial initial errors were processed.
The initial cross track error(cte) is ``f(x=0) = coeffs[0]`` as polyfit returns coeffs as ``coeffs[0]+coeffs[1]*x+coeffs[2]*x^2+...+coeffs[n]*x^n``. The initial orientation error was computed by deriving f(x).
As ``df/dx = coeffs[1]+2*coeffs[2]*x+..+n*coeffs[n]*x^(n-1)`` the derivation of x=0 is ``df/dx(x=0)=coeffs[1]`` and therefore the orintation error: ``epsi=-atan(coeffs[1])``.
With all states available for the optimization issue the method mpc.Solve(state,coeffs)(described later) was called with the initial states and polynomial coeffs returning the processed solution as vector containing [steer_value,throttle_value,trajectory_x0,trajectory_y0,...,trajectory_xn,trajectory_yn]

#### Content of mpc.cpp
The MPC.cpp is entered at MPC::Solve from the main function.
At first the number of variables for the optimization solver is defined to the number of states(n_states) multiplied by the desired number of predictions(N). The controls are also added to the number of variables as to be processed by the solver. Then the number of constraints is set up to n_states x N
The boundaries for all states and controls were defined especially regarding the maximum and minimum values for the vehicles controls. This step is important as the solver could find a solution to the optimization having less cost but physically impossible controls.
The boundaries for all future controls were set to:
```
vars_lowerbound[i] = -.43633231299; //-25deg in radians
vars_upperbound[i] = -vars_lowerbound[i];
```
and 
```
vars_lowerbound[i] = -1.;
vars_upperbound[i] = -vars_lowerbound[i];
```
All constraints other than the initial constraints were set to 0. The initial upper- and lowerbound constraints were set to the vehicles state as change of this state during optimization is not possible.

As last step of the preparation before the actual optimization, the cost function had to be setup in the FG_eval class.
The constructor was modified to take in the current velocity as optional argument. The reason was to make the cost function adaptive to the current velocity. The cost for the steering actuation and the change in steering actuation was multiplied by the current velocity to put higher penalties on the actuation at higher velocities to prevent effects like swerving or drifting:
```
fg[0] += 40.*this->velocity*CppAD::pow(vars[delta_start + t], 2); 
fg[0] += 150.*this->velocity*CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2); 
```
All other parts of the cost function were chosen constant, multiplied with a constant factor and squared. The final cost function for optimization for each time step is:
```
for (unsigned int t = 0; t < N; t++) {
	fg[0] += 4.*CppAD::pow(vars[v_start + t] - v_ref, 2);
	fg[0] += 6000.*CppAD::pow(vars[cte_start + t], 2);
	fg[0] += 5000.*CppAD::pow(vars[epsi_start + t], 2);
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
```
Finally the model equations describing the error over state development had to be appended to the fg vector for each time step(t) using:
```
AD<double> f_0 = coeffs[0] + coeffs[1] * x_0 + coeffs[2] * CppAD::pow(x_0, 2) + coeffs[3] * CppAD::pow(x_0, 3);
AD<double> atandf_0 = CppAD::atan(coeffs[1] + 2. * coeffs[2] * x_0 + 3. * coeffs[3] * CppAD::pow(x_0, 2));

//Add model constraints
fg[2 + x_start + t] = x_1 - (x_0 + v_0*CppAD::cos(psi_0)*dt);
fg[2 + y_start + t] = y_1 - (y_0 + v_0*CppAD::sin(psi_0)*dt);
fg[2 + psi_start + t] = psi_1 - (psi_0 + v_0*(-delta_0)*dt / Lf);
fg[2 + v_start + t] = v_1 - (v_0 + a_0*dt);
fg[2 + cte_start + t] = cte_1 - ((f_0-y_0) + (v_0*CppAD::sin(epsi_0)*dt));
fg[2 + epsi_start + t] = epsi_1 - ((psi_0 - atandf_0) + v_0*(-delta_0)*dt/Lf);
```
With all required arguments defined for ipopt to solve the optimization issue, the function: 
```
CppAD::ipopt::solve<Dvector, FG_eval>(
		options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
		constraints_upperbound, fg_eval, solution);
```
is called and returns the solution containing values for all optimized states and controls.

#### Choice of parameters
* *dt* was chosen to 100ms for latency compensation after first step: 
* *N*  was chosen to 10 giving future values up to 1 second. At higher velocities this seems to be suitable for the given (racing) track containing no sharp corners.
* The choice of the factors for the single components of the cost function was chosen to minimize cte and orientation error while keeping the actuator use moderate, especially at high velocities.

#### Discussion
As can be seen in the MPC project the controller performs much better than a simple PID Controller due to future state prediction.
By additional tuning and modification in the cost function even better control performance could be achieved
