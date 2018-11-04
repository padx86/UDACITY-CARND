# Extended Kalman Filter
#### Project submission
### Remarks V2
* Corrected typo causing compile error
* Changed noises from 9.0 to 30.0 for Q matrix to meet RMSE requirements
* Included normalization for atan in ekf_.UpdateEKF method
* Precalculated P_*H^T to prevent multiple calculation
### Remarks V1
* src and build directories containing files 
* CMakeLists.txt is included and unchanged
* RMSE is not within boundaries defined in project specification