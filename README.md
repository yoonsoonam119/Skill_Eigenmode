This is a git for "An exactly solvable model for emergence and scaling laws".

* plot_power_law_all.py <br />
    * This code plots scaling law figure (Fig.2) in the paper, which is saved as plots/power_law_all. <br />
    * WARNING: The constant for time scaling law requires an integral calculation, which was done by Wolfram Mathematica and is hard-coded in the code.<br />   
 
* plot_compute_power_law_all.py <br />
    * This code plots the optimal compute scaling law figure (Fig. 3) in the paper, which is saved as plots/compute_power_law. <br />
    * WARNING: The constant for compute scaling law requires an integral calculation, which was done by Wolfram Mathematica and is hard-coded in the code.  <br />

* train_{time,data,parameter}.py <br />
    * This code runs the experiments for time, data, and parameter emergence and saves the results in data/zero/{time,data,parameter}. <br />
  
* plot_emergence_all.py <br />
    * After the data has been gathered in data/zero/{time,data,parameter}, this code plots the emergence plot (Fig.1) in the paper. <br />

* train1_{time,data,parameter}.py <br />
    * This code runs the experiments for time, data, and parameter for $n_s=1$ system and saves the results in data/zero/{time,data,parameter}. <br />

* plot_emergence_calibration_all.py <br />
    * After the $n_s=1$ system data has been gathered in data/zero/{time,data,parameter}, this code plots the calibration plot (Fig.11) in the appendix. <br />

* train_train_transformer.py <br />
    * Train the transformer (Fig.4) and save the data. <br />

* plot_time_transformer.py <br />
    * Create transformer figure (Fig.4) in the main text from learned data. <br />
