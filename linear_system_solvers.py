import numpy as np
import sys
import time

from ParametersDefault import Parameters
from qlsa import *
from iterative_refinement_linear_system_solver import *
from lss_output import *

#===========================================================================
# linear system solvers
#===========================================================================
def linear_system_solver(cofficent_matrix, rhs_vector, parameters):

	output 						= Output(cofficent_matrix, rhs_vector)

	if parameters.do_print:
		np.set_printoptions(precision=3, formatter={'float': '{:+.3f}'.format})
		print("\n",17*"-" + "Linear system Information" + 17*"-")
		print("{:<25}{:}".format("Linear system solver:", parameters.LS_Method))
		print("{:<25}{:}".format("Is quantum:", parameters.Is_Quantum))
		print("{:<25}{:}\n".format("Is simulator:", parameters.Is_Simulator))

		print("{:<25}{:}".format("LS_Precision:", parameters.LS_Precision))
		print("{:<25}{:}\n".format("IR_LS_Precision:", parameters.IR_LS_Precision))

		print("{:<25}{:}".format("LS_ScalFact:", parameters.LS_ScalFact))
		print("{:<25}{:}\n".format("LS_IncScalLim:", parameters.LS_IncScalLim))

		print("{:<25}{:.2e}".format("Norm of RHS vector:", np.linalg.norm(rhs_vector)))
		print("{:<25}{:.2e}".format("Norm of matrix:", np.linalg.norm(cofficent_matrix, 2)))
		print("{:<25}{:+.2e}".format("Minimum eigenvalue:", min(np.linalg.eigvals(cofficent_matrix))))
		print("{:<25}{:.2e}\n".format("Condition number:", np.linalg.cond(cofficent_matrix)))

		print("{:<25}{:}".format("RHS vector transpose:", rhs_vector))
		print("{:<25}[{:}".format("Coffcient matrix:", cofficent_matrix[0]))
		for row in cofficent_matrix[1:-1]: print("{:<25} {:}".format(" ", row))
		print("{:<25} {:}]".format(" ", cofficent_matrix[-1]))
		print(61*"-")
		print()

	output.start_time 			= time.time()


	exact_solution 				= np.linalg.solve(cofficent_matrix, rhs_vector)

	if parameters.LS_Method == "IR-LS":
		(output.solution, 
		output.LS_Precision, 
		output.iteration) 		= iterative_refinement_linear_system_solver(cofficent_matrix, rhs_vector, parameters)
		output.is_sign_changed 	= False

	elif parameters.LS_Method == "NC-LS":
		error_vector 			= np.random.uniform(low=-1.0, high=1.0, size=(rhs_vector.shape[0],)) 
		norm_one_error_vector 	= error_vector / np.linalg.norm(error_vector)
		final_error_vector 		= parameters.LS_Noise * norm_one_error_vector  # np.linalg.solve(cofficent_matrix, norm_one_error_vector)

		output.solution 		= np.linalg.solve(cofficent_matrix, rhs_vector + final_error_vector)
		output.is_sign_changed 	= False

	elif parameters.LS_Method == "LS" and parameters.Is_Quantum == False:
		output.solution 		= exact_solution
		output.is_sign_changed 	= False
		

	elif parameters.LS_Method == "LS" and parameters.Is_Quantum == True:
		(output.solution,
		output.is_sign_changed)	= QLSA(cofficent_matrix, rhs_vector, parameters)

	output.Is_Quantum 			= parameters.Is_Quantum
	parameters.QLSA_First_Run 	= False
	output.LS_Method 			= parameters.LS_Method
	output.exact_solution 		= exact_solution
	output.norm_of_difference	= np.linalg.norm(output.solution - exact_solution)
	output.norm_of_residual 	= np.linalg.norm(rhs_vector - np.matmul(cofficent_matrix, output.solution))
	
	output.LS_Precision 		= output.norm_of_difference / np.linalg.norm(exact_solution)


	output.end_time 			= time.time()

	if parameters.do_print == True: print(output)

	return (output.solution, output.norm_of_residual, output.is_sign_changed)
	


