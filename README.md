# Fitbox

A tool for trying and running various optimization strategies

Declare an objective function and optimize accross one or more algorithms

Currently implemented:

 * LBFGS – running a backtracking line search on the first iteration and stepping forward, following Wolfe conditions on subsequent iterations

 * SGD with optional momentum

 * Logistic regression – with optional regularization

 * Feature hashing – a la VW

 * Designed to run on functions of offline data or data represented in Spark RDDs
