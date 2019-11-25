# CrystalDiff
This is a repo for the simulation from crystals.

## Important Notice
Notice that, even though, in reality, most of the time, people will not
need such complete implementation of the 2-beam dynamical diffraction
theory, because I am maintaining this repo by myself, I need to keep the 
code as short and clean as possible. Therefore, some of the useful
simplifications will not be used here, even though that may reduce the 
calculation time in some special situations. 

The ultimate requirement is that there is only 1 function for a specific
operation and that there is only 1 way for a specific simulation. 
In this way, if I need to make modifications to some functions, I do 
not need to worry about the compatibility with the other functions.
 
 
## ToDo list
1. Change the name of util.get_rocking_curve
2. There are two kinds of rotations in this package. I think the current
    handling is not ideal.