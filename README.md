# Assignment 2 COMP3320 - Cloth Simulation

## Worth 20%, due by [5PM Friday 28th October, 2022](https://www.timeanddate.com/countdown/generic?iso=20221028T17&p0=57&msg=COMP3320%2FCOMP6464+2022+Assignment+2+Due&ud=1&font=cursive)

***Please read this document in full before attempting the assignment to ensure you are fully aware of the task requirements***

This project extends the concepts introduced in assignment 1 to build a simulation of a piece of cloth falling under gravity onto a stationary spherical ball. The cloth is modelled as a 2-D rectangular network, where node $`(x,y)`$ ($`x`$ and $`y`$ are integers) in an ($`N*N`$) square network interacts with all other nodes $`(x',y')`$ such that $`(x-\delta \le x' \le x+\delta,y-\delta \le y' \le y+\delta)`$ where $`\delta`$ is some (low value) integer. Thus if $`\delta=1`$ a typical node in the center of the cloth will interact with 8 neighboring nodes, while if $`\delta=2`$ there are 24 interactions to consider. When a node is located near the cloth edge there will be fewer interactions. Thus in contrast to the MD simulation, each node in the network interacts with a finite number of other nodes, making the evaluation of the total interaction potential scale as $`O(N)`$, where N is the number of nodes.

Between each pair of nodes $`i`$ and $`j`$, we define an interaction given by [Hooke's law](http://en.wikipedia.org/wiki/Hooke's_law). The potential energy is

```math
PE_{ij} = K * (R_{ij}-E_{ij})
```

Where $`K`$ is the force constant determining how stiff the spring (or cloth) is, $`R_{ij}`$ is the Euclidean distance between the two nodes and $`E_{ij}`$ is the equilibrium distance between these two nodes.
For example if node (1,1) and node (1,2) have equilibrium distance $`E_{(1,1)(1,2)}=d`$ then node (1,1) and node (2,2) have equilibrium distance $`E_{(1,1)(2,2)}=d\sqrt{2}`$. Exactly like the first assignment, the force on each node can be calculated using the first derivative of the potential energy such that the force contribution on node $`i`$ from node $`j`$ assuming they are within the required distance is

```math
F_{x_{ij}} = K * \frac{(R_{ij}-E_{ij}) * (x_i - x_j)}{R_{ij}}
```

Each node is given a constant mass $`m`$, noting that the force and acceleration are related by $`F=ma`$. The cloth is initially positioned in the xz plane and subjected to a gravitational force g in the y direction. As a consequence, the cloth will slowly fall under gravity.

Positioned below the cloth is a ball of radius, r, such that the centre of the cloth is located 1+r units above the centre of the ball. The cloth is allowed to fall under gravity until it collides with the ball. You follow the motions of the nodes using the same velocity verlet algorithm that you used for the MD program in assignment 1. The difference arises when a node in the cloth hits the ball. You detect this by noticing that the updated position of the node is within the radius of the ball. At this point, you move the node to the nearest point on the surface of the ball.

While building on the understanding you gained in assignment 1 concerning numerical simulation, an important part of this assignment will be the optimization of the simulation code including vectorization and parallelization.

### STEP 1: Collision Correction

***NOTE: This step has no marks associated with it, it is purely to assist you in understanding the numerical methods used in this assignment. You do not have to submit any python code.***

We provide you with the python code [cloth.py](cloth.py) that performs this basic simulation. Make sure you understand what this code is doing and try to run it on your machine (it should be very similar to the first assignment). As discussed in lectures, something is not quite right! If you inspect the code you will see that if the new coordinates for the cloth are within the ball, then that node of the cloth is moved back to the surface of the ball. This happens here

```python
for node in nodes:
    dist = node.pos - vector(myball.x, myball.y, myball.z)
    if dist.mag < myball.radius:
        fvector = dist / dist.mag * myball.radius
        node.pos = vector(myball.x, myball.y, myball.z) + fvector
```

A problem arises in that the velocity of the node remains the same. In this step modify the code such that the component of the velocity that is in the direction of the `fvector` above is set to zero. In other words, only the component of the velocity that is tangential to the surface is allowed to be non-zero following a collision of a cloth node with the ball. (If you are uncertain how to do this, try using the piazza forum).

### STEP 2: Performance Assessment

As discussed in lectures this repo contains a C version of the cloth simulation code that runs using openGL ([opengl_main.cpp](opengl_main.cpp)), and another version that just contains the numerical kernel ([kernel_main.cpp](kernel_main.cpp)). It is envisaged that you use the kernel only version for performance testing on Gadi, the visualization is only provided for general interest and to assist with debugging. Your overall task is to vectorize the code and parallelize it with OpenMP.

* Modify the code as provided so that when the cloth collides with the ball only the component of the velocity that is directed towards the centre of the ball is set to zero, i.e. what you did above for the Python code. Additionally, ensure that you add an additional damping factor of 0.1 for damping the velocities that are changed due to this collision correction - i.e. the updated velocity should be `0.1*velocity_parallel_ball_surface`. This allows us to incorporate friction between the ball and the cloth. The damping factor for the velocity update loop should stay 0.995.
* Run the code for a range of different problem sizes and gather some basic performance data. Data that you will then use to compare with when tuning and parallelizing the code. (Include this performance data in your write-up.)
* The supplied code is coded rather poorly in the computationally intensive parts. Identify as many problems as you can and fix them. Tune the code to perform as best as possible using a single core without manual vectorization. (You are required to submit a copy of this code.)

### STEP 3: SSE and OpenMP Vectorization

* Vectorize the above code by having it use SSE intrinsic functions. (You are required to submit a copy of this code.)
* Vectorize the code using OpenMP. Use Intel Advisor to produce a report indicating which loops have been successfully vectorized. Include this information in your write-up. (You are required to submit a copy of this code)
* Gather data to compare the performance of the vectorized and non-vectorized code. (Include this performance data  and analysis in your write-up.)

### STEP 4: Parallelization using OpenMP

Extend your SSE vectorized code as follows:

* Add another input flag to your program denoted -p, that reads in an integer that is used to set the maximum number of OpenMP threads used by your code.
* Parallelize your code using OpenMP by distributing a number of consecutive rows of nodes in the cloth to threads using a round-robin approach, i.e. use `SCHEDULE(STATIC, CHUNKSIZE)` directive. (You are required to submit a copy of this code.)
* Analyse the performance of your OpenMP code on Gadi giving consideration to the number of threads, problem size, chunksize and relating your results to Amdahl's law, cost of thread synchronization etc. (Include the performance data and analysis in your write-up.)

### STEP 5: Roofline analysis

* Produce a roofline plot (using Intel Advisor) for the bottleneck of your program. Analyse your roofline plot, discussing whether your code is compute-bound or memory-bound. Motivate your answer.

### STEP 6: COMP6464 ONLY

* Produce an OpenMP version of your code that splits both columns and rows among available threads in a block-wise decomposition. For example, if you have 4 threads and the cloth is $`16\times16`$, then each thread gets a block of $`8\times8`$ nodes. Adjust your make file so that it builds both versions of the OpenMP code. (You are required to submit a copy of this code.)
* Compare the performance of this version with the previous version. Is it what you expected? Explain your answer.

### MARKING

#### COMP3320 Requirements (Total Marks 28)**

You are required to submit a zip file that contains the following

* A CMake build system that if run will build all the different versions of the code required
* A version of the code that is modified such that only the velocity component of a node of the cloth colliding with the ball that is directed towards the centre of the ball is set to zero, and that has been optimized for execution on a single core. Call this version of the code `kernel_opt`
* A version of `kernel_opt` that has been vectorized using SSE intrinsic functions. Call this version of the code `kernel_sse`
* A version of `kernel_opt` that has been vectorized using OpenMP. Call this version of the code `kernel_vect_omp`
* A version of `kernel_sse` that has been parallelized using OpenMP. Call this version of the code `kernel_omp`
* A file called `writeup.pdf` that contains performance data with suitable commentary and analysis, including also your roofline analysis.
* A `USAGE.md` file that details the content of the zip file and specifies how to build the different versions of the code.

Marks will be awarded as follows

* Clean zip file, with CMake build system that works and USAGE.md file that is sensible (2 marks)
* Code for collision correction and basic performance tuning with a suitable writeup of performance results (6 marks)
* SSE vectorized code with a suitable writeup of performance results (6 marks)
* OpenMP vectorized code with a suitable writeup of performance results (4 marks)
* OMP parallel code with a suitable write-up of performance results (6 marks)
* A roofline plot obtained using Intel Advisor with suitable write-up explaining the performance (4 marks)

**PLEASE NOTE: marks will be awarded based on the following three metrics:**

1. Code correctness and performance
2. Performance data added to the report that proves performance improvement/deterioration
3. A written performance analysis explaining the data and the reasons for the performance improvement/deterioration.

#### COMP6464 Requirements (Total Marks 33)

1. Requirements as above for COMP3320. (28/33 Marks)
2. Version of the OpenMP parallel code that uses a block-wise distribution, cmake system that also builds this version, and analysis of performance as part of PDF writeup file. (5 marks)

### SUBMISSION (Both COMP3320 and COMP6464 all parts)

You are required to submit a single zipped file using Wattle. Name the zip file comp3320-2022-assignment-2.zip. This zip file can be generated using the command `zip comp3320-2022-assignment-2.zip ../comp3320-2022-assignment-2/ -r -x *.git* *__pycache__*` on a linux machine (such as Gadi) from the root folder of the repo. When unzipped this file should contain a README file that provides details of all other files, how to make and run your codes etc.

Late penalty of 5% of the awarded mark for each working day late or part thereof (as per course administrative handout).

#### Code Performance Ranking

On top of assessing the correctness of your code and marking the assignment, we will measure the performance of your code. More specifically, we will time your `kernel_omp`, that is the version of your code that has been both vectorized and parallelized using OpenMP. **_We will compile a list that ranks the performance of your codes_**.

The code will be timed when operating on three different input sets: 

* For all input sets 1 < delta < 10 and the mass associated with each node will be unitary.  

* For all input sets the number of nodes _per dimension_ will be 999 < N < 4000.
* For all input sets your code will be timed when running on a compute node of the Gadi supercomputer, using all 24 cores of a single Cascade Lake CPU and being allocated 96 GB of main memory.
* Your code will be timed in _pure kernel mode_, that is without rendering and visualization. 
* Your code will be built using the Intel compiler icc (ICC) 2021.6.0 20220226 (The default on Gadi)

PLEASE NOTE: if for any of the input sets the code gives incorrect values for the potential and kinetic energies it will be ranked as N/A. 

The list will be released as a standalone document on Wattle.

### Compiling and building the code

This code uses a CMake build system.
In order to build the code you will need to have CMake version >= 3.12 installed.
If CMake is not installed on your system, please follow the installation instructions
   on the CMake website https://cmake.org/install/

IMPORTANT NOTE: you will not be able to use the code in visualization mode on Gadi. 
                Gadi supports only the kernel mode execution of this code.
                You will be able to use the visualization mode on your machine 
                if you have OpenGL and GLUT installed. On Ubuntu (or WSL ubuntu) they can be installed
                with (`sudo apt-get install freeglut3-dev`). It may be possible to forward the X display to your own machine with `ssh -X` display forwarding (and a terminal that supports it such as MobaXterm) if you cannot get a local version to work.

In order to build the code on your machine execute the following code from within your assign2 directory:

    1. `mkdir build` 
    2. `cd build`
    3. `cmake ..`
    4. `make`

This will create two executables in the build directory, `opengl_main` and `kernel_main`.

IF YOU ARE ON GADI: please use the intel compiler. In order to do so, BEFORE 
    executing the above build instructions, please load the intel 
    compiler module and the cmake module by executing
    
    module load intel-compiler
    module load cmake

### Running the code

You can run the code simply with the command `./opengl_main` or `./kernel_main` from within the build folder. There are a number of additional parameters that you can modify when calling the program:

```
Nodes_per_dimension:             -n int 
Grid_separation:                 -s float 
Mass_of_node:                    -m float 
Force_constant:                  -f float 
Node_interaction_level:          -d int 
Gravity:                         -g float 
Radius_of_ball:                  -b float 
offset_of_falling_cloth:         -o float 
timestep:                        -t float 
num iterations:                  -i int
Timesteps_per_display_update:    -u int 
Perform X timesteps without GUI: -x int
Rendermode (1 for face shade, 2 for vertex shade, 3 for no shade):
                                 -r (1,2,3)
```

So for example, the command ```./opengl_main -n 20 -i 1000``` will run the cloth code for 1000 timesteps with 20 nodes per dimension in the cloth using the visualization.

## Testing the code

Just like the first assignment, you have been provided with several sample outputs in the ```auto_test_outputs``` folder and a test script that can be run using ```python3 auto_test.py```. This script is new this year so please let us know with a 2022 post on piazza if you believe there are any issues and we will investigate. A similar script will be used to test your final submission so please make sure your compiled executables match the names used in the test script **exactly**
