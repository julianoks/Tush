# Overview
Tush is a Push programming language that includes tensor operations. 
Like Push, Tush is not intended to be written by human programmers, but rather to be used as a representation for evolving programs.
A further deviation from traditional Push implementions, Tush programs may experience an "epigenetic" or "experiential" learning phase, 
where numeric values are refined by gradient descent. This document aims to introduce the Tush representation and TushGP system, 
discuss the Push family of languages, computational graphs, Tush and "experiential" learning, and finally identify experiments and directions for the Tush system.

# Push Languages
Push is a family of languages intended for evolutionary search. 
Briefly, Push languages are executed using a stack based architecture (think [Forth](https://en.wikipedia.org/wiki/Forth_(programming_language))) 
with a characterisitcally robust execution strategy. 
A Push program is a (possibly nested) list of *instructions* and *literals*. 
There is a stack for every data type, and an ```exec``` stack for instruction. 
Literals are placed onto their respective stack, and instructions are executed, taking literals from requested stacks. 
Crucially, if an instruction fails to execute, nothing will happen; it will "NOOP". 
A NOOP may occur if values are unavailable, or if the supplied values are innapropriate, etc. 
Psuedocode for the execution of a Push program is outlined below (from the [Push 3.0 description](http://faculty.hampshire.edu/lspector/push3-description.html))
```
To execute program P:
    Push P onto the EXEC stack
    LOOP until the EXEC stack is empty:
        If the first item on the EXEC stack is a single instruction 
            then pop it and execute it.
        Else if the first item on the EXEC stack is a literal 
            then pop it and push it onto the appropriate stack.
        Else (the first item must be a list) pop it and push all of the
            items that it contains back onto the EXEC stack individually,
            in reverse order (so that the item that was first in the list
            ends up on top).
```

Basically, Push is a robust representation where almost every program is valid. 
Although a Push program is somewhat incomprehensible to humans, 
it has shown to be a potent ingredient to evolutionary search. 
_PushGP_ is a genetic programming system that evolves Push programs, which has 
empirically performed well on a range of software synthesis problems. 

To learn more about Push and PushGP, take a look at the
[Push Project Page](http://faculty.hampshire.edu/lspector/push.html),
[Push Redux](https://erp12.github.io/push-redux/pages/intro_to_push/index.html),
[Push 3.0 description](http://faculty.hampshire.edu/lspector/push3-description.html), 
[Introduction to Push](https://push-language.hampshire.edu/t/introduction-to-push/794), 
etc.

# Computational Graphs
The execution of a Neural Networks can be expressed as a DAG (directed acyclic graph), where nodes are either variables or operations, both of which produce tensor values that flow along edges.
Although not all Neural Networks behave consistently; 
static Neural Networks will invariably produce the same computational graph irrespective of their inputs, 
whereas dynamic Neural Networks may produce a different computational graphs for different inputs. 
The computational graph of a dynamic neural network is defined at runtime, in a "define-by-run" manner.

Ultimately, a graph of tensor operations is extracted from the trace of the execution of a program. 
For static neural networks, the graph structure is defined before execution, ie "define-then-run". 
Thus, a static graph can be written as a closed form expression. 
For dynamic neural networks, the structure of the graph depends on the input; thus the definition of the neural network is inextricably tied to the program.

Although a single execution of a dynamic Neural Network can be expressed as a graph of tensor operations, it is not to say that it does not contain non-tensor operations and datastructures;
they may be used to organize computation (eg functions) or bring inputs into a tensor form (eg word2vec, other preprocessing). 
Further, Neural Networks always use non-tensor operations: data wrangling, algorithms to interface with the input/output of the network, and the eventual usage of the network(s) will be part of a larger system/agent architecture.

To backpropagate through a dynamic neural network, one simply constructs the computational graph of tensor operations from a particular input, and backpropagates through that instance's graph. A visualization of this process is shown below.
Although dynamic Neural Networks are strictly more general, they forego optimizations that static graphs enjoy; because they lack the prescience of static graphs, they can't transform the graph (eg pruning) or schedule/stage execution as effectively.
<p align="center"> <img src='http://pytorch.org/static/img/dynamic_graph.gif' width=500px></p>

# Tush
Tush is a Push language that includes tensor operations and a tensor stack. Additionally, Tush also includes a "shape" stack (and related operations) for handling tensor shapes. 
The execution of a Tush program is like that of a Push program. Recall that invalid operations (including tensor operations) will simply result in a NOOP. 
When the EXEC stack is empty, execution terminates, resulting in a collection of stacks which the output may be extracted from.

Because Tush contains both tensor operations and non-tensor operations, it may be used like a conventional Push language. 
In this case, one generally evaluates the fitness of the program based on the top value of the appropriate stack (or a default value if the stack is empty).
However, because the output must be of a certain type and shape, a more sophistacated method is used to extract the top value:
<details>
 <summary>Sophisticated Method for Extracting Tensor from Tensor Stack</summary>
If tensor stack is empty, return default (usually tensor of all ones)
    
Else, descend tensor stack until requested shape "fits" in tensor's shape.
This tensor must also be of the correct (or cast-able) data type, and must be a descendent of an input tensor and a tensor that is being learned by gradient descent.

The descendent constraint exists so that the output is dependent on the input and values that will be learned in the "experiential learning" phase. If it were not dependent on the input, the output would be constant, and converge to the expectation. If it were not dependent on a variable, there is no need for experiential learning (and no variables will be updated).

If no such tensor exists, the descendent constraint is dropped. 

If no such tensor exists, even without the descendent constraint, the default value is returned.

This method is ad-hoc, untested, and may change. Source can be found [here](https://github.com/julianoks/Tush/blob/master/tush.py#L87).
</details>


Unlike Push, the fitness of a Tush program is not immediately evaluated. 
Trainable tensor values are first initialized and then refined during an "experiential" learning phase. 
First, "blueprints", which are themselves Tush programs that do not contain other blueprints, are evaluated in a seperate environment and the top value on the tensor stack replaces the blueprint and is marked as trainable.
Next, the trainable tensor values are learned via gradient descent during a training phase. 
Finally, the fitness of the Tush program is evaluated using validation data.

A loss function is provided by the user, which evaluates the extracted tensor. 
Backpropagation starts from the loss, and trainable variables are updated. 
Fortunately, the computational graph is implicitly constructed by the [Pytorch](http://pytorch.org/about/) system, which keeps track of the trace/graph of tensor values; Tush is not responsible for any bookkeeping.

Although Push execution is sometimes understood as a series of instruction executions, it is better understood as a state machine. Another interpretation is modeling the execution as a static single assignment form (SSA). It is perhaps more intuititive to understand the derivation of a computational graph from a Tush execution when the execution is modeled as SSA.

# Future
Tush is still in development, and we look forward to running our first experiments (very soon). 
For our first experiment, we'd like to evolve simple classifiers, then approach more complicated problems, like those involving data structures, preprocessing, or multiple datasets.

We've also identified several areas for future research. First, we're interested in creating some mechanism to encourage modularity. Second, we'd like to allow a program to have more control over training (eg define its own loss(es) and optimization routines).
