# Overview
Tush is a Push programming language that includes tensor operations. 
Like Push, Tush is not intended to be written by human programmers, but rather to be used as a representation for evolving programs.
A further deviation from traditional Push implementions, Tush programs may experience an "epigenetic" or "experiential" learning phase, 
where numeric values are refined by gradient descent. This document aims to introduce the Tush representation and TushGP system, 
discuss the Push family of languages, computational graphs, Tush and "experiential" learning, and finally some exciting capabilities and directions for the Tush system.

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

To backpropagate through a dynamic neural network, one simply constructs the computational graph of tensor operations from a particular input, and backpropagates through that instance's graph. 
Although dynamic Neural Networks are strictly more general, they forego optimizations that static graphs enjoy; because they lack the prescience of static graphs, they can't transform the graph (eg pruning) or schedule/stage execution as effectively.

