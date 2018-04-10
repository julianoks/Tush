# The Upshot
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
[Push Project Page](http://faculty.hampshire.edu/lspector/push.html)
[Push Redux](https://erp12.github.io/push-redux/pages/intro_to_push/index.html),
[Push 3.0 description](http://faculty.hampshire.edu/lspector/push3-description.html), 
[Introduction to Push](https://push-language.hampshire.edu/t/introduction-to-push/794), 
etc.
