# TODO
- [x] Figure out why PyTorch can't find a differentiable path, so that variables receive gradient update. Problem solved: this definitely  emanated from the `tush.get_tensor_out` function.


# Overview
Tush is a programming language used to encode differentiable tensor operations using a representation intended for evolutionary search. In this section, we will define the Tush specification and execution.


# Program Definition
A program is expressed as a list of items, where each item is a pair denoting type and value, ```(type, value)```. There are two classes of items that may be in a program: a stack item and a blueprint item:
- a **stack item** is a pair ```(type, value)``` such that `type` is the name of a stack, ie `type ∈ {'exec', 'tensor', 'shape', 'integer'}`. The `value` must be appropriate for the corresponding stack, eg if `type` is `'integer'`, `value` must be a valid value for the `'integer'` stack, so ```['integer', 3]``` would be a valid item wherease ```['integer', 3.3]``` would not be. We will define appropriate values for each stack in the stacks section.
- a **blueprint item** is a pair `(type, value)` such that `type` is `'blueprint'` and `value` is a list of stack items. The `value` is essentially a (sub)program that does not contain other (sub)programs/blueprints. We will define the usage of blueprint items in the stage 1 execution section.


An example Tush program is as follows:

```
[['integer', 4], ['size', torch.Size([4, 3])], ['tensor', torch.ones(3,5)], ['exec', 'add'], ['blueprint', [['size', torch.Size([2, 3])], ['tensor', zeros(4,5)], ['exec', 'matmul']]]]
```

