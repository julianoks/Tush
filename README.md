# TODO
- [x] Figure out why PyTorch can't find a differentiable path, so that variables receive gradient update. Problem solved: this definitely  emanated from the `tush.get_tensor_out` function.


# Overview
Tush is a programming language used to encode differentiable tensor operations using a representation intended for evolutionary search. In this document we will define the Tush specification and execution.


# Program Definition
A program is expressed as a list of items, where each item is a pair denoting type and value, ```(type, value)```. There are two classes of items that may be in a program: a stack item and a blueprint item:
- a **stack item** is a pair ```(type, value)``` such that `type` is the name of a stack, ie `type ∈ {'exec', 'tensor', 'shape', 'integer'}`. The `value` must be appropriate for the corresponding stack, eg if `type` is `'integer'`, `value` must be a valid value for the `'integer'` stack, so ```['integer', 3]``` would be a valid item wherease ```['integer', 3.3]``` would not be. We will define appropriate values for each stack in the stacks section.
- a **blueprint item** is a pair `(type, value)` such that `type` is `'blueprint'` and `value` is a list of stack items. The `value` is essentially a (sub)program that does not contain other (sub)programs/blueprints. We will define the usage of blueprint items in the stage one execution section.


An example Tush program is as follows:

```
[['integer', 4], ['size', torch.Size([4, 3])], ['tensor', torch.ones(3,5)], ['exec', 'add'], ['blueprint', [['size', torch.Size([2, 4])], ['tensor', zeros(4,5)], ['exec', 'matmul']]]]
```



To generate a program, we require the number of items to produce, along with the probability of producing each item. This functionality is in the [`program_generator`](https://github.com/julianoks/Tush/blob/master/programmer.py#L4) class.

To initialize a program generator, we pass as an argument the probabilies for producing items whose types belong to the `exec`, `integer`, and `blueprint` stacks. Currently, we don't produce program with items for the `shape` or `tensor` stacks. Although in the stage one execution section, we will describe that `blueprint` s produce tensors.


When generating a program, once an item's type is selected, we produce the item's value according to:
- `integer` is generated using such a generator: ```lambda : 2**random.randint(0,6)```
- `exec` is produced by sampling instructions proportional to their [instruction probabilities](https://github.com/julianoks/Tush/blob/master/instructions.py#L124)
- `blueprint` is produced using the same logic to generate a program, but does not contain other blueprints. Each blueprint contains 15-25 items, as defined in [`blueprint_size`](https://github.com/julianoks/Tush/blob/master/programmer.py#L10)


An example of producing a program with 50 items:
```
generator = program_generator({'exec': 5, 'integer': 2, 'blueprint': 3}})
program = generator.generate_program(50)
```


# Execution of Tush Program
A Tush program has three stages:
- **stage one** maps a Tush program to a collection of stacks, and replaces blueprints with tensor Variables
- **stage two** is a training phase that optimizes some Variables
- **stage three** the stacks have been optimized, and the program can be queried

### Querying Program for Output


### Stages

#### Stage One
The goal of stage one is to map a Tush program to a collection of stacks.

First, we map each blueprint to a tensor Variable. This is done by executing the blueprint's value (which is a program) in a "fresh" stack, and picking the top tensor off the tensor stack. It is then marked as a Variable to be optimized.

At this point, we have a list of items, without blueprints. We finally place each item's `value` on its corresponding stack, as marked by the item's `type`. Along with the value, we also indicate whether it is dependent on an input, and whether it's dependent on a Variable. Both of these indications are set to false during this stage, except for Variables produced from blueprints, which are marked as dependent on a Variable.

#### Stage Two