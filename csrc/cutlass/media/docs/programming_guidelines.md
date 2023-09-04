![ALT](/media/images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Programming Guidelines")

[README](/README.md#documentation) > **Programming Guidelines**

# Programming Guidelines

## Hierarchical Organization

The [CUTLASS 3.0 GEMM API](./gemm_api_3x.md) document
explains CUTLASS 3.0's hierarchical organization,
based conceptually on parallelization strategy.
This differs from CUTLASS 2.x's approach,
which more closely mirrors the GPU hardware hierarchy
of thread blocks, warps, and threads.

## Design Patterns

CUTLASS aims for the highest performance possible on NVIDIA GPUs.
It also offers flexible components that can be assembled and customized
to solve new problems related to deep learning and linear algebra.
Given a tradeoff between simplicity and performance,
CUTLASS chooses performance.
Consequently, several design patterns are necessary
to yield a composable structure
while also satisfying these performance objectives.

### Templates

CUDA C++ templates and modern generic programming techniques enable CUTLASS device code to span a large design space.

This design space includes:
* Mixed precision arithmetic and data storage
* Kernels specialized for layout and problem size
* Support for kernel fusion

Moreover, templates provided a structured approach to collecting compile-time constants such as tile dimensions. These
must be template arguments to target static array allocation and take advantage of loop unrolling, constant folding,
and function inlining.

### Constant Memory

Several CUTLASS template classes exhibit a pattern in which problem-specific internal state is known at kernel 
launch time and remains invariant throughout the execution of a kernel. For example, tile iterators compute several 
offsets based on the strides of the input tensor that is added to an internal pointer when loading the elements 
of a tile. These are computed from the tensor stride and never updated; the per-thread internal state consists 
only of the internal global memory pointer.

CUTLASS can take advantage of this CUDA grid-invariant property by constructing the object in host code and passing 
a composed parameters structure to the kernel. This confers two benefits: (1.) invariant state is held in constant 
memory, and (2.) there is no overhead to compute the initial state by each thread.

The design pattern in CUTLASS is for classes with nontrivial constructors to define `struct Params` as an inner class 
which contains grid-invariant state. These should define a constructor and an `initialize()` method. The `Params` 
structure should also include a data member corresponding to each data member in the parent class, so these too can 
be properly constructed in host code. The parent class should define a constructor which accepts `Params const &` as 
its first argument.


### Composable Shared Memory

Shared memory requires explicit effort by the programmer to allocate and de-allocate. CUTLASS follows the paradigm 
introduced by [CUB](https://nvlabs.github.io/cub/) to define composed structures for storing data intended to be held 
in shared memory. Any object requiring shared memory storage for itself or its data members should define a child 
structure called `SharedStorage`. This holds data needed by the class and also instantiates `SharedStorage` 
objects for each data member.

To be consistent, this pattern defines a convention in which classes define internal shared memory storage requirements. 
Classes should consider all SharedStorage structures to be opaque other than their own child class. When the lifetimes 
of child objects are known to be non-overlapping, `union`s may be used to alias multiple SharedStorage objects to the same
shared memory region and reduce overall shared memory capacity.  Developers should carefully note that C++ `union` rules
require that they only access the most recently written ("active") member of the `union`; this differs from C rules.

### Loop Unrolling

CUTLASS requires tiles of data to be stored in registers for high-bandwidth access. Simultaneously, high-throughput math instructions
must be issued concurrently with memory instructions to hide latency with relatively few concurrent threads. These objectives are
achieved by unrolling loops whose iteration counts are known at compile time.

Consequently, most loops within the CUTLASS GEMM implementation are specified by constant values and template arguments. The CUDA compiler
is able to unroll the loop bodies, map array elements to registers, and construct an efficient instruction schedule.

All loops expected to be unrolled should be annotated with `CUTLASS_PRAGMA_UNROLL` to explicitly direct the compiler
to unroll them. 

```c++
int const kN = 8;
Array<float, kN> x;                       // Array we would like to store in registers

CUTLASS_PRAGMA_UNROLL                     // Directs the CUDA compiler to unroll this loop.
for (int idx = 0; idx < kN; ++idx) {      // Loop has constant number of iterations.

  x[i] = float(idx);                      // Indirect access by induction variable results in 
                                          // direct register access.
}
```

## Style

### No automatic code formatting

Do not use any kind of automatic code formatting,
like `clang-format`, on CUTLASS code.

### C++ style

#### CUTLASS is a C++ project

CUTLASS is a C++ project.  CUDA C++ is a C++ dialect.
Therefore, we write using standard C++ idioms as much as possible.
We aim for portability to as many compilers as possible,
by writing host code in Standard C++
and device code in CUDA C++
that resembles Standard C++ as much as possible.
This improves usability
for the general community of C++ developers,
and makes it easier for new staff to join the project.

#### Follow Standard C++ idioms where possible

Regarding "standard C++ idioms,"
CUTLASS source code follows the following guidelines,
with deviations only because of compiler limitations
or where performance absolutely requires it.
"Performance requires it" implies measurement.
Deviations should be limited in scope
and we should always strive to eliminate them.

* [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)

* [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

#### Spacing and line length

* Use spaces, not tabs.

* Use 2 spaces to indent.

* Max 100 characters per line.

Lines longer than 100 characters typically wrap unfavorably
when viewed in Github's pretty printer.

#### Function indentation

When calling a function or function object with a long name,
break the line right after the invoking open parenthesis.
Here is an example.

```c++
detail::very_long_function_object_name<TemplateArgument>{}(
  params.long_parameter_name, some_operator.another_long_function_name());
```

When declaring functions, indent function parameters like this.

```c++
void possibly_an_unusually_long_function_name(
  std::uint32_t foo
  std::uint32_t const* bar,
  TypeA a,
  TypeB b,
  TypeC c)
{
  // ... the function's body ...
}
```

For function definitions only,
break the line between the parenthesis
that closes the function's parameters,
and the curly bracket
that opens the function's body.

#### If-else brackets and spacing

* Always use braces with conditionals such as `if`.

* Use a space after control flow keywords
  such as `if`, `for`, and `while`.

* Use a space after the parenthesis closing a conditional
  such as `if`, and the curly bracket opening a scope.

* Use a new line between the closing brace
  of an `if` branch, and the `else` keyword.

```c++
if (condition) {
  // ... code ...
}
else {
  // ... other code ...
}

for (int k = 0; k < num_iters; ++k) {
  // ... still more code ...
}
```

#### East const

CUTLASS uses the
["East const"](http://slashslash.info/2018/02/a-foolish-consistency/)
convention.
That is, the `const` or `constexpr` keyword
goes after the type, not before.
The general rule is that `const` or `constexpr`
modifies the type to the left of it.
Here are some examples.

```c++
float constexpr compile_time_constant = 42.3f;

float const const_float = /* whatever */;
float const& reference_to_const_float = const_float;
float const* pointer_to_const_float = &const_float;
float const* const const_pointer_to_const_float = &const_float;

float nonconst_float;
float& reference_to_nonconst_float = nonconst_float;
float* pointer_to_nonconst_float = &nonconst_float;
float* const pointer_to_nonconst_float = &nonconst_float;
```

Contrast this with "West const" style, e.g.,

```c++
const float const_float = /* whatever */;
const float* pointer_to_const_float = &const_float;
```

#### Alignment of reference and pointer types

For reference and pointer types,
align the `&` resp. `*` flush against the type
that it modifies.  This is called "left alignment."

For example, do this:

```c++
int const& var;
int const* var;
```

and not this.

```c++
int const &var;
int const *var;
```

#### Avoid calling functions "fast" or "optimized"

Putting words like "fast" or "optimized"
in the name of a function
assumes that the "fast" path is actually faster.
That might be true now, but later changes
(in the code, compilers, or GPU hardware)
might make it false.  In that case,
your name could be unintentionally misleading.
Consider instead a name that briefly describes
the algorithm or feature that is relevant for optimization.
For example, `compute_on_host` is more meaningful
than `compute_slowly`, and computing on host
might be faster in some cases
(e.g., if the data are already on host
and the algorithm is not GPU-friendly).

CUTLASS code has not always followed this rule in the past.
Some functions and classes might have words like "fast" in their name.
New code should follow this rule, however.

#### Avoid creating unconstrained templated functions with common names

See [C++ Core Guidelines T.47](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#t47-avoid-highly-visible-unconstrained-templates-with-common-names):
"Avoid highly visible unconstrained templates
with common names."
Argument-dependent lookup (ADL) means that
if users call a function name without specifying the namespace,
the compiler can find overloads
of that function in any namespace.
This can lead to ambiguous overloads in users' code,
just because they happened to include one of your header files
that exposes an unconstrained function template.
The following illustrates this
with an unconstrained swap overload in the `cutlass` namespace.

```c++
#include <cassert>
#include <memory>
#include <utility>

// Uncomment the line below to observe unwarranted build errors.
//#define BAD_CUTLASS_SWAP 1

namespace cutlass {
struct Bar {
  float f;
};
} // namespace cutlass

#ifdef BAD_CUTLASS_SWAP
namespace cutlass {

template<class T>
void swap(T& a, T& b) // don't do this
{
  T tmp = a;
  a = b;
  b = tmp;
}

} // namespace cutlass
#endif // BAD_CUTLASS_SWAP

namespace other {

#ifdef BAD_CUTLASS_SWAP
using cutlass::swap;
#endif // BAD_CUTLASS_SWAP

// Imagine for the sake of this example
// that "foo" is a less common name,
// and that T is constrained via
// std::enable_if or a requires clause.
template<class T>
void foo(T& a, T& b)
{
  // The usual idiom for using std::swap is the "swap two-step":
  //
  // 1. import std::swap into the current scope, then
  // 2. call swap without namespace qualification.
  //
  // That won't build if we have another swap
  // overload available in the scope already.

  using std::swap;
  swap(a, b); // OBSERVE UNWARRANTED BUILD ERROR HERE
}

} // namespace other

int main()
{
  int x = 42;
  int y = 43;
  other::foo(x, y);
  assert(x == 43);
  assert(y == 42);

  cutlass::Bar a{42.0};
  cutlass::Bar b{43.0};
  other::foo(a, b);
  assert(a.f == 43.0);
  assert(b.f == 42.0);

  // GCC 7.5 std::unique_ptr::reset calls swap,
  // leading to the same issue as above.
  // GCC 12.2's implementation of std::unique_ptr
  // does not have this issue.  Nevertheless,
  // breaking the swap two-step will break users' code,
  // just by them happening to include your headers.
  auto ptr = std::make_unique<cutlass::Bar>(cutlass::Bar{666.0f});
  ptr.reset(new cutlass::Bar{777.0f}); // OBSERVE UNWARRANTED BUILD ERROR HERE

  return 0;
}
```

#### Function return values and in-out parameters

##### Prefer return values to output parameters

In general, avoid in-out mutable references to return a value.
If you need to return multiple values,
you can return them by `struct` or `tuple`,
rather than by output references.
This includes the special case of error reporting
by returning either a value or an error code.
Please see the next section for details.

```c++
// Instead of passing in-out mutable references ...
void not_preferred(float& input_and_output); // not preferred

// keep functions pure and return value types instead
float preferred(float input); // preferred
```

##### Return multiple values by struct or tuple

Sometimes a function needs to return multiple values.  In that case, consider the following, in decreasing order of preference.

1. Return a `struct`.  This lets you name the fields
   (for more self-documenting code),
   yet still permits use of structured binding.

2. Return a `tuple`.  If you need a tuple type
   that works on device, use `cute::tuple`.
   (Please note that `cute::tuple` does not work
   for all the types that work in `std::tuple`.
   CuTe's documentation explains.)

Here is an example of the struct approach for named values.
For a comparable example in the C++ Standard,
please see [`std::allocate_at_least`](https://en.cppreference.com/w/cpp/memory/allocate_at_least),
which returns `std::allocation_result`.

```c++
struct my_computation_result {
  float value = 0.0f;
  float relative_error = 0.0f;
  bool success = false;
};

my_computation_result my_computation(float tolerance);

void foo(float tolerance)
{
  // Approach 1: Use structured binding.  The names
  // you choose on the left-hand side have nothing
  // to do with the struct, so it's up to you
  // to get the order right.  On the other hand,
  // this code works whether my_computation returns
  // a struct or a tuple.
  auto [val, rel_err, ok] = my_computation(tolerance);

  // Approach 2: Keep the struct and use its named fields.
  // This approach prevents errors like mixing the order of return types.
  // However, it only works for structs, not for tuples.

  auto result = my_computation(tolerance);
  if (not result.success) {
    // computation did not succeed
  }
  else if (result.relative_error > tolerance) {
    // successful but relative error too large
  }
  else {
    // successful and relative error is in bounds
  }
}
```

##### Reporting errors from a function that returns one or more values

We may want to return one or more values
from a function that could fail
or otherwise report errors.
That is, the function either

* returns one or more valid values, or

* does not return any values and reports an error,

but NOT BOTH.  We contrast this with cases
when it's meaningful to report both a result
and whether the result is satisfactory.
For example, when solving
a system of nonlinear equations iteratively,
users may want the approximate computed solution,
even if the iteration did not succeed
by converging to the desired tolerance
in the desired number of steps.
(Users may want to invest more steps,
or use the current approximation
to jump-start a different algorithm.)

We're talking here about the "either valid value(s),
or error, but not both" case.
For this case, C++ offers a few options.

1. Return the value(s), or throw an exception on error

2. `std::expected` (requiring C++23) or something like it

3. `std::optional` (for a Boolean error state)
   or something like it

4. `std::variant` (a C++17 fall-back for `std::expected`)
   or something like it

5. C-style interface: return an error code,
   and "return" the values as output parameters

We usually cannot or do not want to
throw exceptions on device.
Some code projects forbid exceptions entirely
(on host or device)
and tell the compiler to disable them.
If we exclude a C-style interface (the last option)
as not idiomatic C++, then for host-only code,
`std::expected`, `std::optional`, and `std::variant`
all work.
For code that needs to build and run on device,
we can fall back to libcu++ equivalents
in the `cuda::std::` namespace, when they exist.
Otherwise, we must resort to returning a struct or tuple
with the value and the error information,
and ask users not to use the value on error.
This is acceptable if the value can be constructed
cheaply with a reasonable default.

##### Performance of different value-or-error reporting methods

[P1886R0](https://wg21.link/P1886R0)
(Ben Craig, "Error speed benchmarking")
surveys different ways in Standard C++
to report errors from a function
that returns one or more values,
and compares their (host-only) performance
with different compilers.

##### Use aggregate initialization when returning a struct or tuple

Use aggregate initialization when returning a struct or tuple.
This avoids duplication of the return type name.

```c++
struct foo_result {
  float value = 0.0f;
  float error = 0.0f;
  bool success = false;
};

foo_result foo(std::span<const float> input)
{
  // ... code  ...

  // Prefer this.  We know what type the function returns.
  return {val, err, ok}; // prefer this

  // Naming foo_result again here is unnecessary.
  // return foo_result{val, err, ok};
}
```

However, note that this won't work if the function returns `auto`.
The general rule is to avoid code duplication.

```c++
auto foo(std::span<const float> input)
{
  // ... code  ...

  if constexpr (some_condition) {
    return foo_result{val, err, ok};
  }
  else {
    return bar_result{val, err, ok};
  }
}
```

##### Prefer using the actual return type to auto, if you know the type

C++ lets you use `auto` to deduce the type returned from a function.

* If you know the actual type, prefer using the type instead of `auto`.

* Use [Constructor Type Argument Deduction](https://en.cppreference.com/w/cpp/language/class_template_argument_deduction)
  (CTAD) if you know that a function returns some type
  (e.g., `Tensor`), but don't know the type's template arguments.

* Use `auto` in structured bindings (where you have to use it anyway).  This also makes your code agnostic of whether the return type is a `struct`, `tuple`, `pair`, or other tuple-like type.

* Be careful using `auto` with types that provide expression templates.

Contrast this with "Almost Always Auto" (AAA) style.
We deliberately choose not to follow AAA style,
for the following reasons.

* Using the actual type when we know it can help prevent common loss-of-precision errors in mixed-precision computations, an important use case for CUTLASS.

* CTAD gives us much of the brevity of AAA, with more clarity.

* Using the actual type instead of `auto` can prevent common dangling errors with expression templates.

#### Classes and structs

Type names use `CamelCase`.
That is, words start with capital letters.
The remaining letters in the word are lower case,
and words are joined with no intervening underscores.
The only exception is when implementations are
a drop-in replacement for C++ Standard Library components.

Follow the
[C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-struct)
to decide whether to use `class` or `struct`.

* Use `class` when the object must maintain an invariant.
  Data members related to the invariant should be `private`.

* Use `struct` when the class has no invariant to maintain,
  and data members may vary arbitrarily with respect to each other.

Prefer nonmember functions and statelessness where possible.
Member functions imply invariants.
More invariants make code maintenance and testing harder.

#### Class members

Methods and members are written using `snake_case`.

Private data and function members have suffix `_`.

#### Class Member Order

Members within classes and structures should be organized as follows:

1. Type and constant definitions

2. Data members

3. Constructors

4. Other methods

This convention follows the
[CUB library](https://nvlabs.github.io/cub/)
and is also described by 
[Howard Hinnant](https://howardhinnant.github.io/classdecl.html).
It also approximates the usual ordering of chapters
in a typical Systems and Controls textbook.
That is, it

1. identifies relevant constants,

2. defines a state-space representation
   of the dynamical system under study
   (the class's data members), and then

3. devotes the remaining "chapters" to defining
   the system's dynamical behavior
   (the class's methods).

Here is an example class.

```c++
class A {
public:
  // type definitions
protected:
  // protected type definitions
private:
  // private type definitions

public:
  // data members
protected:
  // protected data members
  // STRONGLY TO BE AVOIDED;
  // please see C++ Core Guidelines
private:
  // private data members

public:
  // methods
protected:
  // protected methods
private:
  // private methods
};
```

#### Use scoped enums

Use scoped enums (a C++11 feature) for enumerated types.
Use capital letters for the enumerated type name
and prefix `k` for enumerators like other constants.

```c++
enum class MatrixOperation {
  kNone,
  kTranspose,
  kConjugate,
  kHermitian
};
```

#### Namespaces

Namespaces are all lower case.
The top-level namespace is `cutlass::`.
The second nested namespace refers to
the general category of operation
performed by its members: e.g., `gemm::`.
The third nested namespace refers to
the operations' position in the conceptual hierarchy:
e.g., `device::`, `kernel::`, or `collective::`.

The bodies of namespace definitions should not be indented.
Comments on the closing brace to indicate
the namespace being closed are welcome.

```c++
namespace cutlass {
namespace gemm {
namespace kernel {

struct AnotherGemmKernel {
  // ... contents ...
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
```

#### File Names

New files should be named using `snake_case`
with extension `.hpp` for header files,
`.cu` for CUDA sources,
and `.cpp` for C++ host-only source files.

Header files with extension `.h`
are CUTLASS 2.x legacy headers.

#### Macros

Only use macros when the preprocessor
is the only way to accomplish the task.
Do not use macros for literal constants.
Instead, if inside the body of a function,
use `constexpr` values,
and if at namespace scope, use
[`inline constexpr` variables](https://en.cppreference.com/w/cpp/language/inline)
(a C++17 feature).

"Namespace" macros by starting them with the module name, e.g., `CUTLASS_`.
Macros and ONLY MACROS use all capital letters with underscores between words.
For example:

```c++
#define CUTLASS_MACROS_USE_ALL_CAPS inline __host__ __device__
```

Header files such as
[cutlass/cutlass.h](../../include/cutlass/cutlass.h)
and
[cute/config.hpp](../../include/cutlass/cutlass.h)
offer macros for expressing compiler-dependent behavior.
These include

* replacements for `__device__` and/or `__host__`
  annotations:

  * `CUTLASS_HOST_DEVICE` or `CUTE_HOST_DEVICE`
    for functions that run on the host and the device,

  * `CUTLASS_DEVICE` or `CUTE_DEVICE`
    for functions that run on the device only, and

  * `CUTE_HOST`
    for functions that run on the host only; and

* annotations to loop unrolling:

  * `CUTLASS_PRAGMA_UNROLL` or `CUTE_UNROLL`
    for full unrolling of loops with constant trip counts, and

  * `CUTLASS_PRAGMA_NO_UNROLL` or `CUTE_NO_UNROLL` to prevent unrolling.

#### Guard all headers with `#pragma once`

Use `#pragma once` to guard all headers.

### CUDA C++ style

#### CUDA Built-in Variables

Avoid direct access to CUDA built-in variables `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` within
CUTLASS components except in special circumstances. 

Using built-in global variables directly within resuable components necessitates that all components
use them consistently which may not be possible if CUTLASS components are used in other contexts.

Instead, components should accept a linear ID identifying threads, warps, and threadblocks from calling
code. The top-level kernel may then decide how to map threads, warps, and blocks to the problem it is
solving.

#### Use CUTLASS's and CuTe's fundamental types and operations

Use the
[fundamental types and operations](fundamental_types.md)
defined in CUTLASS consistently.
This contributes to a framework of interoperable, consistent components.
It reduces code duplication, which reduces build and test times.
It also saves developer effort.

CUTLASS's fundamental types and operations include

* [Numeric types](fundamental_types.md#numeric-types) to represent numeric data in host and device code, and

* [functional.h](fundamental_types.md#functional) to perform numeric operations in generic code.

CUTLASS 3.0 uses CuTe components to represent data layouts and multidimensional arrays.
Please refer to the [CuTe Tutorial](./cute/00_quickstart.md) for details.
CuTe has replaced CUTLASS 2.x components such as
[Containers](fundamental_types.md#containers),
[Layouts](layout.md), and
[`TensorRef` and `TensorView`](layout.md#tensorref).

# Copyright

Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
