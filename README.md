Simple Toolsets for HPC development. Depends on FTXUI, pybind11.
## Features
- support for C++ and Python
- support for very large matrix over 10000 * 10000



## Preview

![demo of print large matrix](./assets/largematrix.gif)

![demo of diff function](./assets/diff.gif)

![demo of print_matrx function](./assets/printmatrx.gif)

## install
for python
~~~shell
git clone  --recursive git@github.com:HeduAiDev/TuiToolsetsForHpc.git
cd TuiToolsetsForHpc
pip install -e .
~~~
for C++
~~~shell
git clone  --recursive git@github.com:HeduAiDev/TuiToolsetsForHpc.git
cd TuiToolsetsForHpc
cmake -S . -B build
cmake --build build --config Release -j8
cmake --install build --config Release --component CPPInterface --prefix <your_prefix>
~~~

## usage
| Keybinding | description         |
| ---------- | ------------------- |
| `Esc`      | Exit                |
| `Enter`    | Exit                |
| j          | scroll down slowly  |
| k          | scroll up slowly    |
| h          | scroll left slowly  |
| l          | scroll right slowly |




Python
~~~shell
>>> import tui_toolsets as tui
>>> a = [[1,2,3,4]] * 2
>>> tui.print_matrix(a)
████
 0 │ 1 │ 2 │ 3 │
1.0│2.0│3.0│4.0│0█
───┼───┼───┼───┤│
1.0│2.0│3.0│4.0│1
>>>
>>> import numpy as np
>>> tui.print_matrix(np.random.randn(100,100))
█████████████████████████▌
7 │48 │49 │50 │51 │52 │53 │54 │55 │56 │57 │58 │5
──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───
0.│1.5│0.3│0.3│-1.│0.7│0.3│-1.│0.1│1.8│-0.│0.0│147
──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───
.3│0.1│1.5│0.3│-1.│0.1│-0.│2.2│0.4│0.2│-0.│0.3│048
──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───
0.│0.4│0.4│-0.│0.0│0.7│1.9│0.9│0.1│-0.│-1.│-1.│-49
──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───▄
0.│-0.│0.4│1.0│-0.│2.2│0.9│-1.│0.8│-0.│1.1│-1.│050█
──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───█
1.│0.8│-1.│-0.│-0.│0.5│-0.│1.3│0.1│-1.│-0.│0.2│-51█
──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───█
0.│-0.│-0.│0.8│0.1│0.1│0.3│0.6│0.1│-0.│1.8│-0.│052█
──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───█
>>>
>>> import torch
>>> tui.print_matrix(torch.randn(100,100))
███████████████████
 │35 │36 │37 │38 │39 │40 │41 │42 │43 │44 │45 │46
─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼────
9│0.3│-0.│1.2│-1.│-0.│0.1│0.1│0.3│-0.│-0.│-0.│0.47
─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼────
5│-0.│1.9│-0.│1.0│0.6│1.5│-0.│0.4│-0.│0.1│-0.│0.48
─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼────
.│-1.│-0.│1.6│-1.│0.2│0.0│1.6│0.7│-0.│1.5│2.4│-149
─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼────▄
5│0.8│-1.│-1.│-0.│-0.│0.7│1.0│0.5│0.6│2.6│0.7│0.50█
─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼────█
5│-1.│-0.│0.3│-1.│-0.│0.7│-2.│0.2│-1.│0.3│0.8│-051█
─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼────█
.│-0.│-2.│0.6│0.0│1.1│0.3│1.0│-0.│-1.│0.8│-0.│-052█
─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼────█
>>>
>>> a = torch.randn(20, 100)
>>> b = torch.randn(20,100)
>>> tui.diff(a,b)

// Error nums : 1792                                           │Accuracy : 0.001000
//  Max diff  : 9.000000                                       │  Rows   : 20
//  Avg diff  : 3.334500                                       │  Cols   : 100
//                                                             │
// ────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────
// ╭─────────────────────────Matrix A─────────────────────────╮│╭────────────────────────Matrix B─────────────────────────╮
// │███████████████████████████                               │││██████████████████████████▌                              │
// │ │44 │45 │46 │47 │48 │49 │50 │51 │52 │53 │54 │55 │56 │    │││ │44 │45 │46 │47 │48 │49 │50 │51 │52 │53 │54 │55 │56     │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──  ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────  │
// │.│0.0│-5.│-4.│-1.│0.0│1.0│-3.│-5.│-2.│0.0│1.0│1.0│1.0│5   │││.│1.0│-4.│-5.│0.0│0.0│-3.│0.0│3.0│-2.│-3.│2.0│3.0│0.05   │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──  ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────  │
// │.│0.0│-3.│3.0│-5.│2.0│-3.│-3.│0.0│4.0│-4.│-5.│3.0│0.0│6   │││0│-1.│-1.│-1.│-2.│0.0│3.0│-5.│-3.│4.0│-1.│4.0│3.0│-2.6   │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──  ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────  │
// │.│1.0│3.0│4.0│-3.│4.0│0.0│4.0│-1.│0.0│-5.│0.0│3.0│3.0│7   │││.│4.0│-4.│3.0│-5.│-1.│2.0│2.0│-1.│-4.│-4.│4.0│3.0│-4.7   │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──  ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────  │
// │.│0.0│1.0│1.0│-4.│-1.│-1.│1.0│-3.│1.0│3.0│-3.│0.0│0.0│8   │││.│2.0│-2.│-5.│2.0│-2.│-5.│2.0│-2.│1.0│-1.│1.0│2.0│-1.8   │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──  ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────  │
// │0│0.0│0.0│-5.│4.0│-3.│3.0│-3.│-1.│0.0│-5.│-1.│-1.│-5.│9   │││.│-4.│2.0│0.0│-2.│-4.│-5.│-5.│-5.│0.0│-5.│-5.│2.0│3.09   │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──▄ ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────▄ │
// │0│-5.│-2.│1.0│-3.│-1.│0.0│3.0│3.0│-4.│-3.│-4.│-1.│4.0│10█ │││.│4.0│3.0│4.0│1.0│-1.│-5.│-3.│4.0│-5.│-2.│-3.│-3.│-1.10█ │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──█ ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────█ │
// │0│-5.│-2.│-5.│-2.│0.0│-5.│2.0│1.0│2.0│3.0│-2.│1.0│-1.│11█ │││.│-4.│0.0│-1.│-5.│1.0│1.0│3.0│-4.│-2.│-2.│-4.│-5.│0.011█ │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──█ ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────█ │
// │0│3.0│-3.│-3.│2.0│3.0│0.0│3.0│-2.│4.0│-1.│-1.│1.0│4.0│12█ │││.│3.0│2.0│4.0│-1.│-1.│-2.│0.0│-1.│-1.│-3.│2.0│2.0│0.012█ │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──█ ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────█ │
// │.│4.0│0.0│4.0│2.0│-3.│-5.│-3.│3.0│1.0│2.0│-5.│3.0│-4.│13█ │││0│-5.│-3.│-1.│-1.│-2.│-2.│1.0│2.0│1.0│1.0│-4.│-1.│3.013█ │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──█ ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────█ │
// │0│-2.│3.0│-3.│-2.│-3.│-4.│-5.│-1.│-5.│2.0│0.0│4.0│-5.│14█ │││.│-1.│-2.│4.0│3.0│0.0│-2.│3.0│-4.│-3.│-2.│3.0│4.0│3.014█ │
// ├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼──█ ││├─┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────█ │
~~~

C++
~~~cpp
#include <stdio.h>
#include "include/tui_tool_sets_runable.hpp"
// #pragma comment(lib, "./lib/tui_tool_sets.lib")

int main() {
    int rows = 50;
    int cols = 50;
    float* a = new float[rows * cols];
    float* b = new float[rows * cols];
    a[rows + cols] = -1.f;
    tui::runable::print_matrix(a, rows, cols);
    // tui::runable::diff(a, b, rows, cols);
    return 0;
}
~~~