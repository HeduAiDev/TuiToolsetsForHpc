Simple Toolsets for HPC development. Depends on FTXUI, pybind11.
## Features
- support for C++ and Python
- support for very large matrix over 10000 * 10000



## Preview

<table>
  <tr>
    <td>
      <img src="./assets/largematrix.gif"/>
    </td>
    <td>
      <img src="./assets/diff.gif"/>
    </td>
    <td>
      <img src="./assets/printmatrx.gif"/>
    </td>
  </tr>
</table>

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
| `Tab`      | ChangeFocus         |
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