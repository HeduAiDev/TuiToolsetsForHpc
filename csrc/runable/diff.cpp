#include "tui_tool_sets.hpp"
#ifdef __CUDA__
#include <cuda_fp16.h>
#endif
#include <limits>

namespace tui {
    namespace runable {
        using namespace ftxui;
        using namespace tui::component;

        template <typename T>
        void print_matrix_(T *ptr, int rows, int cols, int screen_size_x, int screen_size_y) {
            ::tui::component::MatrixFrameOptions<T> options;
            Component frame = ::tui::component::MatrixFrame(ptr, rows, cols, options);
            Component main_renderer =  Renderer(frame,[&] {
                auto terminal = Terminal::Size();
                return frame -> Render() | size(HEIGHT, LESS_THAN, terminal.dimy < screen_size_y ? terminal.dimy -1 : screen_size_y) | size(WIDTH, EQUAL, screen_size_x);
            });
            ScreenInteractive::TerminalOutput().Loop(main_renderer);
        }
        
        template <typename T>
        // a,b is default row-major
        void diff_(T *ptr_a, T *ptr_b, int rows, int cols, double accuracy = 1e-3) {
            // Processing a large amount of data on the CPU can be very time-consuming
            printf("start computing ...\n");
            double avg_diff = 0;
            double max_diff = std::numeric_limits<double>::lowest();
            std::vector<std::pair<int,int>> err_indices;
            for (int i = 0; i < rows * cols; ++i)
            {
                double diff = std::abs(static_cast<double>(ptr_a[i] - ptr_b[i]));
                if (diff >= accuracy) {
                    err_indices.push_back({i / cols, i % cols}); // row, col 
                }
                max_diff = std::max(max_diff, diff);
                avg_diff += diff;
            }
            if (err_indices.size() == 0) {
                avg_diff  = 0;
            } else {
                avg_diff /= err_indices.size();
            }
            Component block1 = Renderer([&] {
                return gridbox({
                    {text("Error nums") | bold | hcenter | color(Color::Gold3Bis), text(" : ") | bold, text(std::to_string(err_indices.size())) | bold | color(Color::Red1)},
                    {text("Max diff") | bold | hcenter | color(Color::Gold3Bis), text(" : ") | bold, text(std::to_string(max_diff)) | bold | (max_diff > accuracy ? color(Color::Red1) : color(Color::Green1))},
                    {text("Avg diff") | bold | hcenter | color(Color::Gold3Bis), text(" : ") | bold, text(std::to_string(avg_diff)) | bold | (max_diff > accuracy ? color(Color::Red1) : color(Color::Green1))},
                });
            });

            Component block2 = Renderer([&] {
                return gridbox({
                    {text("Accuracy") | bold | hcenter | color(Color::Gold3Bis), text(" : ") | bold, text(std::to_string(accuracy))},
                    {text("Rows") | bold | hcenter | color(Color::Gold3Bis), text(" : ") | bold, text(std::to_string(rows))},
                    {text("Cols") | bold | hcenter | color(Color::Gold3Bis), text(" : ") | bold, text(std::to_string(cols))},
                });
            });

            MatrixFrameOptions<T> matrixAB_options;
            for (auto [row, col] : err_indices) {
                matrixAB_options.label_marks.push_back({"row", row, Color::Red1});
                matrixAB_options.label_marks.push_back({"col", col, Color::Red1});
                matrixAB_options.element_style_stack.push_back(MatrixFrameOptionsCommonElementStyle::mark_point(row, col, Color::Red1));
            }
            Component matrixA = ::tui::component::MatrixFrame(ptr_a, rows, cols, matrixAB_options);
            auto matrixA_r = Renderer(matrixA, [&] {
                return window(text("Matrix A") | hcenter | bold, matrixA -> Render());
            });
            
            Component matrixB = ::tui::component::MatrixFrame(ptr_b, rows, cols, matrixAB_options);
            auto matrixB_r = Renderer(matrixB, [&] {
                return window(text("Matrix B") | hcenter | bold, matrixB -> Render());
            });
            int base_y = 9;
            Resizable4BlockOptions options;
            options.base_y = &base_y;
            if (rows * cols > 2000) {
                options.placeholder_block3 = text("Redraw matrix is inefficient") | center | bold;
                options.placeholder_block4 = text("Redraw matrix is inefficient") | center | bold;
            }
            auto screen = ScreenInteractive::Fullscreen();
            Component main_renderer = Resizable4Block(block1, block2, matrixA_r, matrixB_r, screen, options);
            main_renderer |= CatchEvent([&](Event event) {
                if (event == Event::Character('q') || event == Event::Escape || event == Event::Return) {
                    screen.ExitLoopClosure()();
                    return true;
                }
                return false;
            });
            screen.Loop(main_renderer);
        }

        void diff(float *ptr_a, float *ptr_b, int rows, int cols, double accuracy) {
            diff_<float>(ptr_a, ptr_b, rows, cols, accuracy);
        }
        void diff(double *ptr_a, double *ptr_b, int rows, int cols, double accuracy) {
            diff_<double>(ptr_a, ptr_b, rows, cols, accuracy);
        }
        void diff(int *ptr_a, int *ptr_b, int rows, int cols, double accuracy) {
            diff_<int>(ptr_a, ptr_b, rows, cols, accuracy);
        }
        #ifdef __CUDA__
        void diff(half *ptr_a, half *ptr_b, int rows, int cols, double accuracy) {
            diff_<half>(ptr_a, ptr_b, rows, cols, accuracy);
        }
        #endif
       
    }
}