#include "tui_tool_sets.hpp"
#ifdef __CUDA__
#include <cuda_fp16.h>
#endif

namespace tui {
    namespace runable {
        using namespace ftxui;
        using namespace tui::component;

        template <typename T>
        void print_matrix_(T *ptr, int rows, int cols, int screen_size_x, int screen_size_y, int text_width = 5) {
            ::tui::component::MatrixFrameOptions<T> options;
            options.text_width = text_width;
            Component frame = ::tui::component::MatrixFrame(ptr, rows, cols, options);
            Component main_renderer =  Renderer(frame,[&] {
                auto terminal = Terminal::Size();
                return frame -> Render() | size(HEIGHT, LESS_THAN, terminal.dimy < screen_size_y ? terminal.dimy -1 : screen_size_y) | size(WIDTH, EQUAL, screen_size_x);
            });
            auto screen = ScreenInteractive::TerminalOutput();
            main_renderer |= CatchEvent([&](Event event) {
                if (event == Event::Character('q') || event == Event::Escape || event == Event::Return) {
                    screen.ExitLoopClosure()();
                    return true;
                }
                return false;
            });
            screen.Loop(main_renderer);
        }
        
        void print_matrix(float *ptr, int rows, int cols, int screen_size_x, int screen_size_y, int text_width) {
            print_matrix_<float>(ptr, rows, cols, screen_size_x, screen_size_y, text_width);
        }

        void print_matrix(double *ptr, int rows, int cols, int screen_size_x, int screen_size_y, int text_width) {
            print_matrix_<double>(ptr, rows, cols, screen_size_x, screen_size_y, text_width);
        }

        void print_matrix(int *ptr, int rows, int cols, int screen_size_x, int screen_size_y, int text_width) {
            print_matrix_<int>(ptr, rows, cols, screen_size_x, screen_size_y, text_width);
        }

        #ifdef __CUDA__
        void print_matrix(half *ptr, int rows, int cols, int screen_size_x, int screen_size_y, int text_width) {
            print_matrix_<half>(ptr, rows, cols, screen_size_x, screen_size_y, text_width);
        }
        #endif

       template <typename T>
        void print_matrix_glance_(T *ptr, int rows, int cols, int row_id, int col_id, int screen_size_x, int screen_size_y) {
            ::tui::component::MatrixFrameOptions<T> options;
            
            float offset_x = static_cast<float>(col_id)/cols;
            float offset_y = static_cast<float>(row_id)/rows;
            options.focus_x = &offset_x;
            options.focus_y = &offset_y;
            options.point_style_map[{col_id, row_id}] = tui::component::MatrixFrameOptionsCommonElementStyle::mark_point(row_id, col_id, Color::GrayDark);
            options.col_label_style_map[col_id] = {col_id, Color::Red1};
            options.row_label_style_map[row_id] = {row_id, Color::Red1};
            Component main_renderer =  ::tui::component::MatrixFrame(ptr, rows, cols, options);
            auto screen = Screen::Create(Dimension::Fixed(screen_size_x), Dimension::Fixed(screen_size_y));
            Render(screen, main_renderer -> Render());
            screen.Print();
        };

        void print_matrix_glance(float *ptr, int rows, int cols, int row_id, int col_id, int screen_size_x, int screen_size_y) {
            print_matrix_glance_<float>(ptr, rows, cols, row_id, col_id, screen_size_x, screen_size_y);
        }

        void print_matrix_glance(double *ptr, int rows, int cols, int row_id, int col_id, int screen_size_x, int screen_size_y) {
            print_matrix_glance_<double>(ptr, rows, cols, row_id, col_id, screen_size_x, screen_size_y);
        }
        void print_matrix_glance(int *ptr, int rows, int cols, int row_id, int col_id, int screen_size_x, int screen_size_y) {
            print_matrix_glance_<int>(ptr, rows, cols, row_id, col_id, screen_size_x, screen_size_y);
        }
        #ifdef __CUDA__
        void print_matrix_glance(half *ptr, int rows, int cols, int row_id, int col_id, int screen_size_x, int screen_size_y) {
            print_matrix_glance_<half>(ptr, rows, cols, row_id, col_id, screen_size_x, screen_size_y);
        }
        #endif
    }
}